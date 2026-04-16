"""
verification/entity_count_verifier.py
职责: Shot 1 生成后验证实体数量是否与 prompt 中预期一致

核心问题:
  Shot 1 是纯 T2V 生成，没有参考图约束。如果 prompt 描述 "三个人"，
  但模型生成了四个人，这个错误会传播到所有后续 shot（因为后续 shot
  都基于 Shot 1 的 grounding 结果作为 reference）。

解决方案:
  1. LLM 解析 prompt 时，提取每类实体的预期数量
  2. T2V 生成后，用检测模型统计实际生成的实体数量
  3. 如果预期 != 实际，则 retry（换 seed 或增强 prompt）
  4. 最多 retry N 次，超出则警告用户

适用场景:
  - 主要针对 Shot 1（纯 T2V 生成，无 reference 约束）
  - 也可用于后续 shot 中新实体首次出现的情况
"""

import os
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


class VerificationStatus(Enum):
    """验证状态"""
    PASSED = "passed"           # 数量匹配
    COUNT_MISMATCH = "count_mismatch"  # 数量不匹配
    DETECTION_FAILED = "detection_failed"  # 检测失败
    SKIPPED = "skipped"         # 跳过验证（非 T2V 模式）


@dataclass
class EntityCountExpectation:
    """单个实体类型的预期数量"""
    entity_type: str            # "person", "object", etc.
    expected_count: int         # 预期数量
    description: str            # 描述，如 "three men"
    tolerance: int = 0          # 容忍误差，默认为 0（精确匹配）
    is_critical: bool = True    # 是否为关键检查（人数通常是关键的）


@dataclass
class VerificationResult:
    """验证结果"""
    status: VerificationStatus
    expected_counts: Dict[str, int]   # {entity_type: expected_count}
    actual_counts: Dict[str, int]     # {entity_type: actual_count}
    mismatches: List[str]             # 不匹配的实体类型列表
    details: Dict = field(default_factory=dict)  # 详细信息

    @property
    def passed(self) -> bool:
        return self.status == VerificationStatus.PASSED

    def summary(self) -> str:
        """生成验证摘要"""
        if self.passed:
            return f"验证通过: {self.actual_counts}"
        elif self.status == VerificationStatus.COUNT_MISMATCH:
            lines = ["数量不匹配:"]
            for etype in self.mismatches:
                exp = self.expected_counts.get(etype, "?")
                act = self.actual_counts.get(etype, "?")
                lines.append(f"  - {etype}: 预期 {exp}, 实际 {act}")
            return "\n".join(lines)
        elif self.status == VerificationStatus.DETECTION_FAILED:
            return f"检测失败: {self.details.get('error', 'unknown')}"
        else:
            return f"状态: {self.status.value}"


class EntityCountVerifier:
    """
    实体数量验证器

    用法:
        verifier = EntityCountVerifier(grounder=grounder)

        # 从 parse_result 中提取预期数量
        expectations = verifier.extract_expectations(parse_result)

        # 生成视频后验证
        result = verifier.verify(video_path, expectations)

        if not result.passed:
            # retry with different seed
            ...
    """

    # 用于计数的通用检测 query（覆盖各种人物描述）
    PERSON_DETECTION_QUERIES = [
        "person",
        "man",
        "woman",
        "people",
        "human",
    ]

    def __init__(
        self,
        grounder=None,
        detection_threshold: float = 0.35,
        iou_dedup_threshold: float = 0.5,
        device: str = "cuda",
    ):
        """
        Args:
            grounder: VideoGrounder 实例（可选，会自动创建）
            detection_threshold: 检测置信度阈值
            iou_dedup_threshold: 同类检测去重的 IoU 阈值
            device: 设备
        """
        self.grounder = grounder
        self.detection_threshold = detection_threshold
        self.iou_dedup_threshold = iou_dedup_threshold
        self.device = device
        self._lazy_grounder = None

    def _get_grounder(self):
        """懒加载 grounder"""
        if self.grounder is not None:
            return self.grounder
        if self._lazy_grounder is None:
            from visual_grounding.grounder import VideoGrounder
            self._lazy_grounder = VideoGrounder(
                device=self.device,
                box_threshold=self.detection_threshold,
            )
        return self._lazy_grounder

    def extract_expectations_from_entities(
        self,
        entities: List,  # List[Entity]
    ) -> List[EntityCountExpectation]:
        """
        从解析出的实体列表中提取数量预期

        逻辑:
          - 统计每种类型的实体数量
          - character 类型的数量是关键检查项
          - object/location 通常不做严格数量验证

        Args:
            entities: EntityParser 解析出的实体列表

        Returns:
            预期数量列表
        """
        expectations = []

        # 统计 character 类型
        characters = [e for e in entities if e.type == "character"]
        if characters:
            expectations.append(EntityCountExpectation(
                entity_type="person",
                expected_count=len(characters),
                description=f"{len(characters)} character(s): " +
                           ", ".join(e.entity_id for e in characters),
                tolerance=0,  # 人数必须精确
                is_critical=True,
            ))

        # 统计需要 grounding 的 object
        objects = [
            e for e in entities
            if e.type == "object" and e.grounding_priority in ("high", "medium")
        ]
        if objects:
            expectations.append(EntityCountExpectation(
                entity_type="object",
                expected_count=len(objects),
                description=f"{len(objects)} object(s)",
                tolerance=1,  # 物体允许 ±1 误差
                is_critical=False,
            ))

        return expectations

    def verify(
        self,
        video_path: str,
        expectations: List[EntityCountExpectation],
        sample_frames: int = 3,
        temp_dir: Optional[str] = None,
    ) -> VerificationResult:
        """
        验证生成的视频中实体数量是否符合预期

        Args:
            video_path: 视频路径
            expectations: 预期数量列表
            sample_frames: 采样帧数（会取中位数作为最终结果）
            temp_dir: 临时目录（存放采样帧）

        Returns:
            验证结果
        """
        if not expectations:
            return VerificationResult(
                status=VerificationStatus.SKIPPED,
                expected_counts={},
                actual_counts={},
                mismatches=[],
                details={"reason": "no expectations provided"},
            )

        # 构建预期字典（只取 critical 的）
        expected_counts = {
            e.entity_type: e.expected_count
            for e in expectations if e.is_critical
        }

        if not expected_counts:
            return VerificationResult(
                status=VerificationStatus.SKIPPED,
                expected_counts={},
                actual_counts={},
                mismatches=[],
                details={"reason": "no critical expectations"},
            )

        # 采样帧并检测
        try:
            actual_counts = self._detect_and_count(
                video_path,
                list(expected_counts.keys()),
                sample_frames,
                temp_dir,
            )
        except Exception as e:
            return VerificationResult(
                status=VerificationStatus.DETECTION_FAILED,
                expected_counts=expected_counts,
                actual_counts={},
                mismatches=[],
                details={"error": str(e)},
            )

        # 比较预期和实际
        mismatches = []
        for etype, expected in expected_counts.items():
            actual = actual_counts.get(etype, 0)
            tolerance = next(
                (e.tolerance for e in expectations if e.entity_type == etype), 0
            )
            if abs(actual - expected) > tolerance:
                mismatches.append(etype)

        status = (
            VerificationStatus.PASSED if not mismatches
            else VerificationStatus.COUNT_MISMATCH
        )

        return VerificationResult(
            status=status,
            expected_counts=expected_counts,
            actual_counts=actual_counts,
            mismatches=mismatches,
            details={
                "sample_frames": sample_frames,
                "video_path": video_path,
            },
        )

    def _detect_and_count(
        self,
        video_path: str,
        entity_types: List[str],
        sample_frames: int,
        temp_dir: Optional[str],
    ) -> Dict[str, int]:
        """
        在视频帧中检测并统计实体数量

        策略:
          - 采样多帧，每帧独立检测
          - 对每帧的检测结果做 IoU 去重（同一个人可能被多次检测）
          - 取各帧检测数量的众数（mode）作为最终结果
        """
        import cv2
        import numpy as np
        from collections import Counter

        # 采样帧
        if temp_dir is None:
            temp_dir = f"/tmp/verify_{os.path.basename(video_path).split('.')[0]}"
        os.makedirs(temp_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            raise ValueError(f"无法读取视频帧: {video_path}")

        # 均匀采样
        sample_indices = np.linspace(
            total_frames * 0.1,  # 跳过开头 10%
            total_frames * 0.9,  # 跳过结尾 10%
            sample_frames,
            dtype=int,
        )

        frame_paths = []
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                path = os.path.join(temp_dir, f"verify_frame_{idx:06d}.jpg")
                cv2.imwrite(path, frame)
                frame_paths.append(path)
        cap.release()

        if not frame_paths:
            raise ValueError("采样帧失败")

        # 每种类型分别检测
        results_per_type = {etype: [] for etype in entity_types}
        grounder = self._get_grounder()

        for etype in entity_types:
            if etype == "person":
                # 人物检测：使用通用 query
                frame_counts = []
                for fp in frame_paths:
                    count = self._count_persons_in_frame(grounder, fp)
                    frame_counts.append(count)
                results_per_type[etype] = frame_counts
            else:
                # 其他类型暂不实现详细检测
                results_per_type[etype] = [0] * len(frame_paths)

        # 取众数作为最终计数
        final_counts = {}
        for etype, counts in results_per_type.items():
            if counts:
                counter = Counter(counts)
                final_counts[etype] = counter.most_common(1)[0][0]
            else:
                final_counts[etype] = 0

        return final_counts

    def _count_persons_in_frame(self, grounder, frame_path: str) -> int:
        """
        统计单帧中的人数

        策略:
          - 使用 "person" 作为检测 query
          - 对检测结果做 NMS 去重
        """
        from groundingdino.util.inference import predict, load_image
        import numpy as np

        grounder._load_gdino()
        image_source, image_tensor = load_image(frame_path)

        # 使用 "person" 检测
        boxes, logits, phrases = predict(
            model=grounder._gdino_model,
            image=image_tensor,
            caption="person",
            box_threshold=self.detection_threshold,
            text_threshold=0.25,
        )

        if len(boxes) == 0:
            return 0

        # NMS 去重
        h, w = image_source.shape[:2]
        boxes_xyxy = []
        scores = []

        for i, box in enumerate(boxes):
            cx, cy, bw, bh = box.tolist()
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            boxes_xyxy.append([x1, y1, x2, y2])
            scores.append(logits[i].item())

        # 简单 NMS
        kept = self._nms(boxes_xyxy, scores, self.iou_dedup_threshold)

        return len(kept)

    @staticmethod
    def _nms(boxes: List[List[int]], scores: List[float], iou_threshold: float) -> List[int]:
        """Non-Maximum Suppression"""
        if not boxes:
            return []

        import numpy as np

        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        order = scores.argsort()[::-1]
        kept = []

        while order.size > 0:
            i = order[0]
            kept.append(i)

            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return kept


def build_count_enhanced_prompt(original_prompt: str, person_count: int) -> str:
    """
    构建带有人数强调的增强 prompt

    当 T2V 生成的人数不对时，retry 时使用更明确的 prompt

    Args:
        original_prompt: 原始 prompt
        person_count: 预期人数

    Returns:
        增强后的 prompt
    """
    count_words = {
        1: "one single",
        2: "exactly two",
        3: "exactly three",
        4: "exactly four",
        5: "exactly five",
        6: "exactly six",
    }

    count_desc = count_words.get(person_count, f"exactly {person_count}")

    # 在 prompt 开头添加明确的人数约束
    enhanced = (
        f"[IMPORTANT: This scene contains {count_desc} people, no more, no less.]\n\n"
        f"{original_prompt}"
    )

    return enhanced


# ── 快速测试 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("用法: python entity_count_verifier.py <video_path> <expected_person_count>")
        print("示例: python entity_count_verifier.py shot_001.mp4 3")
        sys.exit(1)

    video_path = sys.argv[1]
    expected_count = int(sys.argv[2])

    verifier = EntityCountVerifier()
    expectations = [
        EntityCountExpectation(
            entity_type="person",
            expected_count=expected_count,
            description=f"{expected_count} person(s)",
            tolerance=0,
            is_critical=True,
        )
    ]

    result = verifier.verify(video_path, expectations)
    print(f"\n验证结果: {result.status.value}")
    print(result.summary())
