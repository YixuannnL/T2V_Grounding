"""
visual_grounding/reid.py
职责: 对 Grounding 结果进行质量评分，筛选最佳参考帧

Character: InsightFace FaceID 置信度 + 清晰度 + 正面角度
  综合分 = 0.4×清晰度 + 0.4×ID置信度 + 0.2×正面角度

Location:  清晰度 + 内容丰富度 + inpaint残留惩罚
  综合分 = 0.5×清晰度 + 0.3×内容丰富度 + 0.2×(1-inpaint惩罚)
"""

import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from PIL import Image


@dataclass
class QualityScore:
    crop_path: str
    sharpness: float          # 0-1，Laplacian 方差归一化
    id_confidence: float      # 0-1，人脸 ID 置信度（非人物取 1.0）
    frontal_score: float      # 0-1，正面程度（yaw 角接近 0 则高）
    occlusion_score: float    # 0-1，无遮挡程度
    final_score: float        # 加权综合分
    # Location 专用字段
    content_richness: float = 0.0   # 0-1，内容丰富度（边缘密度）
    inpaint_penalty: float = 0.0    # 0-1，inpaint 白色区域占比（越低越好）
    sharpness_raw: float = 0.0      # 原始 Laplacian 方差值（未归一化）


class ReferenceQualityScorer:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._face_analyzer = None
        self._clip_model = None
        self._clip_preprocess = None

    def _load_face_analyzer(self):
        if self._face_analyzer is not None:
            return
        try:
            import insightface
            from insightface.app import FaceAnalysis
            self._face_analyzer = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            self._face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
            print("[ReID] InsightFace loaded.")
        except ImportError:
            print("[ReID] InsightFace 未安装，将跳过人脸 ID 评分 (pip install insightface)")
            self._face_analyzer = None

    def _load_clip(self):
        if self._clip_model is not None:
            return
        try:
            import clip
            import torch
            self._clip_model, self._clip_preprocess = clip.load("ViT-B/32", device=self.device)
            print("[ReID] CLIP loaded.")
        except ImportError:
            print("[ReID] CLIP 未安装，将跳过 CLIP 相似度评分 (pip install git+https://github.com/openai/CLIP.git)")

    # ── 清晰度评分（Laplacian 方差）─────────────────────────────────────────
    @staticmethod
    def _compute_sharpness(img_bgr: np.ndarray, return_raw: bool = False) -> float:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # 归一化：经验上 500 以上算清晰，映射到 0-1
        normalized = min(1.0, lap_var / 500.0)
        if return_raw:
            return normalized, lap_var
        return normalized

    # ── Location 专用：内容丰富度（边缘密度）─────────────────────────────────
    @staticmethod
    def _compute_content_richness(img_bgr: np.ndarray) -> float:
        """
        计算图像的内容丰富度：边缘密度 + 颜色方差
        用于评估背景是否有足够的场景细节
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 1. 边缘密度（Canny 边缘占比）
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        # 归一化：0.05 以上算丰富
        edge_score = min(1.0, edge_density / 0.05)

        # 2. 颜色方差（HSV 空间的 Hue 和 Saturation 方差）
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        sat_var = np.var(hsv[:, :, 1]) / 255.0  # 饱和度方差
        # 归一化
        color_score = min(1.0, sat_var / 0.1)

        # 综合
        return 0.7 * edge_score + 0.3 * color_score

    # ── Location 专用：检测 inpaint 白色区域 ─────────────────────────────────
    @staticmethod
    def _compute_inpaint_penalty(img_bgr: np.ndarray) -> float:
        """
        检测 inpaint 残留的白色/过曝区域占比
        返回 0-1，值越大表示白色区域越多（越不好）
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # 白色区域：高亮度 + 低饱和度
        white_mask = (gray > 240) & (hsv[:, :, 1] < 30)
        white_ratio = np.sum(white_mask) / white_mask.size

        # 归一化：超过 20% 白色区域就给最大惩罚
        return min(1.0, white_ratio / 0.2)

    # ── 人脸正面度 & ID 置信度 ────────────────────────────────────────────
    def _compute_face_scores(self, img_bgr: np.ndarray):
        """返回 (id_confidence, frontal_score)"""
        self._load_face_analyzer()
        if self._face_analyzer is None:
            return 0.8, 0.8  # fallback

        faces = self._face_analyzer.get(img_bgr)
        if not faces:
            return 0.0, 0.0  # 未检测到人脸：直接给零分，不能作为 character 参考

        best = max(faces, key=lambda f: f.det_score)
        id_conf = float(best.det_score)

        # yaw 角度：0 = 正面，±90 = 侧面
        yaw = abs(best.pose[1]) if hasattr(best, "pose") else 0.0
        frontal = max(0.0, 1.0 - yaw / 90.0)

        return id_conf, frontal

    # ── 遮挡评分（简易：检测 box 面积 / 图像面积）─────────────────────────
    @staticmethod
    def _compute_occlusion(img_bgr: np.ndarray, bbox_area: Optional[float] = None) -> float:
        h, w = img_bgr.shape[:2]
        if bbox_area is None:
            return 0.8  # 无 bbox 信息时默认较好
        img_area = h * w
        ratio = bbox_area / max(img_area, 1)
        # 太小的检测框可能被遮挡
        if ratio < 0.05:
            return 0.3
        elif ratio < 0.15:
            return 0.6
        else:
            return 1.0

    # ── 主接口 ────────────────────────────────────────────────────────────
    def score(
        self,
        crop_path: str,
        entity_type: str = "character",
        bbox_area: Optional[float] = None,
    ) -> QualityScore:
        img_bgr = cv2.imread(crop_path)
        if img_bgr is None:
            return QualityScore(crop_path, 0, 0, 0, 0, 0)

        sharpness = self._compute_sharpness(img_bgr)
        occlusion = self._compute_occlusion(img_bgr, bbox_area)

        if entity_type == "character":
            id_conf, frontal = self._compute_face_scores(img_bgr)
        else:
            id_conf, frontal = 1.0, 1.0  # 非人物不需要人脸评分

        final = 0.4 * sharpness + 0.4 * id_conf + 0.2 * frontal

        return QualityScore(
            crop_path=crop_path,
            sharpness=round(sharpness, 3),
            id_confidence=round(id_conf, 3),
            frontal_score=round(frontal, 3),
            occlusion_score=round(occlusion, 3),
            final_score=round(final, 3),
        )

    def score_location(
        self,
        crop_path: str,
        bbox_area: Optional[float] = None,
    ) -> QualityScore:
        """
        Location 专用评分：清晰度 + 内容丰富度 + inpaint 惩罚

        评分公式：
            final = 0.5×清晰度 + 0.3×内容丰富度 + 0.2×(1-inpaint惩罚)

        清晰度权重最高，因为模糊背景会严重影响生成质量。
        """
        img_bgr = cv2.imread(crop_path)
        if img_bgr is None:
            return QualityScore(crop_path, 0, 0, 0, 0, 0)

        sharpness, sharpness_raw = self._compute_sharpness(img_bgr, return_raw=True)
        content_richness = self._compute_content_richness(img_bgr)
        inpaint_penalty = self._compute_inpaint_penalty(img_bgr)

        # Location 评分公式
        final = (
            0.5 * sharpness +
            0.3 * content_richness +
            0.2 * (1 - inpaint_penalty)
        )

        return QualityScore(
            crop_path=crop_path,
            sharpness=round(sharpness, 3),
            id_confidence=1.0,  # location 不需要
            frontal_score=1.0,  # location 不需要
            occlusion_score=1.0,
            final_score=round(final, 3),
            content_richness=round(content_richness, 3),
            inpaint_penalty=round(inpaint_penalty, 3),
            sharpness_raw=round(sharpness_raw, 1),
        )

    def rank_references(
        self,
        crop_paths: List[str],
        entity_type: str = "character",
        ground_results=None,  # Optional[List[GroundingResult]]
    ) -> List[QualityScore]:
        """对多张候选参考图打分并排序，返回从高到低。
        如果传入 ground_results，会利用 bbox 面积参与评分。
        """
        bbox_areas = {}
        if ground_results is not None:
            for r in ground_results:
                x1, y1, x2, y2 = r.bbox
                bbox_areas[r.crop_path] = (x2 - x1) * (y2 - y1)

        # 根据实体类型选择评分方法
        if entity_type == "location":
            scores = [
                self.score_location(p, bbox_area=bbox_areas.get(p))
                for p in crop_paths
            ]
        else:
            scores = [
                self.score(p, entity_type, bbox_area=bbox_areas.get(p))
                for p in crop_paths
            ]

        scores.sort(key=lambda s: s.final_score, reverse=True)
        return scores


# ── 快速测试 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python reid.py <crop_dir>")
        sys.exit(1)

    import glob
    crop_paths = glob.glob(os.path.join(sys.argv[1], "*.jpg"))
    scorer = ReferenceQualityScorer()
    ranked = scorer.rank_references(crop_paths, entity_type="character")

    print(f"\n共 {len(ranked)} 张参考图，质量排序:")
    for i, s in enumerate(ranked[:5]):
        print(f"  #{i+1}  final={s.final_score:.3f}  sharp={s.sharpness:.3f}  "
              f"id_conf={s.id_confidence:.3f}  frontal={s.frontal_score:.3f}")
        print(f"        {s.crop_path}")
