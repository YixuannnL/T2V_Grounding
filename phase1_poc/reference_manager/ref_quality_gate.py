"""
reference_manager/ref_quality_gate.py
职责: VLM 驱动的参考图质量门控 —— 判断 crop 是否值得入库

核心问题：
    某些 Grounding 结果虽然检测到了目标，但 crop 质量极差：
    - 纯黑剪影（如夜景中的马）
    - 严重遮挡/截断
    - 模糊到无法辨认
    - 与描述不符（误检测）

    这些低质量 ref 如果入库，会在后续 S2V 中产生负面影响。
    与其用一堆启发式规则，不如直接让 VLM 来判断：
    "这张图片能否作为 [entity_description] 的参考图？"

使用方式：
    gate = ReferenceQualityGate(model="claude-sonnet-4-6")

    # 单张判断
    result = gate.check(crop_path, entity_id="obj_horse", entity_description="a brown horse")
    if result.should_register:
        registry.register(entity_id, entry)
    else:
        print(f"拒绝入库: {result.reason}")

    # 批量判断（更高效，一次 API 调用）
    results = gate.check_batch(crop_paths, entity_id, entity_description)
"""

import os
import base64
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum


class RejectionReason(Enum):
    """拒绝入库的原因"""
    TOO_DARK = "too_dark"                    # 图像过暗，无法辨认
    TOO_BRIGHT = "too_bright"                # 图像过亮/过曝
    SILHOUETTE = "silhouette"                # 只有剪影轮廓
    BLURRY = "blurry"                        # 过于模糊
    OCCLUDED = "occluded"                    # 严重遮挡
    TRUNCATED = "truncated"                  # 主体被截断
    MISMATCH = "mismatch"                    # 与描述不符（误检测）
    LOW_INFO = "low_info"                    # 信息量过低
    UNRECOGNIZABLE = "unrecognizable"        # 无法辨认是什么
    PASSED = "passed"                        # 通过检查


@dataclass
class QualityGateResult:
    """质量门控结果"""
    crop_path: str
    should_register: bool           # 是否应该入库
    confidence: float               # VLM 判断的置信度 (0-1)
    reason: RejectionReason         # 原因
    detail: str                     # 详细说明
    recognizable_content: str       # VLM 识别出的内容（用于交叉验证）
    suggested_score: float          # VLM 建议的质量分 (0-1)


class ReferenceQualityGate:
    """
    VLM 驱动的参考图质量门控

    在 crop 入库前，用 VLM 判断其是否值得作为参考图。
    比基于规则的启发式方法更智能、更准确。
    """

    # VLM 判断 Prompt
    QUALITY_CHECK_PROMPT = """你是一个视频生成参考图质量审核员。

请判断这张图片是否适合作为视频生成的参考图。

**期望的主体**: {entity_description}
**实体类型**: {entity_type}

请评估以下维度：

1. **可识别性** (0-10): 能否清楚辨认出图中是什么？
   - 10: 非常清晰，一眼就能认出
   - 5: 勉强能辨认
   - 0: 完全无法辨认（纯黑/纯白/严重模糊）

2. **匹配度** (0-10): 图中内容是否与期望的主体匹配？
   - 10: 完全匹配
   - 5: 部分匹配或有歧义
   - 0: 完全不匹配（误检测）

3. **完整度** (0-10): 主体是否完整呈现？
   - 10: 主体完整，没有截断或遮挡
   - 5: 部分截断或遮挡
   - 0: 严重截断或遮挡

4. **信息量** (0-10): 图片是否包含足够的视觉信息？
   - 10: 细节丰富，纹理清晰
   - 5: 信息量一般
   - 0: 信息量极低（剪影、纯色块）

**最终判断**:
- 如果 (可识别性 >= 5) AND (匹配度 >= 5) AND (完整度 >= 3) AND (信息量 >= 4)，则 PASS
- 否则 REJECT

请严格按以下 JSON 格式输出：
```json
{{
  "recognizable_content": "你在图中看到的内容描述",
  "scores": {{
    "recognizability": <0-10>,
    "match": <0-10>,
    "completeness": <0-10>,
    "information": <0-10>
  }},
  "decision": "PASS" 或 "REJECT",
  "reason": "reject原因，如too_dark/silhouette/mismatch/blurry/truncated/occluded/low_info/unrecognizable",
  "detail": "简短解释",
  "suggested_quality_score": <0.0-1.0>
}}
```

只输出 JSON，不要其他内容。"""

    BATCH_CHECK_PROMPT = """你是一个视频生成参考图质量审核员。

请判断这批图片（共 {num_images} 张）是否适合作为视频生成的参考图。

**期望的主体**: {entity_description}
**实体类型**: {entity_type}

对每张图片，评估：
1. **可识别性** (0-10): 能否清楚辨认？
2. **匹配度** (0-10): 与期望主体是否匹配？
3. **完整度** (0-10): 主体是否完整？
4. **信息量** (0-10): 视觉信息是否充足？

判断规则：
- PASS: 可识别性>=5 AND 匹配度>=5 AND 完整度>=3 AND 信息量>=4
- 否则: REJECT

请严格按以下 JSON 格式输出：
```json
{{
  "results": [
    {{
      "image_index": 0,
      "recognizable_content": "图中内容描述",
      "scores": {{"recognizability": X, "match": X, "completeness": X, "information": X}},
      "decision": "PASS/REJECT",
      "reason": "passed/too_dark/silhouette/mismatch/blurry/truncated/occluded/low_info/unrecognizable",
      "detail": "简短解释",
      "suggested_quality_score": 0.X
    }},
    ...
  ]
}}
```

只输出 JSON，不要其他内容。"""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        enabled: bool = True,
        cache_results: bool = True,
    ):
        """
        Args:
            model: VLM 模型名称
            enabled: 是否启用（禁用时所有 check 都返回 PASS）
            cache_results: 是否缓存结果（避免重复检查同一张图）
        """
        self.model = model
        self.enabled = enabled
        self.cache_results = cache_results
        self._cache = {}  # crop_path -> QualityGateResult
        self._llm_client = None

    def _get_llm_client(self):
        """延迟初始化 LLM 客户端"""
        if self._llm_client is None:
            try:
                from utils.llm_client import LLMClient
                self._llm_client = LLMClient(model=self.model)
                print(f"[QualityGate] VLM 客户端初始化成功 (model={self.model})")
            except Exception as e:
                print(f"[QualityGate] VLM 客户端初始化失败: {e}")
                self._llm_client = None
        return self._llm_client

    def _encode_image_base64(self, image_path: str) -> str:
        """将图片编码为 base64"""
        with open(image_path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")

    def _get_media_type(self, image_path: str) -> str:
        """获取图片的 MIME 类型"""
        ext = os.path.splitext(image_path)[1].lower()
        return {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }.get(ext, "image/jpeg")

    def _parse_reason(self, reason_str: str) -> RejectionReason:
        """解析 VLM 返回的 reason 字符串"""
        # 处理 None 或空字符串
        if not reason_str:
            return RejectionReason.UNRECOGNIZABLE

        reason_map = {
            "too_dark": RejectionReason.TOO_DARK,
            "too_bright": RejectionReason.TOO_BRIGHT,
            "silhouette": RejectionReason.SILHOUETTE,
            "blurry": RejectionReason.BLURRY,
            "occluded": RejectionReason.OCCLUDED,
            "truncated": RejectionReason.TRUNCATED,
            "mismatch": RejectionReason.MISMATCH,
            "low_info": RejectionReason.LOW_INFO,
            "unrecognizable": RejectionReason.UNRECOGNIZABLE,
            "passed": RejectionReason.PASSED,
        }
        return reason_map.get(reason_str.lower(), RejectionReason.UNRECOGNIZABLE)

    def check(
        self,
        crop_path: str,
        entity_id: str,
        entity_description: str = "",
    ) -> QualityGateResult:
        """
        检查单张 crop 是否值得入库

        Args:
            crop_path: crop 图片路径
            entity_id: 实体 ID (如 "obj_horse", "char_001")
            entity_description: 实体描述 (如 "a brown horse")

        Returns:
            QualityGateResult
        """
        # 禁用时直接通过
        if not self.enabled:
            return QualityGateResult(
                crop_path=crop_path,
                should_register=True,
                confidence=1.0,
                reason=RejectionReason.PASSED,
                detail="Quality gate disabled",
                recognizable_content="",
                suggested_score=0.7,
            )

        # 检查缓存
        if self.cache_results and crop_path in self._cache:
            return self._cache[crop_path]

        # 检查文件存在
        if not os.path.exists(crop_path):
            result = QualityGateResult(
                crop_path=crop_path,
                should_register=False,
                confidence=1.0,
                reason=RejectionReason.UNRECOGNIZABLE,
                detail="File not found",
                recognizable_content="",
                suggested_score=0.0,
            )
            return result

        # 推断实体类型
        entity_type = "object"
        if entity_id.startswith("char_"):
            entity_type = "character"
        elif entity_id.startswith("loc_"):
            entity_type = "location"

        # 如果没有描述，使用 entity_id
        if not entity_description:
            entity_description = entity_id.replace("_", " ")

        # 调用 VLM
        result = self._call_vlm_single(crop_path, entity_type, entity_description)

        # 缓存结果
        if self.cache_results:
            self._cache[crop_path] = result

        return result

    def _call_vlm_single(
        self,
        crop_path: str,
        entity_type: str,
        entity_description: str,
    ) -> QualityGateResult:
        """调用 VLM 检查单张图片"""
        llm = self._get_llm_client()
        if llm is None:
            # VLM 不可用时，fallback 到通过
            print(f"[QualityGate] VLM 不可用，默认通过: {crop_path}")
            return QualityGateResult(
                crop_path=crop_path,
                should_register=True,
                confidence=0.5,
                reason=RejectionReason.PASSED,
                detail="VLM unavailable, fallback to pass",
                recognizable_content="",
                suggested_score=0.6,
            )

        try:
            # 构建 prompt
            prompt = self.QUALITY_CHECK_PROMPT.format(
                entity_description=entity_description,
                entity_type=entity_type,
            )

            # 编码图片
            image_base64 = self._encode_image_base64(crop_path)
            media_type = self._get_media_type(crop_path)

            # 使用 chat_with_images 方法（OpenAI 兼容格式）
            content_parts = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_base64,
                    },
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ]

            # 调用 VLM
            response = llm.chat_with_images(content_parts, max_tokens=1024)

            # 解析响应
            return self._parse_vlm_response(crop_path, response)

        except Exception as e:
            print(f"[QualityGate] VLM 调用失败: {e}")
            # 出错时默认通过（宽松策略）
            return QualityGateResult(
                crop_path=crop_path,
                should_register=True,
                confidence=0.5,
                reason=RejectionReason.PASSED,
                detail=f"VLM error: {e}",
                recognizable_content="",
                suggested_score=0.6,
            )

    def _parse_vlm_response(self, crop_path: str, response: str) -> QualityGateResult:
        """解析 VLM 返回的 JSON"""
        import json

        # 处理 None 响应
        if response is None:
            return QualityGateResult(
                crop_path=crop_path,
                should_register=True,
                confidence=0.3,
                reason=RejectionReason.PASSED,
                detail="VLM returned empty response",
                recognizable_content="",
                suggested_score=0.5,
            )

        try:
            # 尝试提取 JSON
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]

            data = json.loads(response.strip())

            decision = data.get("decision", "REJECT").upper()
            should_register = (decision == "PASS")

            scores = data.get("scores", {})
            # 计算置信度：基于各维度分数的一致性
            score_values = list(scores.values())
            confidence = 1.0 - (max(score_values) - min(score_values)) / 10.0 if score_values else 0.5

            reason_str = data.get("reason", "passed" if should_register else "unrecognizable")
            reason = self._parse_reason(reason_str)

            return QualityGateResult(
                crop_path=crop_path,
                should_register=should_register,
                confidence=confidence,
                reason=reason,
                detail=data.get("detail", ""),
                recognizable_content=data.get("recognizable_content", ""),
                suggested_score=data.get("suggested_quality_score", 0.5),
            )

        except json.JSONDecodeError as e:
            print(f"[QualityGate] JSON 解析失败: {e}")
            print(f"[QualityGate] 原始响应: {response[:200]}...")
            return QualityGateResult(
                crop_path=crop_path,
                should_register=True,  # 解析失败时宽松处理
                confidence=0.3,
                reason=RejectionReason.PASSED,
                detail=f"JSON parse error: {e}",
                recognizable_content="",
                suggested_score=0.5,
            )

    def check_batch(
        self,
        crop_paths: List[str],
        entity_id: str,
        entity_description: str = "",
    ) -> List[QualityGateResult]:
        """
        批量检查多张 crop（更高效，一次 API 调用）

        Args:
            crop_paths: crop 图片路径列表
            entity_id: 实体 ID
            entity_description: 实体描述

        Returns:
            QualityGateResult 列表
        """
        if not self.enabled:
            return [
                QualityGateResult(
                    crop_path=p,
                    should_register=True,
                    confidence=1.0,
                    reason=RejectionReason.PASSED,
                    detail="Quality gate disabled",
                    recognizable_content="",
                    suggested_score=0.7,
                )
                for p in crop_paths
            ]

        # 检查缓存，分离出需要检查的
        results = {}
        to_check = []
        for p in crop_paths:
            if self.cache_results and p in self._cache:
                results[p] = self._cache[p]
            else:
                to_check.append(p)

        # 如果都命中缓存，直接返回
        if not to_check:
            return [results[p] for p in crop_paths]

        # 批量检查（限制单次最多 4 张，避免 token 超限）
        batch_size = 4
        for i in range(0, len(to_check), batch_size):
            batch = to_check[i:i+batch_size]
            batch_results = self._call_vlm_batch(batch, entity_id, entity_description)
            for p, r in zip(batch, batch_results):
                results[p] = r
                if self.cache_results:
                    self._cache[p] = r

        return [results[p] for p in crop_paths]

    def _call_vlm_batch(
        self,
        crop_paths: List[str],
        entity_id: str,
        entity_description: str,
    ) -> List[QualityGateResult]:
        """批量调用 VLM"""
        llm = self._get_llm_client()
        if llm is None:
            return [
                QualityGateResult(
                    crop_path=p,
                    should_register=True,
                    confidence=0.5,
                    reason=RejectionReason.PASSED,
                    detail="VLM unavailable",
                    recognizable_content="",
                    suggested_score=0.6,
                )
                for p in crop_paths
            ]

        # 推断实体类型
        entity_type = "object"
        if entity_id.startswith("char_"):
            entity_type = "character"
        elif entity_id.startswith("loc_"):
            entity_type = "location"

        if not entity_description:
            entity_description = entity_id.replace("_", " ")

        try:
            # 构建 prompt
            prompt = self.BATCH_CHECK_PROMPT.format(
                num_images=len(crop_paths),
                entity_description=entity_description,
                entity_type=entity_type,
            )

            # 构建多图消息
            content = []
            for i, p in enumerate(crop_paths):
                if os.path.exists(p):
                    content.append({
                        "type": "text",
                        "text": f"[Image {i}]",
                    })
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": self._get_media_type(p),
                            "data": self._encode_image_base64(p),
                        },
                    })

            content.append({
                "type": "text",
                "text": prompt,
            })

            # 使用 chat_with_images 方法
            response = llm.chat_with_images(content, max_tokens=2048)

            # 解析批量响应
            return self._parse_vlm_batch_response(crop_paths, response)

        except Exception as e:
            print(f"[QualityGate] 批量 VLM 调用失败: {e}")
            return [
                QualityGateResult(
                    crop_path=p,
                    should_register=True,
                    confidence=0.5,
                    reason=RejectionReason.PASSED,
                    detail=f"VLM error: {e}",
                    recognizable_content="",
                    suggested_score=0.6,
                )
                for p in crop_paths
            ]

    def _parse_vlm_batch_response(
        self,
        crop_paths: List[str],
        response: str,
    ) -> List[QualityGateResult]:
        """解析批量 VLM 响应"""
        import json

        try:
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]

            data = json.loads(response.strip())
            results_data = data.get("results", [])

            results = []
            for i, p in enumerate(crop_paths):
                if i < len(results_data):
                    item = results_data[i]
                    decision = item.get("decision", "REJECT").upper()
                    should_register = (decision == "PASS")
                    reason_str = item.get("reason", "passed" if should_register else "unrecognizable")

                    results.append(QualityGateResult(
                        crop_path=p,
                        should_register=should_register,
                        confidence=0.8,
                        reason=self._parse_reason(reason_str),
                        detail=item.get("detail", ""),
                        recognizable_content=item.get("recognizable_content", ""),
                        suggested_score=item.get("suggested_quality_score", 0.5),
                    ))
                else:
                    # 响应不完整，默认通过
                    results.append(QualityGateResult(
                        crop_path=p,
                        should_register=True,
                        confidence=0.3,
                        reason=RejectionReason.PASSED,
                        detail="Missing in batch response",
                        recognizable_content="",
                        suggested_score=0.5,
                    ))

            return results

        except json.JSONDecodeError as e:
            print(f"[QualityGate] 批量 JSON 解析失败: {e}")
            return [
                QualityGateResult(
                    crop_path=p,
                    should_register=True,
                    confidence=0.3,
                    reason=RejectionReason.PASSED,
                    detail=f"JSON parse error",
                    recognizable_content="",
                    suggested_score=0.5,
                )
                for p in crop_paths
            ]

    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()


# ══════════════════════════════════════════════════════════════════════════════════
# 便捷函数：集成到 SmartRegistry
# ══════════════════════════════════════════════════════════════════════════════════

def should_register_with_vlm_gate(
    crop_path: str,
    entity_id: str,
    entity_description: str,
    gate: Optional[ReferenceQualityGate] = None,
    model: str = "claude-sonnet-4-6",
) -> Tuple[bool, str, float]:
    """
    便捷函数：检查 crop 是否应该入库

    Returns:
        (should_register, reason, suggested_score)
    """
    if gate is None:
        gate = ReferenceQualityGate(model=model)

    result = gate.check(crop_path, entity_id, entity_description)

    return (
        result.should_register,
        f"{result.reason.value}: {result.detail}",
        result.suggested_score,
    )


# ══════════════════════════════════════════════════════════════════════════════════
# 快速测试
# ══════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    # 测试用例
    test_cases = [
        # (crop_path, entity_id, entity_description)
        ("./output_rider_3/run_001_seed1337/crops/shot_001/obj_horse/obj_horse_frame_000000_crop.jpg",
         "obj_horse", "a brown horse being ridden"),
        ("./output_rider_3/run_001_seed1337/crops/shot_003/char_001/char_001_frame_000072_crop.jpg",
         "char_001", "a man in traditional Chinese attire"),
    ]

    if len(sys.argv) > 1:
        # 命令行指定测试图片
        test_cases = [(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "obj_test",
                       sys.argv[3] if len(sys.argv) > 3 else "test object")]

    print("\n" + "="*60)
    print("VLM Reference Quality Gate 测试")
    print("="*60)

    gate = ReferenceQualityGate(model="claude-sonnet-4-6", enabled=True)

    for crop_path, entity_id, entity_desc in test_cases:
        if not os.path.exists(crop_path):
            print(f"\n⚠️  文件不存在: {crop_path}")
            continue

        print(f"\n测试: {crop_path}")
        print(f"  实体: {entity_id}")
        print(f"  描述: {entity_desc}")

        result = gate.check(crop_path, entity_id, entity_desc)

        status = "✅ PASS" if result.should_register else "❌ REJECT"
        print(f"  结果: {status}")
        print(f"  原因: {result.reason.value}")
        print(f"  详情: {result.detail}")
        print(f"  识别内容: {result.recognizable_content}")
        print(f"  建议分数: {result.suggested_score:.2f}")
        print(f"  置信度: {result.confidence:.2f}")
