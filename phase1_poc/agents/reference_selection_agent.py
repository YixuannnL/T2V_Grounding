"""
agents/reference_selection_agent.py

Agentic Reference Selection - 使用 VLM 智能选择最佳参考图

核心优势（相比传统 InsightFace 打分）:
  1. 通用性: 人物、物体、场景都能语义评估，无需针对不同实体类型设计不同打分公式
  2. 上下文感知: 根据当前 shot 需求选择（动作戏选动态姿态，对话选正面表情）
  3. 语义理解: 理解 "需要正面"、"需要侧面"、"需要特定表情" 等复杂要求
  4. 可解释性: Agent 输出选择理由，便于 debug 和人工审核

使用方式:
    agent = ReferenceSelectionAgent()

    # 单实体选择
    best_ref = agent.select_best_reference(
        entity=entity,
        candidates=[crop_path1, crop_path2, ...],
        shot_context="A close-up of Alex smiling warmly at the camera",
        shot_type="close-up"
    )

    # 批量选择（多实体）
    selections = agent.select_references_for_shot(
        entities=[entity1, entity2],
        registry=registry,
        shot_context="...",
        shot_type="medium"
    )
"""

import os
import sys
import json
import base64
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.llm_client import LLMClient
from reference_manager.registry import EntityRegistry, ReferenceEntry


# ============================================================================
# 数据结构定义
# ============================================================================

@dataclass
class SelectionResult:
    """参考图选择结果"""
    entity_id: str
    selected_path: Optional[str]        # 选中的参考图路径，None 表示无合适候选
    selected_index: int                 # 选中的候选图索引 (0-based)，-1 表示未选中
    confidence: float                   # 选择置信度 0-1
    reason: str                         # 选择理由（VLM 生成）
    alternatives: List[str] = field(default_factory=list)  # 备选参考图路径
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据


@dataclass
class ShotRequirements:
    """镜头需求分析结果"""
    shot_type: str                      # close-up / medium / wide
    required_angle: str                 # frontal / side / back / any
    required_expression: str            # neutral / smiling / serious / any
    action_type: str                    # static / dynamic / dialogue / action
    lighting_condition: str             # bright / dark / mixed / any
    special_requirements: List[str] = field(default_factory=list)


# ============================================================================
# 参考图选择 Agent
# ============================================================================

class ReferenceSelectionAgent:
    """
    Agentic 参考图选择器

    使用 VLM (Vision Language Model) 分析候选参考图，
    结合当前镜头的上下文需求，智能选择最佳参考图。

    支持的实体类型:
      - character: 人物（分析姿态、表情、光线、清晰度）
      - object: 物体（分析角度、完整度、清晰度）
      - location: 场景（分析构图、光线、环境特征）
    """

    # 支持视觉的模型列表（Claude 系列）
    VISION_MODELS = [
        "claude-opus-4-6", "claude-opus-4-5", "claude-sonnet-4-6", "claude-sonnet-4-5",
        "claude-haiku-4-5",
    ]

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",  # 使用 Sonnet 平衡性能和成本
        temperature: float = 0.2,           # 低温度保证选择稳定性
        max_candidates: int = 6,            # 单次最多评估的候选图数量（避免 prompt 过长）
        enable_fallback: bool = True,       # VLM 失败时是否回退到传统评分
        verbose: bool = True,
    ):
        self.model = model
        self.temperature = temperature
        self.max_candidates = max_candidates
        self.enable_fallback = enable_fallback
        self.verbose = verbose

        # 延迟初始化 LLM 客户端
        self._llm_client: Optional[LLMClient] = None

        # 传统评分器（fallback 用）
        self._traditional_scorer = None

    @property
    def llm_client(self) -> LLMClient:
        if self._llm_client is None:
            self._llm_client = LLMClient(model=self.model)
        return self._llm_client

    def _get_traditional_scorer(self):
        """延迟加载传统评分器"""
        if self._traditional_scorer is None:
            from visual_grounding.reid import ReferenceQualityScorer
            self._traditional_scorer = ReferenceQualityScorer()
        return self._traditional_scorer

    # ========================================================================
    # 核心接口：单实体选择
    # ========================================================================

    def select_best_reference(
        self,
        entity_id: str,
        entity_type: str,
        entity_description: str,
        candidates: List[str],
        shot_context: str,
        shot_type: str = "medium",
        history_context: Optional[str] = None,
        custom_requirements: Optional[List[str]] = None,
    ) -> SelectionResult:
        """
        为单个实体从候选参考图中选择最佳参考

        Args:
            entity_id: 实体 ID (e.g., "char_alex")
            entity_type: 实体类型 ("character" / "object" / "location")
            entity_description: 实体文本描述 (e.g., "Alex, a young man with short brown hair")
            candidates: 候选参考图路径列表
            shot_context: 当前镜头描述 (e.g., "A close-up of Alex smiling at the camera")
            shot_type: 镜头类型 ("close-up" / "medium" / "wide")
            history_context: 历史镜头上下文（可选，用于保持连贯性）
            custom_requirements: 自定义选择要求（可选）

        Returns:
            SelectionResult: 选择结果，包含选中的参考图、理由等
        """
        if not candidates:
            return SelectionResult(
                entity_id=entity_id,
                selected_path=None,
                selected_index=-1,
                confidence=0.0,
                reason="No candidate references available",
            )

        # 限制候选数量，避免 prompt 过长
        eval_candidates = candidates[:self.max_candidates]

        try:
            # 尝试使用 VLM 智能选择
            result = self._vlm_select(
                entity_id=entity_id,
                entity_type=entity_type,
                entity_description=entity_description,
                candidates=eval_candidates,
                shot_context=shot_context,
                shot_type=shot_type,
                history_context=history_context,
                custom_requirements=custom_requirements,
            )

            if self.verbose:
                print(f"[RefSelAgent] {entity_id}: 选择 #{result.selected_index + 1} "
                      f"(confidence={result.confidence:.2f})")
                print(f"[RefSelAgent] 理由: {result.reason}")

            return result

        except Exception as e:
            if self.verbose:
                print(f"[RefSelAgent] VLM 选择失败: {e}")

            if self.enable_fallback:
                return self._fallback_select(
                    entity_id=entity_id,
                    entity_type=entity_type,
                    candidates=eval_candidates,
                )
            else:
                raise

    # ========================================================================
    # 核心接口：批量选择（多实体）
    # ========================================================================

    def select_references_for_shot(
        self,
        entities: List[Any],  # List[Entity] from entity_parser
        registry: EntityRegistry,
        shot_context: str,
        shot_type: str = "medium",
        max_refs_per_entity: int = 1,
        min_quality: float = 0.3,
    ) -> Dict[str, SelectionResult]:
        """
        为一个镜头的多个实体批量选择参考图

        Args:
            entities: 实体列表 (来自 EntityParser)
            registry: 实体参考图注册库
            shot_context: 当前镜头描述
            shot_type: 镜头类型
            max_refs_per_entity: 每个实体最多选择几张参考图
            min_quality: 候选图最低质量分

        Returns:
            Dict[entity_id, SelectionResult]: 每个实体的选择结果
        """
        results = {}

        for entity in entities:
            # 从 registry 获取候选参考图
            refs = registry.query(
                entity.entity_id,
                top_k=self.max_candidates,
                min_quality=min_quality,
                anchor_strategy="earliest_good",  # 优先早期高质量
            )

            if not refs:
                results[entity.entity_id] = SelectionResult(
                    entity_id=entity.entity_id,
                    selected_path=None,
                    selected_index=-1,
                    confidence=0.0,
                    reason="No references in registry",
                )
                continue

            candidates = [r.crop_path for r in refs]

            result = self.select_best_reference(
                entity_id=entity.entity_id,
                entity_type=entity.type,
                entity_description=entity.text_description,
                candidates=candidates,
                shot_context=shot_context,
                shot_type=shot_type,
            )

            # 添加 registry 元数据
            if result.selected_index >= 0:
                selected_ref = refs[result.selected_index]
                result.metadata["source_shot"] = selected_ref.shot_id
                result.metadata["quality_score"] = selected_ref.quality_score
                result.metadata["id_confidence"] = selected_ref.id_confidence

            results[entity.entity_id] = result

        return results

    # ========================================================================
    # VLM 智能选择实现
    # ========================================================================

    def _vlm_select(
        self,
        entity_id: str,
        entity_type: str,
        entity_description: str,
        candidates: List[str],
        shot_context: str,
        shot_type: str,
        history_context: Optional[str] = None,
        custom_requirements: Optional[List[str]] = None,
    ) -> SelectionResult:
        """使用 VLM 进行智能选择"""

        # 构建 prompt
        system_prompt = self._build_system_prompt(entity_type)
        user_prompt, images = self._build_user_prompt(
            entity_id=entity_id,
            entity_type=entity_type,
            entity_description=entity_description,
            candidates=candidates,
            shot_context=shot_context,
            shot_type=shot_type,
            history_context=history_context,
            custom_requirements=custom_requirements,
        )

        # 构建多模态消息
        content = self._build_multimodal_content(user_prompt, images)

        # 调用 VLM
        response = self._call_vlm(system_prompt, content)

        # 解析响应
        return self._parse_vlm_response(
            response=response,
            entity_id=entity_id,
            candidates=candidates,
        )

    def _build_system_prompt(self, entity_type: str) -> str:
        """构建系统提示词"""

        base_prompt = """You are an expert visual reference selector for video generation.
Your task is to select the BEST reference image for a given entity and shot context.

You will be shown:
1. Multiple candidate reference images (numbered)
2. The entity description
3. The current shot context (what will be generated)
4. The shot type (close-up / medium / wide)

Your job is to analyze each candidate and select the ONE that will work best for the upcoming shot.
"""

        type_specific = {
            "character": """
For CHARACTER references, evaluate:
- **Angle**: Does the angle match the shot needs? (frontal for dialogue, side for action, etc.)
- **Expression**: Is the expression appropriate? (smiling for happy scenes, neutral for serious)
- **Pose**: Is the body pose suitable? (dynamic for action, static for portraits)
- **Lighting**: Does the lighting match the scene mood?
- **Clarity**: Is the face/body clear and well-defined?
- **Completeness**: Is enough of the character visible?

IMPORTANT: For close-ups, prefer frontal faces. For action shots, dynamic poses are better.
""",
            "object": """
For OBJECT references, evaluate:
- **Angle**: Is the object shown from a useful angle?
- **Completeness**: Is the entire object visible?
- **Clarity**: Is the object sharp and well-defined?
- **Context**: Does the surrounding context match the scene?
- **Scale**: Can the object's details be preserved at the target shot scale?
""",
            "location": """
For LOCATION/SCENE references, evaluate:
- **Coverage**: Does it show the key environmental elements?
- **Lighting**: Does the lighting mood match the shot?
- **Composition**: Is the scene composition useful as reference?
- **Clarity**: Are scene details sharp?
- **Atmosphere**: Does the atmosphere match the narrative tone?
"""
        }

        output_format = """
OUTPUT FORMAT (JSON):
{
  "selected_index": <0-based index of best candidate>,
  "confidence": <0.0-1.0 confidence score>,
  "reason": "<brief explanation of why this is the best choice>",
  "analysis": {
    "candidate_1": "<brief analysis>",
    "candidate_2": "<brief analysis>",
    ...
  },
  "alternatives": [<indices of acceptable alternatives, in order of preference>]
}

IMPORTANT:
- selected_index is 0-based (first image is index 0)
- If NO candidate is suitable, set selected_index to -1 and explain why
- Be concise but specific in your reasoning
"""

        return base_prompt + type_specific.get(entity_type, type_specific["object"]) + output_format

    def _build_user_prompt(
        self,
        entity_id: str,
        entity_type: str,
        entity_description: str,
        candidates: List[str],
        shot_context: str,
        shot_type: str,
        history_context: Optional[str] = None,
        custom_requirements: Optional[List[str]] = None,
    ) -> Tuple[str, List[str]]:
        """构建用户提示词和图片列表"""

        prompt_parts = [
            f"## Entity to Match",
            f"- **ID**: {entity_id}",
            f"- **Type**: {entity_type}",
            f"- **Description**: {entity_description}",
            "",
            f"## Current Shot Context",
            f"- **Shot Type**: {shot_type}",
            f"- **Shot Description**: {shot_context}",
        ]

        if history_context:
            prompt_parts.extend([
                "",
                f"## History Context",
                f"{history_context}",
            ])

        if custom_requirements:
            prompt_parts.extend([
                "",
                "## Special Requirements",
                *[f"- {req}" for req in custom_requirements],
            ])

        prompt_parts.extend([
            "",
            f"## Candidate Reference Images",
            f"Below are {len(candidates)} candidate images. Analyze each and select the best one.",
            "",
        ])

        # 添加图片占位说明
        for i, path in enumerate(candidates):
            prompt_parts.append(f"**Candidate {i + 1}**: [Image {i + 1}]")

        prompt_parts.extend([
            "",
            "Please analyze all candidates and provide your selection in the specified JSON format.",
        ])

        return "\n".join(prompt_parts), candidates

    def _build_multimodal_content(self, text_prompt: str, image_paths: List[str]) -> List[Dict]:
        """构建多模态内容（文本 + 图片）"""
        content = []

        # 添加图片
        for i, img_path in enumerate(image_paths):
            if os.path.exists(img_path):
                img_data = self._encode_image(img_path)
                if img_data:
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_data}",
                            "detail": "high"
                        }
                    })

        # 添加文本
        content.append({
            "type": "text",
            "text": text_prompt
        })

        return content

    def _encode_image(self, image_path: str) -> Optional[str]:
        """将图片编码为 base64"""
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            if self.verbose:
                print(f"[RefSelAgent] 图片编码失败 {image_path}: {e}")
            return None

    def _call_vlm(self, system_prompt: str, content: List[Dict]) -> str:
        """调用 VLM API"""
        messages = [{"role": "user", "content": content}]

        response = self.llm_client.client.chat.completions.create(
            model=self.llm_client.model,
            messages=[
                {"role": "system", "content": system_prompt},
                *messages
            ],
            max_tokens=1024,
            temperature=self.temperature,
        )

        return response.choices[0].message.content

    def _parse_vlm_response(
        self,
        response: str,
        entity_id: str,
        candidates: List[str],
    ) -> SelectionResult:
        """解析 VLM 响应"""

        # 尝试提取 JSON
        try:
            # 查找 JSON 块
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)

                selected_index = data.get("selected_index", -1)
                confidence = float(data.get("confidence", 0.5))
                reason = data.get("reason", "VLM selection")
                alternatives = data.get("alternatives", [])

                # 验证 index 范围
                if selected_index >= len(candidates):
                    selected_index = -1

                selected_path = candidates[selected_index] if 0 <= selected_index < len(candidates) else None
                alt_paths = [candidates[i] for i in alternatives if 0 <= i < len(candidates)]

                return SelectionResult(
                    entity_id=entity_id,
                    selected_path=selected_path,
                    selected_index=selected_index,
                    confidence=confidence,
                    reason=reason,
                    alternatives=alt_paths,
                    metadata={"vlm_response": response, "analysis": data.get("analysis", {})},
                )

        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"[RefSelAgent] JSON 解析失败: {e}")

        # 解析失败，尝试简单启发式
        # 假设 VLM 提到了 "Candidate X" 或 "image X" 作为选择
        import re
        match = re.search(r'(?:candidate|image)\s*(\d+)', response.lower())
        if match:
            idx = int(match.group(1)) - 1  # 转为 0-based
            if 0 <= idx < len(candidates):
                return SelectionResult(
                    entity_id=entity_id,
                    selected_path=candidates[idx],
                    selected_index=idx,
                    confidence=0.6,  # 降低置信度
                    reason=f"Heuristic extraction from VLM response",
                    metadata={"vlm_response": response, "parse_method": "heuristic"},
                )

        # 完全解析失败，默认选第一个
        return SelectionResult(
            entity_id=entity_id,
            selected_path=candidates[0] if candidates else None,
            selected_index=0 if candidates else -1,
            confidence=0.3,
            reason="Fallback to first candidate (VLM response parsing failed)",
            metadata={"vlm_response": response, "parse_method": "fallback"},
        )

    # ========================================================================
    # Fallback: 传统评分方式
    # ========================================================================

    def _fallback_select(
        self,
        entity_id: str,
        entity_type: str,
        candidates: List[str],
    ) -> SelectionResult:
        """回退到传统评分方式"""
        if self.verbose:
            print(f"[RefSelAgent] 使用传统评分方式 (fallback)")

        scorer = self._get_traditional_scorer()
        ranked = scorer.rank_references(candidates, entity_type=entity_type)

        if not ranked:
            return SelectionResult(
                entity_id=entity_id,
                selected_path=None,
                selected_index=-1,
                confidence=0.0,
                reason="Traditional scoring failed",
            )

        best = ranked[0]
        selected_index = candidates.index(best.crop_path) if best.crop_path in candidates else 0

        return SelectionResult(
            entity_id=entity_id,
            selected_path=best.crop_path,
            selected_index=selected_index,
            confidence=best.final_score,
            reason=f"Traditional scoring: sharpness={best.sharpness:.2f}, "
                   f"id_conf={best.id_confidence:.2f}, frontal={best.frontal_score:.2f}",
            alternatives=[r.crop_path for r in ranked[1:3]],
            metadata={"method": "traditional", "scores": [r.final_score for r in ranked]},
        )


# ============================================================================
# 扩展功能：镜头需求分析
# ============================================================================

class ShotRequirementsAnalyzer:
    """
    分析镜头描述，提取对参考图的需求

    这是 ReferenceSelectionAgent 的辅助模块，
    用于在选择前分析当前镜头对参考图的具体要求。
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self._llm_client = llm_client

    @property
    def llm_client(self) -> LLMClient:
        if self._llm_client is None:
            self._llm_client = LLMClient(model="claude-haiku-4-5")  # 用 Haiku 节省成本
        return self._llm_client

    def analyze(self, shot_text: str, entity_type: str = "character") -> ShotRequirements:
        """
        分析镜头文本，提取对参考图的需求

        Args:
            shot_text: 镜头描述文本
            entity_type: 主要实体类型

        Returns:
            ShotRequirements: 镜头需求分析结果
        """
        prompt = f"""Analyze this shot description and extract requirements for reference image selection.

Shot description: "{shot_text}"
Entity type: {entity_type}

Extract:
1. shot_type: "close-up" / "medium" / "wide" / "unknown"
2. required_angle: "frontal" / "side" / "back" / "any" (what angle is needed?)
3. required_expression: "neutral" / "smiling" / "serious" / "any" (for characters)
4. action_type: "static" / "dynamic" / "dialogue" / "action"
5. lighting_condition: "bright" / "dark" / "mixed" / "any"
6. special_requirements: list of any special requirements mentioned

Output JSON only:
{{"shot_type": "...", "required_angle": "...", "required_expression": "...", "action_type": "...", "lighting_condition": "...", "special_requirements": [...]}}
"""

        response = self.llm_client.chat(prompt, temperature=0.1)

        try:
            # 提取 JSON
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                data = json.loads(response[json_start:json_end])
                return ShotRequirements(
                    shot_type=data.get("shot_type", "medium"),
                    required_angle=data.get("required_angle", "any"),
                    required_expression=data.get("required_expression", "any"),
                    action_type=data.get("action_type", "static"),
                    lighting_condition=data.get("lighting_condition", "any"),
                    special_requirements=data.get("special_requirements", []),
                )
        except:
            pass

        # 解析失败，返回默认值
        return ShotRequirements(
            shot_type="medium",
            required_angle="any",
            required_expression="any",
            action_type="static",
            lighting_condition="any",
        )


# ============================================================================
# 测试
# ============================================================================

if __name__ == "__main__":
    import tempfile
    from PIL import Image

    # 创建测试图片
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建几张测试图片
        test_images = []
        for i in range(3):
            img = Image.new("RGB", (256, 256), color=(i * 80, i * 40, 200 - i * 50))
            path = os.path.join(tmpdir, f"test_{i}.jpg")
            img.save(path)
            test_images.append(path)

        print("=== 测试 ReferenceSelectionAgent ===")
        print(f"测试图片: {test_images}")

        # 测试 Agent（先测试 fallback 模式）
        agent = ReferenceSelectionAgent(enable_fallback=True, verbose=True)

        result = agent.select_best_reference(
            entity_id="char_test",
            entity_type="character",
            entity_description="A young man with short brown hair",
            candidates=test_images,
            shot_context="A close-up of the man smiling at the camera",
            shot_type="close-up",
        )

        print(f"\n选择结果:")
        print(f"  selected_path: {result.selected_path}")
        print(f"  selected_index: {result.selected_index}")
        print(f"  confidence: {result.confidence}")
        print(f"  reason: {result.reason}")
