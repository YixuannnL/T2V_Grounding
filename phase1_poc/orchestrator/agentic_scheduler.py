"""
orchestrator/agentic_scheduler.py

Quality-Aware Shot Scheduling via Agentic Planning

核心洞察：叙事顺序 ≠ 最优生成顺序

传统方法按 shot 1 → shot 2 → ... → shot N 线性执行，隐含假设是：
- 实体首次出现时能获得足够好的参考
- Reference 质量在 shot 间均匀分布

但现实中，同一实体在不同 shot 的 grounding 质量差异巨大：
- Close-up shot: 实体占画面 60% → 高质量参考
- Wide shot:     实体占画面 5%  → 低质量参考

本模块使用 LLM 分析 script，构建最优执行 DAG：
1. 预测每个 shot 对每个实体的 grounding 质量
2. 识别每个实体的最佳 reference source shot
3. 构建执行依赖图，确定最优执行顺序
"""

import os
import sys
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.llm_client import LLMClient


class ShotType(Enum):
    CLOSEUP = "close-up"
    MEDIUM = "medium"
    WIDE = "wide"
    ESTABLISHING = "establishing"
    UNKNOWN = "unknown"


@dataclass
class QualityPrediction:
    """单个 shot-entity pair 的 grounding 质量预测"""
    shot_id: int
    entity_id: str
    predicted_quality: float  # 0-1
    shot_type: ShotType
    factors: Dict[str, float] = field(default_factory=dict)  # 各因素得分
    reasoning: str = ""


@dataclass
class ScheduleResult:
    """调度分析结果"""
    # 质量矩阵: shot_id -> entity_id -> predicted_quality
    quality_matrix: Dict[int, Dict[str, float]]

    # 每个实体的最佳 reference source shot
    reference_sources: Dict[str, int]  # entity_id -> shot_id

    # 优化后的执行顺序
    execution_order: List[int]  # shot_ids in execution order

    # 叙事顺序（原始顺序）
    narrative_order: List[int]

    # 依赖图边: (from_shot, to_shot, reason)
    dependencies: List[Tuple[int, int, str]]

    # 调度分析的 reasoning
    reasoning: str = ""

    # 是否使用了 DAG 优化
    dag_optimized: bool = False

    # 预期收益分析
    expected_benefit: Dict[str, float] = field(default_factory=dict)


@dataclass
class EntityInfo:
    """实体信息（用于调度分析）"""
    entity_id: str
    entity_type: str  # character, object, location
    text_description: str
    first_appearance_shot: int = -1  # 首次出现的 shot


@dataclass
class ShotInfo:
    """Shot 信息（用于调度分析）"""
    shot_id: int
    text: str
    shot_type: ShotType = ShotType.UNKNOWN
    entities: List[str] = field(default_factory=list)  # entity_ids in this shot


# ── LLM Prompts ──────────────────────────────────────────────────────────────

SHOT_TYPE_ANALYSIS_PROMPT = """你是一个专业的影视镜头分析师。

给定一个镜头描述，请判断其镜头类型：

镜头类型定义：
- close-up: 特写镜头，聚焦于人物面部或物体细节，主体占画面 50%+
  关键词: close-up, closeup, tight shot, focuses on face, detail shot
- medium: 中景镜头，人物腰部以上，主体占画面 20-50%
  关键词: medium shot, waist shot, mid shot
- wide: 远景/全景镜头，展示完整场景，人物占画面 < 20%
  关键词: wide shot, full shot, long shot, wide angle
- establishing: 建立镜头，通常用于展示场景/地点
  关键词: establishing shot, exterior, landscape, overview

输出 JSON 格式：
{
  "shot_type": "close-up|medium|wide|establishing",
  "confidence": 0.0-1.0,
  "reasoning": "简短理由"
}
"""

SCHEDULING_ANALYSIS_PROMPT = """你是一个视频生成流程优化专家。

## 背景
在多镜头视频生成中，我们需要为每个实体（人物、物体、场景）维护 visual reference。
Reference 质量直接影响跨镜头一致性。

关键洞察：**同一实体在不同镜头的 grounding 质量差异巨大**
- Close-up 镜头：实体占画面大，可获得高分辨率参考 (质量 0.8-0.95)
- Medium 镜头：实体中等大小，参考质量中等 (质量 0.5-0.7)
- Wide 镜头：实体很小，参考质量低 (质量 0.2-0.4)
- 遮挡/暗光场景：参考质量进一步降低

## 你的任务
分析给定的 script，确定最优 shot 执行顺序。

## 输入
- Shots: 镜头列表，包含 shot_id 和描述
- Entities: 实体列表，包含 entity_id、类型和描述

## 分析步骤

### 1. 预测 Grounding 质量
对每个 (shot, entity) pair 预测 grounding 质量 (0-1):

基础分数（按镜头类型）：
- close-up + 实体是焦点: 0.9
- close-up + 实体出现但非焦点: 0.7
- medium shot: 0.6
- wide shot: 0.35
- establishing shot: 0.25

修正因子：
- 良好光线: ×1.0
- 暗光/逆光: ×0.7
- 部分遮挡: ×0.6
- 完全遮挡: ×0.1
- 侧面/背面: ×0.5 (仅 character)
- 正面清晰: ×1.0 (仅 character)

### 2. 识别 Reference Source
对每个实体，找到能提供最高质量参考的 shot（reference source）。
这个 shot 应该优先执行。

### 3. 构建依赖图
规则：
- 实体的 reference source shot 没有依赖（最先执行）
- 其他包含该实体的 shot 依赖其 reference source
- 如果一个 shot 包含多个实体，它依赖所有相关实体的 reference source

### 4. 确定执行顺序
- 拓扑排序依赖图
- 同层级的 shot 按原始叙事顺序排列
- 如果无法优化（如线性依赖），保持原始顺序

### 5. 评估收益
计算 DAG 优化相对于线性执行的预期收益：
- anchor_quality_improvement: 锚点参考质量提升
- affected_entities: 受益的实体数量

## 输出格式
```json
{
  "quality_matrix": {
    "1": {"char_alex": 0.35, "loc_cafe": 0.6},
    "2": {"char_alex": 0.9, "loc_cafe": 0.4},
    ...
  },
  "reference_sources": {
    "char_alex": 2,
    "loc_cafe": 1
  },
  "execution_order": [2, 1, 3, 4],
  "dependencies": [
    [2, 1, "char_alex reference"],
    [2, 3, "char_alex reference"]
  ],
  "reasoning": "详细分析说明...",
  "expected_benefit": {
    "anchor_quality_improvement": 0.55,
    "affected_entities": 2,
    "should_optimize": true
  }
}
```

## 特殊情况
1. 如果所有 shot 都是类似镜头类型，线性执行可能是最优的
2. 如果 shot 间有严格的叙事依赖（如动作连续性），标注并保持顺序
3. 对于 "progressive reveal" 场景（实体逐渐显露），优先执行揭示镜头

请分析以下 script:
"""


class AgenticScheduler:
    """
    Quality-Aware Shot Scheduler

    使用 LLM 分析 script，构建最优执行 DAG

    Usage:
        scheduler = AgenticScheduler()
        schedule = scheduler.analyze(shots, entities)

        # 使用优化后的执行顺序
        for shot_id in schedule.execution_order:
            process_shot(shot_id)
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6-Anthropic",
        min_benefit_threshold: float = 0.15,  # 最小收益阈值，低于此值不优化
        enable_heuristic_fallback: bool = True,  # LLM 失败时使用启发式
        verbose: bool = True,
    ):
        self.llm = LLMClient(model=model)
        self.min_benefit_threshold = min_benefit_threshold
        self.enable_heuristic_fallback = enable_heuristic_fallback
        self.verbose = verbose

    def analyze(
        self,
        shots: List[ShotInfo],
        entities: List[EntityInfo],
    ) -> ScheduleResult:
        """
        分析 script，返回优化后的执行顺序

        Args:
            shots: Shot 信息列表
            entities: Entity 信息列表

        Returns:
            ScheduleResult 包含执行顺序和分析结果
        """
        narrative_order = [s.shot_id for s in shots]

        # Step 1: 分析每个 shot 的镜头类型
        for shot in shots:
            if shot.shot_type == ShotType.UNKNOWN:
                shot.shot_type = self._analyze_shot_type(shot.text)

        # Step 2: 使用 LLM 进行完整调度分析
        try:
            result = self._llm_schedule_analysis(shots, entities)

            # 检查是否值得优化
            if result.expected_benefit.get("should_optimize", False):
                result.dag_optimized = True
                if self.verbose:
                    print(f"[Scheduler] DAG 优化启用 | "
                          f"预期质量提升: +{result.expected_benefit.get('anchor_quality_improvement', 0):.2f}")
                    print(f"[Scheduler] 执行顺序: {result.execution_order} "
                          f"(原始: {narrative_order})")
            else:
                # 收益不足，保持线性顺序
                result.execution_order = narrative_order
                result.dag_optimized = False
                if self.verbose:
                    print(f"[Scheduler] DAG 优化跳过 | 收益不足 "
                          f"(threshold={self.min_benefit_threshold})")

            return result

        except Exception as e:
            if self.verbose:
                print(f"[Scheduler] LLM 分析失败: {e}")

            if self.enable_heuristic_fallback:
                return self._heuristic_schedule(shots, entities)
            else:
                # 返回线性顺序
                return ScheduleResult(
                    quality_matrix={},
                    reference_sources={},
                    execution_order=narrative_order,
                    narrative_order=narrative_order,
                    dependencies=[],
                    reasoning=f"LLM 分析失败: {e}，使用线性顺序",
                    dag_optimized=False,
                )

    def _analyze_shot_type(self, shot_text: str) -> ShotType:
        """分析单个 shot 的镜头类型"""
        text_lower = shot_text.lower()

        # 基于关键词的快速判断
        if any(kw in text_lower for kw in ["close-up", "closeup", "close up", "tight shot", "focuses on"]):
            return ShotType.CLOSEUP
        if any(kw in text_lower for kw in ["wide shot", "wide angle", "full shot", "long shot"]):
            return ShotType.WIDE
        if any(kw in text_lower for kw in ["establishing", "exterior of", "overview"]):
            return ShotType.ESTABLISHING

        # 默认为 medium
        return ShotType.MEDIUM

    def _llm_schedule_analysis(
        self,
        shots: List[ShotInfo],
        entities: List[EntityInfo],
    ) -> ScheduleResult:
        """使用 LLM 进行完整调度分析"""

        # 构建 prompt
        shots_desc = []
        for s in shots:
            shots_desc.append(f"  Shot {s.shot_id} [{s.shot_type.value}]: {s.text}")

        entities_desc = []
        for e in entities:
            entities_desc.append(f"  {e.entity_id} ({e.entity_type}): {e.text_description}")

        user_message = f"""
## Shots
{chr(10).join(shots_desc)}

## Entities
{chr(10).join(entities_desc)}

请分析并输出 JSON。
"""

        response = self.llm.chat(
            user_message=user_message,
            system=SCHEDULING_ANALYSIS_PROMPT,
            max_tokens=4096,
            temperature=0.2,
        ).strip()

        # 解析 JSON
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
            response = response.strip()

        data = json.loads(response)

        # 构建 ScheduleResult
        quality_matrix = {}
        for shot_id_str, entity_scores in data.get("quality_matrix", {}).items():
            shot_id = int(shot_id_str)
            quality_matrix[shot_id] = entity_scores

        reference_sources = data.get("reference_sources", {})
        execution_order = data.get("execution_order", [s.shot_id for s in shots])

        dependencies = []
        for dep in data.get("dependencies", []):
            if len(dep) >= 2:
                reason = dep[2] if len(dep) > 2 else ""
                dependencies.append((dep[0], dep[1], reason))

        expected_benefit = data.get("expected_benefit", {})

        # 判断是否应该优化
        improvement = expected_benefit.get("anchor_quality_improvement", 0)
        if improvement >= self.min_benefit_threshold:
            expected_benefit["should_optimize"] = True
        else:
            expected_benefit["should_optimize"] = False

        return ScheduleResult(
            quality_matrix=quality_matrix,
            reference_sources=reference_sources,
            execution_order=execution_order,
            narrative_order=[s.shot_id for s in shots],
            dependencies=dependencies,
            reasoning=data.get("reasoning", ""),
            dag_optimized=False,  # 由调用方设置
            expected_benefit=expected_benefit,
        )

    def _heuristic_schedule(
        self,
        shots: List[ShotInfo],
        entities: List[EntityInfo],
    ) -> ScheduleResult:
        """
        启发式调度（LLM 失败时的 fallback）

        策略：
        1. 识别 close-up shots
        2. 将包含实体的 close-up shots 提前执行
        3. 保持其他 shot 的相对顺序
        """
        if self.verbose:
            print(f"[Scheduler] 使用启发式调度...")

        narrative_order = [s.shot_id for s in shots]

        # 按镜头类型分组
        closeup_shots = []
        other_shots = []

        for shot in shots:
            if shot.shot_type == ShotType.CLOSEUP:
                closeup_shots.append(shot)
            else:
                other_shots.append(shot)

        # 如果没有 close-up，保持原顺序
        if not closeup_shots:
            return ScheduleResult(
                quality_matrix={},
                reference_sources={},
                execution_order=narrative_order,
                narrative_order=narrative_order,
                dependencies=[],
                reasoning="无 close-up 镜头，保持线性顺序",
                dag_optimized=False,
            )

        # 预测质量矩阵（启发式）
        quality_matrix = {}
        base_scores = {
            ShotType.CLOSEUP: 0.85,
            ShotType.MEDIUM: 0.55,
            ShotType.WIDE: 0.35,
            ShotType.ESTABLISHING: 0.25,
            ShotType.UNKNOWN: 0.5,
        }

        for shot in shots:
            quality_matrix[shot.shot_id] = {}
            base_score = base_scores.get(shot.shot_type, 0.5)
            for entity in entities:
                # 假设所有实体出现在所有 shot 中（简化）
                quality_matrix[shot.shot_id][entity.entity_id] = base_score

        # 识别 reference sources（每个实体选最早的 close-up）
        reference_sources = {}
        for entity in entities:
            best_shot = None
            best_score = 0
            for shot in shots:
                score = quality_matrix[shot.shot_id].get(entity.entity_id, 0)
                if score > best_score:
                    best_score = score
                    best_shot = shot.shot_id
            if best_shot is not None:
                reference_sources[entity.entity_id] = best_shot

        # 构建执行顺序：close-up 优先
        # 但要保持 close-up 之间的相对顺序
        execution_order = []

        # 先添加 close-up shots（按原始顺序）
        for shot in closeup_shots:
            execution_order.append(shot.shot_id)

        # 再添加其他 shots（按原始顺序）
        for shot in other_shots:
            execution_order.append(shot.shot_id)

        # 计算预期收益
        linear_anchor_quality = base_scores.get(shots[0].shot_type, 0.5) if shots else 0
        dag_anchor_quality = base_scores.get(ShotType.CLOSEUP, 0.85) if closeup_shots else linear_anchor_quality
        improvement = dag_anchor_quality - linear_anchor_quality

        should_optimize = improvement >= self.min_benefit_threshold

        if not should_optimize:
            execution_order = narrative_order

        return ScheduleResult(
            quality_matrix=quality_matrix,
            reference_sources=reference_sources,
            execution_order=execution_order,
            narrative_order=narrative_order,
            dependencies=[],
            reasoning=f"启发式调度: close-up 优先 (预期提升 +{improvement:.2f})",
            dag_optimized=should_optimize,
            expected_benefit={
                "anchor_quality_improvement": improvement,
                "affected_entities": len(entities),
                "should_optimize": should_optimize,
            },
        )

    def get_shot_priority_info(
        self,
        schedule: ScheduleResult,
        shot_id: int,
    ) -> Dict:
        """
        获取单个 shot 的优先级信息（用于 pipeline 决策）

        Returns:
            {
                "is_reference_source": bool,  # 是否是某实体的 reference source
                "source_for_entities": [...], # 为哪些实体提供 reference
                "depends_on": [...],          # 依赖哪些 shot
                "execution_rank": int,        # 在执行顺序中的位置
            }
        """
        source_for = [
            eid for eid, src_shot in schedule.reference_sources.items()
            if src_shot == shot_id
        ]

        depends_on = [
            from_shot for from_shot, to_shot, _ in schedule.dependencies
            if to_shot == shot_id
        ]

        exec_rank = schedule.execution_order.index(shot_id) if shot_id in schedule.execution_order else -1

        return {
            "is_reference_source": len(source_for) > 0,
            "source_for_entities": source_for,
            "depends_on": depends_on,
            "execution_rank": exec_rank,
        }


# ── Utility Functions ────────────────────────────────────────────────────────

def build_entity_shot_matrix(
    shots: List[ShotInfo],
    entities: List[EntityInfo],
) -> Dict[str, List[int]]:
    """
    构建实体-shot 矩阵，记录每个实体出现在哪些 shot 中

    Returns:
        {entity_id: [shot_ids where entity appears]}
    """
    matrix = {e.entity_id: [] for e in entities}

    for shot in shots:
        for entity_id in shot.entities:
            if entity_id in matrix:
                matrix[entity_id].append(shot.shot_id)

    return matrix


def visualize_schedule(schedule: ScheduleResult) -> str:
    """生成调度结果的可视化字符串"""
    lines = []
    lines.append("=" * 60)
    lines.append("Shot Scheduling Analysis")
    lines.append("=" * 60)

    lines.append(f"\nNarrative Order: {schedule.narrative_order}")
    lines.append(f"Execution Order: {schedule.execution_order}")
    lines.append(f"DAG Optimized: {schedule.dag_optimized}")

    if schedule.reference_sources:
        lines.append(f"\nReference Sources:")
        for entity_id, shot_id in schedule.reference_sources.items():
            lines.append(f"  {entity_id} <- Shot {shot_id}")

    if schedule.dependencies:
        lines.append(f"\nDependencies:")
        for from_shot, to_shot, reason in schedule.dependencies:
            lines.append(f"  Shot {from_shot} -> Shot {to_shot} ({reason})")

    if schedule.expected_benefit:
        lines.append(f"\nExpected Benefit:")
        for k, v in schedule.expected_benefit.items():
            lines.append(f"  {k}: {v}")

    if schedule.reasoning:
        lines.append(f"\nReasoning: {schedule.reasoning[:200]}...")

    lines.append("=" * 60)
    return "\n".join(lines)


# ── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 测试用例：Close-up 后置场景
    shots = [
        ShotInfo(shot_id=1, text="Wide shot - Sarah walks through crowded marketplace"),
        ShotInfo(shot_id=2, text="Medium shot - Sarah stops at a fruit stall"),
        ShotInfo(shot_id=3, text="Close-up of Sarah's face as she examines an apple"),
        ShotInfo(shot_id=4, text="Wide shot - Sarah continues walking, apple in hand"),
    ]

    entities = [
        EntityInfo(entity_id="char_sarah", entity_type="character",
                   text_description="Sarah, a young woman in casual clothes"),
        EntityInfo(entity_id="loc_marketplace", entity_type="location",
                   text_description="Busy outdoor marketplace"),
    ]

    scheduler = AgenticScheduler(verbose=True)
    result = scheduler.analyze(shots, entities)

    print(visualize_schedule(result))
