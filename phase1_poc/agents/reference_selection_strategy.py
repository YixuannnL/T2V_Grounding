"""
agents/reference_selection_strategy.py

参考图选择策略封装 - 统一接口支持传统模式和 Agent 模式

使用方式:
    # 传统模式（InsightFace 打分）
    strategy = ReferenceSelectionStrategy(mode="traditional")

    # Agent 模式（VLM 智能选择）
    strategy = ReferenceSelectionStrategy(mode="agent")

    # 混合模式（Agent 主导，传统 fallback）
    strategy = ReferenceSelectionStrategy(mode="hybrid")

    # 统一接口选择参考图
    selected = strategy.select(
        entity_id="char_alex",
        entity_type="character",
        entity_description="...",
        candidates=[ref1, ref2, ref3],
        shot_context="...",
        shot_type="close-up",
    )
"""

import os
import sys
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reference_manager.registry import EntityRegistry, ReferenceEntry


class SelectionMode(Enum):
    TRADITIONAL = "traditional"  # 传统 InsightFace 打分
    AGENT = "agent"              # 纯 VLM Agent 选择
    HYBRID = "hybrid"            # Agent 优先，失败时 fallback


@dataclass
class SelectedReference:
    """选择结果（统一格式）"""
    entity_id: str
    crop_path: Optional[str]
    quality_score: float
    id_confidence: float
    source_shot: int
    selection_method: str       # "traditional" / "agent" / "fallback"
    reason: str
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ReferenceSelectionStrategy:
    """
    参考图选择策略 - 统一接口

    支持三种模式:
      - traditional: 传统 InsightFace 打分 + 固定权重公式
      - agent: 纯 VLM Agent 智能选择
      - hybrid: Agent 优先，VLM 失败时自动 fallback 到传统

    推荐使用 hybrid 模式，兼顾智能和稳定性。
    """

    def __init__(
        self,
        mode: str = "hybrid",
        agent_model: str = "claude-sonnet-4-6",
        verbose: bool = True,
    ):
        self.mode = SelectionMode(mode)
        self.agent_model = agent_model
        self.verbose = verbose

        # 延迟初始化
        self._agent = None
        self._traditional_scorer = None

    @property
    def agent(self):
        if self._agent is None:
            from agents.reference_selection_agent import ReferenceSelectionAgent
            self._agent = ReferenceSelectionAgent(
                model=self.agent_model,
                enable_fallback=(self.mode == SelectionMode.HYBRID),
                verbose=self.verbose,
            )
        return self._agent

    @property
    def traditional_scorer(self):
        if self._traditional_scorer is None:
            from visual_grounding.reid import ReferenceQualityScorer
            self._traditional_scorer = ReferenceQualityScorer()
        return self._traditional_scorer

    def select(
        self,
        entity_id: str,
        entity_type: str,
        entity_description: str,
        candidates: List[ReferenceEntry],
        shot_context: str,
        shot_type: str = "medium",
        **kwargs,
    ) -> Optional[SelectedReference]:
        """
        从候选参考图中选择最佳参考

        Args:
            entity_id: 实体 ID
            entity_type: 实体类型
            entity_description: 实体描述
            candidates: 候选 ReferenceEntry 列表（来自 registry）
            shot_context: 当前镜头描述
            shot_type: 镜头类型

        Returns:
            SelectedReference: 选择结果
        """
        if not candidates:
            return None

        if self.mode == SelectionMode.TRADITIONAL:
            return self._traditional_select(entity_id, entity_type, candidates)

        elif self.mode in (SelectionMode.AGENT, SelectionMode.HYBRID):
            return self._agent_select(
                entity_id=entity_id,
                entity_type=entity_type,
                entity_description=entity_description,
                candidates=candidates,
                shot_context=shot_context,
                shot_type=shot_type,
            )

        return None

    def _traditional_select(
        self,
        entity_id: str,
        entity_type: str,
        candidates: List[ReferenceEntry],
    ) -> Optional[SelectedReference]:
        """传统评分选择"""
        crop_paths = [r.crop_path for r in candidates]

        # 使用传统评分器
        ranked = self.traditional_scorer.rank_references(crop_paths, entity_type=entity_type)

        if not ranked:
            return None

        best_score = ranked[0]
        # 找到对应的 ReferenceEntry
        best_entry = next((r for r in candidates if r.crop_path == best_score.crop_path), candidates[0])

        return SelectedReference(
            entity_id=entity_id,
            crop_path=best_score.crop_path,
            quality_score=best_score.final_score,
            id_confidence=best_score.id_confidence,
            source_shot=best_entry.shot_id,
            selection_method="traditional",
            reason=f"sharpness={best_score.sharpness:.2f}, id_conf={best_score.id_confidence:.2f}, frontal={best_score.frontal_score:.2f}",
        )

    def _agent_select(
        self,
        entity_id: str,
        entity_type: str,
        entity_description: str,
        candidates: List[ReferenceEntry],
        shot_context: str,
        shot_type: str,
    ) -> Optional[SelectedReference]:
        """Agent 智能选择"""
        crop_paths = [r.crop_path for r in candidates]

        result = self.agent.select_best_reference(
            entity_id=entity_id,
            entity_type=entity_type,
            entity_description=entity_description,
            candidates=crop_paths,
            shot_context=shot_context,
            shot_type=shot_type,
        )

        if result.selected_path is None:
            return None

        # 找到对应的 ReferenceEntry
        selected_entry = next(
            (r for r in candidates if r.crop_path == result.selected_path),
            candidates[result.selected_index] if 0 <= result.selected_index < len(candidates) else candidates[0]
        )

        method = "agent"
        if result.metadata.get("method") == "traditional":
            method = "fallback"

        return SelectedReference(
            entity_id=entity_id,
            crop_path=result.selected_path,
            quality_score=result.confidence,
            id_confidence=selected_entry.id_confidence,
            source_shot=selected_entry.shot_id,
            selection_method=method,
            reason=result.reason,
            metadata=result.metadata,
        )

    def select_for_entities(
        self,
        entities: List[Any],  # List[Entity]
        registry: EntityRegistry,
        shot_context: str,
        shot_type: str = "medium",
        min_quality: float = 0.3,
        max_candidates: int = 6,
    ) -> Dict[str, SelectedReference]:
        """
        为多个实体批量选择参考图

        Args:
            entities: 实体列表
            registry: 参考图注册库
            shot_context: 当前镜头描述
            shot_type: 镜头类型
            min_quality: 候选最低质量
            max_candidates: 每个实体最多考虑几个候选

        Returns:
            Dict[entity_id, SelectedReference]
        """
        results = {}

        for entity in entities:
            # 从 registry 获取候选
            refs = registry.query(
                entity.entity_id,
                top_k=max_candidates,
                min_quality=min_quality,
                anchor_strategy="earliest_good",
            )

            if not refs:
                if self.verbose:
                    print(f"[Strategy] {entity.entity_id}: 无候选参考图")
                continue

            selected = self.select(
                entity_id=entity.entity_id,
                entity_type=entity.type,
                entity_description=entity.text_description,
                candidates=refs,
                shot_context=shot_context,
                shot_type=shot_type,
            )

            if selected:
                results[entity.entity_id] = selected
                if self.verbose:
                    print(f"[Strategy] {entity.entity_id}: 选择 shot={selected.source_shot} "
                          f"({selected.selection_method}) - {selected.reason[:50]}...")

        return results


# ============================================================================
# Pipeline 集成辅助函数
# ============================================================================

def create_selection_strategy(
    mode: str = "hybrid",
    agent_model: str = "claude-sonnet-4-6",
) -> ReferenceSelectionStrategy:
    """
    创建参考图选择策略

    Args:
        mode: "traditional" / "agent" / "hybrid"
        agent_model: Agent 使用的模型

    Returns:
        ReferenceSelectionStrategy 实例
    """
    return ReferenceSelectionStrategy(
        mode=mode,
        agent_model=agent_model,
    )


def migrate_pipeline_to_agent_selection(
    pipeline_instance,
    mode: str = "hybrid",
    agent_model: str = "claude-sonnet-4-6",
):
    """
    为现有 pipeline 实例启用 Agent 参考图选择

    Args:
        pipeline_instance: T2VGroundingPipeline 实例
        mode: 选择模式
        agent_model: Agent 模型

    用法:
        pipeline = T2VGroundingPipeline(...)
        migrate_pipeline_to_agent_selection(pipeline, mode="hybrid")
    """
    pipeline_instance.ref_selection_strategy = ReferenceSelectionStrategy(
        mode=mode,
        agent_model=agent_model,
        verbose=True,
    )
    print(f"[Migration] Pipeline 已启用 Agent 参考图选择 (mode={mode})")


# ============================================================================
# 测试
# ============================================================================

if __name__ == "__main__":
    print("=== 测试 ReferenceSelectionStrategy ===")

    # 测试传统模式
    print("\n1. 传统模式测试:")
    strategy_trad = ReferenceSelectionStrategy(mode="traditional")
    print(f"   模式: {strategy_trad.mode}")

    # 测试 hybrid 模式
    print("\n2. Hybrid 模式测试:")
    strategy_hybrid = ReferenceSelectionStrategy(mode="hybrid")
    print(f"   模式: {strategy_hybrid.mode}")

    print("\n策略创建成功!")
