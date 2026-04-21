"""
agents/
T2V Grounding Agentic 模块

包含各种 Agent，用于替代传统的固定规则/阈值决策：
  - ReferenceSelectionAgent: 智能参考图选择（替代 InsightFace 打分）
  - ReferenceSelectionStrategy: 策略封装（支持 traditional/agent/hybrid 模式）
  - (future) GenerationQualityAgent: 生成质量评估与自愈
  - (future) ConsistencyMonitorAgent: 跨镜头一致性监控
"""

from .reference_selection_agent import (
    ReferenceSelectionAgent,
    SelectionResult,
    ShotRequirements,
    ShotRequirementsAnalyzer,
)
from .reference_selection_strategy import (
    ReferenceSelectionStrategy,
    SelectedReference,
    SelectionMode,
    create_selection_strategy,
    migrate_pipeline_to_agent_selection,
)

__all__ = [
    # Core Agent
    "ReferenceSelectionAgent",
    "SelectionResult",
    "ShotRequirements",
    "ShotRequirementsAnalyzer",
    # Strategy wrapper
    "ReferenceSelectionStrategy",
    "SelectedReference",
    "SelectionMode",
    # Helper functions
    "create_selection_strategy",
    "migrate_pipeline_to_agent_selection",
]
