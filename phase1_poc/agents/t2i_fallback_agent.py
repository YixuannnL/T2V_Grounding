"""
agents/t2i_fallback_agent.py

T2I Fallback Agent: 当 DAG 调度无法获得高质量 Reference 时，智能决策是否 Fallback 到 T2I 生成

核心策略：DAG 优先 + T2I 兜底
─────────────────────────────────────────────────────────────
1. LLM 预筛：根据 DAG 调度的质量预测，决定 retry 预算
   - 质量 < 0.3: 直接 T2I（所有镜头都是背影/遮挡）
   - 质量 0.3-0.5: 尝试 DAG，低 retry 预算（1 次）
   - 质量 >= 0.5: 正常 DAG 流程（3 次）

2. 实际验证：执行过程中根据 grounding 结果动态决策
   - 若 retry 用尽仍失败 → Fallback T2I

3. 所有决策过程都有详细 log，便于 debug

为什么优先 DAG 而非 T2I：
- T2V 质量通常优于 S2V（动作更自然）
- T2I 生成的 reference 可能与场景不协调
- 保持系统的"自举"特性
"""

import os
import sys
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.llm_client import LLMClient


class FallbackStrategy(Enum):
    """Fallback 策略枚举"""
    DAG_NORMAL = "dag_normal"           # 正常 DAG 流程，完整 retry 预算
    DAG_LOW_BUDGET = "dag_low_budget"   # 尝试 DAG，低 retry 预算
    T2I_IMMEDIATE = "t2i_immediate"     # 直接 T2I，跳过 DAG 尝试


@dataclass
class FallbackDecision:
    """单个实体的 Fallback 决策结果"""
    entity_id: str
    entity_type: str
    strategy: FallbackStrategy
    max_retries: int
    predicted_quality: float
    reasoning: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class FallbackAttemptLog:
    """单次尝试的日志记录"""
    entity_id: str
    attempt_number: int
    max_attempts: int
    action: str                    # "dag_attempt" | "t2i_fallback" | "success"
    grounding_quality: float       # 本次 grounding 得到的质量（-1 表示失败）
    failure_reason: Optional[str]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class FallbackSessionLog:
    """整个 Fallback 会话的日志（一个实体从开始到最终获得 reference 的完整过程）"""
    entity_id: str
    entity_type: str
    initial_decision: FallbackDecision
    attempts: List[FallbackAttemptLog] = field(default_factory=list)
    final_outcome: str = ""        # "dag_success" | "t2i_success" | "failed"
    final_reference_path: str = ""
    total_time_seconds: float = 0.0


# ── LLM Prompts ──────────────────────────────────────────────────────────────

T2I_DECISION_PROMPT = """你是一个视频生成系统的决策 Agent。

## 背景
我们的系统使用 DAG 调度来优化 multi-shot 视频生成的执行顺序，目的是为每个实体获得高质量的 visual reference。

但有些情况下，DAG 调度也无法获得好的 reference：
- 实体在所有镜头都是背影
- 实体始终被遮挡
- 所有镜头都是远景，实体太小

此时需要 fallback 到 T2I（Text-to-Image）直接生成 reference。

## 你的任务
分析给定实体的情况，决定是否需要 T2I fallback。

## 输入
- 实体信息：类型、描述
- DAG 调度预测的质量分数（0-1）
- 各 shot 的质量预测明细

## 决策标准
1. predicted_quality >= 0.5:
   → DAG 应该能成功，正常流程（max_retries=3）

2. 0.3 <= predicted_quality < 0.5:
   → 边缘 case，值得尝试但降低期望（max_retries=1）

3. predicted_quality < 0.3:
   → DAG 大概率失败，直接 T2I（max_retries=0）

## 输出格式（JSON）
{
  "strategy": "dag_normal" | "dag_low_budget" | "t2i_immediate",
  "max_retries": 0-3,
  "reasoning": "简短解释决策理由"
}
"""


class T2IFallbackAgent:
    """
    T2I Fallback 决策 Agent

    负责：
    1. 根据 DAG 调度预测，决定每个实体的 fallback 策略
    2. 在执行过程中记录每次尝试
    3. 管理 retry 预算，决定何时 fallback 到 T2I
    4. 生成详细的决策日志

    Usage:
        agent = T2IFallbackAgent()

        # 阶段 1：执行前决策
        decision = agent.pre_execution_decision(entity, schedule)

        # 阶段 2：执行中记录
        agent.log_attempt(entity_id, attempt_num, quality, failure_reason)

        # 阶段 3：失败后决策
        next_action = agent.post_failure_decision(entity_id, current_attempt)

        # 导出日志
        agent.export_logs("fallback_log.json")
    """

    def __init__(
        self,
        quality_threshold_immediate: float = 0.3,
        quality_threshold_normal: float = 0.5,
        default_max_retries: int = 3,
        low_budget_retries: int = 1,
        use_llm_decision: bool = True,
        llm_model: str = "claude-sonnet-4-6",
        verbose: bool = True,
    ):
        """
        Args:
            quality_threshold_immediate: 低于此阈值直接 T2I
            quality_threshold_normal: 高于此阈值正常 DAG 流程
            default_max_retries: 正常情况下的最大重试次数
            low_budget_retries: 边缘 case 的重试次数
            use_llm_decision: 是否使用 LLM 辅助决策（否则纯规则）
            llm_model: LLM 模型
            verbose: 是否打印详细日志
        """
        self.threshold_immediate = quality_threshold_immediate
        self.threshold_normal = quality_threshold_normal
        self.default_max_retries = default_max_retries
        self.low_budget_retries = low_budget_retries
        self.use_llm_decision = use_llm_decision
        self.verbose = verbose

        if use_llm_decision:
            self.llm = LLMClient(model=llm_model)
        else:
            self.llm = None

        # 存储所有实体的 session log
        self.session_logs: Dict[str, FallbackSessionLog] = {}

    # ── 阶段 1：执行前决策 ────────────────────────────────────────────────────

    def pre_execution_decision(
        self,
        entity_id: str,
        entity_type: str,
        entity_description: str,
        schedule_result,  # ScheduleResult from agentic_scheduler
    ) -> FallbackDecision:
        """
        执行前决策：分析 DAG 调度结果，决定该实体的 fallback 策略

        Args:
            entity_id: 实体 ID
            entity_type: 实体类型（character/object/location）
            entity_description: 实体文本描述
            schedule_result: DAG 调度结果（包含 quality_matrix）

        Returns:
            FallbackDecision: 包含策略、retry 预算、理由
        """
        # 获取该实体的最佳预测质量
        best_shot = schedule_result.reference_sources.get(entity_id)
        if best_shot is None:
            # 实体不在任何 shot 的预测中
            decision = FallbackDecision(
                entity_id=entity_id,
                entity_type=entity_type,
                strategy=FallbackStrategy.T2I_IMMEDIATE,
                max_retries=0,
                predicted_quality=0.0,
                reasoning="实体不在任何 shot 的质量预测中，可能是新实体或脚本问题"
            )
            self._log_decision(decision)
            return decision

        best_quality = schedule_result.quality_matrix.get(best_shot, {}).get(
            entity_id, 0.0
        )

        # 收集所有 shot 的质量预测（用于 LLM 分析）
        shot_qualities = {}
        for shot_id, entity_scores in schedule_result.quality_matrix.items():
            if entity_id in entity_scores:
                shot_qualities[shot_id] = entity_scores[entity_id]

        # 决策逻辑
        if self.use_llm_decision and self.llm is not None:
            decision = self._llm_decision(
                entity_id, entity_type, entity_description,
                best_quality, shot_qualities
            )
        else:
            decision = self._rule_based_decision(
                entity_id, entity_type, best_quality
            )

        # 初始化 session log
        self.session_logs[entity_id] = FallbackSessionLog(
            entity_id=entity_id,
            entity_type=entity_type,
            initial_decision=decision,
        )

        self._log_decision(decision)
        return decision

    def _rule_based_decision(
        self,
        entity_id: str,
        entity_type: str,
        best_quality: float,
    ) -> FallbackDecision:
        """纯规则决策（不使用 LLM）"""
        if best_quality < self.threshold_immediate:
            return FallbackDecision(
                entity_id=entity_id,
                entity_type=entity_type,
                strategy=FallbackStrategy.T2I_IMMEDIATE,
                max_retries=0,
                predicted_quality=best_quality,
                reasoning=f"预测质量 {best_quality:.2f} < {self.threshold_immediate}，所有镜头都难以获得高质量 reference"
            )
        elif best_quality < self.threshold_normal:
            return FallbackDecision(
                entity_id=entity_id,
                entity_type=entity_type,
                strategy=FallbackStrategy.DAG_LOW_BUDGET,
                max_retries=self.low_budget_retries,
                predicted_quality=best_quality,
                reasoning=f"预测质量 {best_quality:.2f} 处于边缘区间 [{self.threshold_immediate}, {self.threshold_normal})，尝试 DAG 但降低 retry 预算"
            )
        else:
            return FallbackDecision(
                entity_id=entity_id,
                entity_type=entity_type,
                strategy=FallbackStrategy.DAG_NORMAL,
                max_retries=self.default_max_retries,
                predicted_quality=best_quality,
                reasoning=f"预测质量 {best_quality:.2f} >= {self.threshold_normal}，正常 DAG 流程"
            )

    def _llm_decision(
        self,
        entity_id: str,
        entity_type: str,
        entity_description: str,
        best_quality: float,
        shot_qualities: Dict[int, float],
    ) -> FallbackDecision:
        """使用 LLM 辅助决策"""
        try:
            user_message = f"""
## 实体信息
- ID: {entity_id}
- 类型: {entity_type}
- 描述: {entity_description}

## DAG 调度预测
- 最佳预测质量: {best_quality:.2f}
- 各 shot 质量明细:
{json.dumps(shot_qualities, indent=2)}

请分析并给出决策。
"""
            response = self.llm.chat(
                user_message=user_message,
                system=T2I_DECISION_PROMPT,
                max_tokens=500,
                temperature=0.1,
            ).strip()

            # 解析 JSON
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
                response = response.strip()

            data = json.loads(response)

            strategy_map = {
                "dag_normal": FallbackStrategy.DAG_NORMAL,
                "dag_low_budget": FallbackStrategy.DAG_LOW_BUDGET,
                "t2i_immediate": FallbackStrategy.T2I_IMMEDIATE,
            }

            strategy = strategy_map.get(data.get("strategy", "dag_normal"), FallbackStrategy.DAG_NORMAL)

            return FallbackDecision(
                entity_id=entity_id,
                entity_type=entity_type,
                strategy=strategy,
                max_retries=data.get("max_retries", self.default_max_retries),
                predicted_quality=best_quality,
                reasoning=data.get("reasoning", "LLM 决策")
            )

        except Exception as e:
            if self.verbose:
                print(f"[T2IFallbackAgent] ⚠️ LLM 决策失败: {e}，回退到规则决策")
            return self._rule_based_decision(entity_id, entity_type, best_quality)

    def _log_decision(self, decision: FallbackDecision):
        """打印决策日志"""
        if not self.verbose:
            return

        strategy_emoji = {
            FallbackStrategy.DAG_NORMAL: "✅",
            FallbackStrategy.DAG_LOW_BUDGET: "⚠️",
            FallbackStrategy.T2I_IMMEDIATE: "🔄",
        }

        emoji = strategy_emoji.get(decision.strategy, "")
        print(f"\n{'='*70}")
        print(f"[T2IFallbackAgent] 预判决策 | {decision.entity_id}")
        print(f"{'='*70}")
        print(f"  实体类型: {decision.entity_type}")
        print(f"  预测质量: {decision.predicted_quality:.2f}")
        print(f"  {emoji} 策略: {decision.strategy.value}")
        print(f"  Retry 预算: {decision.max_retries}")
        print(f"  理由: {decision.reasoning}")
        print(f"{'='*70}\n")

    # ── 阶段 2：执行中记录 ────────────────────────────────────────────────────

    def log_attempt(
        self,
        entity_id: str,
        attempt_number: int,
        action: str,
        grounding_quality: float = -1.0,
        failure_reason: Optional[str] = None,
    ):
        """
        记录一次尝试

        Args:
            entity_id: 实体 ID
            attempt_number: 当前尝试次数（从 1 开始）
            action: "dag_attempt" | "t2i_fallback" | "success"
            grounding_quality: 本次 grounding 得到的质量（-1 表示失败）
            failure_reason: 失败原因（如果失败）
        """
        if entity_id not in self.session_logs:
            return

        session = self.session_logs[entity_id]
        max_attempts = session.initial_decision.max_retries + 1  # +1 是首次尝试

        log = FallbackAttemptLog(
            entity_id=entity_id,
            attempt_number=attempt_number,
            max_attempts=max_attempts,
            action=action,
            grounding_quality=grounding_quality,
            failure_reason=failure_reason,
        )
        session.attempts.append(log)

        if self.verbose:
            self._print_attempt_log(log, session.initial_decision.strategy)

    def _print_attempt_log(self, log: FallbackAttemptLog, strategy: FallbackStrategy):
        """打印尝试日志"""
        action_emoji = {
            "dag_attempt": "🎬",
            "t2i_fallback": "🔄",
            "success": "✅",
        }
        emoji = action_emoji.get(log.action, "")

        print(f"[T2IFallbackAgent] {emoji} {log.entity_id} | "
              f"尝试 {log.attempt_number}/{log.max_attempts} | "
              f"动作: {log.action}")

        if log.grounding_quality >= 0:
            print(f"  └── Grounding 质量: {log.grounding_quality:.2f}")
        if log.failure_reason:
            print(f"  └── 失败原因: {log.failure_reason}")

    # ── 阶段 3：失败后决策 ────────────────────────────────────────────────────

    def post_failure_decision(
        self,
        entity_id: str,
        current_attempt: int,
        failure_reason: str = "",
    ) -> str:
        """
        失败后决策：继续 retry 还是 fallback T2I

        Args:
            entity_id: 实体 ID
            current_attempt: 当前尝试次数
            failure_reason: 失败原因

        Returns:
            "retry": 继续 retry
            "t2i": fallback 到 T2I
        """
        if entity_id not in self.session_logs:
            return "t2i"  # 没有 session，直接 fallback

        session = self.session_logs[entity_id]
        max_retries = session.initial_decision.max_retries

        if current_attempt > max_retries:
            if self.verbose:
                print(f"\n[T2IFallbackAgent] 🔄 {entity_id}: "
                      f"已尝试 {current_attempt} 次（预算 {max_retries}），Fallback 到 T2I")
                print(f"  └── 最后失败原因: {failure_reason}")
            return "t2i"

        if self.verbose:
            remaining = max_retries - current_attempt + 1
            print(f"[T2IFallbackAgent] 🔁 {entity_id}: 继续 retry（剩余预算: {remaining}）")

        return "retry"

    # ── 最终结果记录 ────────────────────────────────────────────────────────

    def record_final_outcome(
        self,
        entity_id: str,
        outcome: str,
        reference_path: str = "",
    ):
        """
        记录最终结果

        Args:
            entity_id: 实体 ID
            outcome: "dag_success" | "t2i_success" | "failed"
            reference_path: 最终获得的 reference 路径
        """
        if entity_id not in self.session_logs:
            return

        session = self.session_logs[entity_id]
        session.final_outcome = outcome
        session.final_reference_path = reference_path

        # 计算总耗时
        if session.attempts:
            first_ts = datetime.fromisoformat(session.attempts[0].timestamp)
            last_ts = datetime.fromisoformat(session.attempts[-1].timestamp)
            session.total_time_seconds = (last_ts - first_ts).total_seconds()

        if self.verbose:
            outcome_emoji = {
                "dag_success": "✅",
                "t2i_success": "🔄✅",
                "failed": "❌",
            }
            emoji = outcome_emoji.get(outcome, "")
            print(f"\n{'='*70}")
            print(f"[T2IFallbackAgent] {emoji} 最终结果 | {entity_id}")
            print(f"{'='*70}")
            print(f"  结果: {outcome}")
            print(f"  尝试次数: {len(session.attempts)}")
            print(f"  总耗时: {session.total_time_seconds:.1f}s")
            if reference_path:
                print(f"  Reference: {reference_path}")
            print(f"{'='*70}\n")

    # ── 日志导出 ────────────────────────────────────────────────────────────

    def export_logs(self, output_path: str):
        """导出所有日志到 JSON 文件"""
        logs = {}
        for entity_id, session in self.session_logs.items():
            logs[entity_id] = {
                "entity_id": session.entity_id,
                "entity_type": session.entity_type,
                "initial_decision": {
                    "strategy": session.initial_decision.strategy.value,
                    "max_retries": session.initial_decision.max_retries,
                    "predicted_quality": session.initial_decision.predicted_quality,
                    "reasoning": session.initial_decision.reasoning,
                    "timestamp": session.initial_decision.timestamp,
                },
                "attempts": [
                    {
                        "attempt_number": a.attempt_number,
                        "max_attempts": a.max_attempts,
                        "action": a.action,
                        "grounding_quality": a.grounding_quality,
                        "failure_reason": a.failure_reason,
                        "timestamp": a.timestamp,
                    }
                    for a in session.attempts
                ],
                "final_outcome": session.final_outcome,
                "final_reference_path": session.final_reference_path,
                "total_time_seconds": session.total_time_seconds,
            }

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)

        if self.verbose:
            print(f"[T2IFallbackAgent] 📝 日志已导出: {output_path}")

    def get_summary(self) -> Dict:
        """获取所有实体的 fallback 决策摘要"""
        summary = {
            "total_entities": len(self.session_logs),
            "dag_normal": 0,
            "dag_low_budget": 0,
            "t2i_immediate": 0,
            "final_dag_success": 0,
            "final_t2i_success": 0,
            "final_failed": 0,
        }

        for session in self.session_logs.values():
            # 统计初始策略
            if session.initial_decision.strategy == FallbackStrategy.DAG_NORMAL:
                summary["dag_normal"] += 1
            elif session.initial_decision.strategy == FallbackStrategy.DAG_LOW_BUDGET:
                summary["dag_low_budget"] += 1
            elif session.initial_decision.strategy == FallbackStrategy.T2I_IMMEDIATE:
                summary["t2i_immediate"] += 1

            # 统计最终结果
            if session.final_outcome == "dag_success":
                summary["final_dag_success"] += 1
            elif session.final_outcome == "t2i_success":
                summary["final_t2i_success"] += 1
            elif session.final_outcome == "failed":
                summary["final_failed"] += 1

        return summary

    def print_summary(self):
        """打印摘要"""
        summary = self.get_summary()
        print(f"\n{'='*70}")
        print(f"[T2IFallbackAgent] 📊 Fallback 决策摘要")
        print(f"{'='*70}")
        print(f"  总实体数: {summary['total_entities']}")
        print(f"  ─────────────────────────────────")
        print(f"  初始策略分布:")
        print(f"    ✅ DAG 正常流程: {summary['dag_normal']}")
        print(f"    ⚠️ DAG 低预算:   {summary['dag_low_budget']}")
        print(f"    🔄 直接 T2I:     {summary['t2i_immediate']}")
        print(f"  ─────────────────────────────────")
        print(f"  最终结果:")
        print(f"    ✅ DAG 成功:   {summary['final_dag_success']}")
        print(f"    🔄 T2I 成功:   {summary['final_t2i_success']}")
        print(f"    ❌ 失败:       {summary['final_failed']}")
        print(f"{'='*70}\n")


# ── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 模拟测试
    from dataclasses import dataclass

    @dataclass
    class MockScheduleResult:
        quality_matrix: dict
        reference_sources: dict

    # 创建 agent
    agent = T2IFallbackAgent(use_llm_decision=False, verbose=True)

    # 模拟三种情况
    test_cases = [
        # Case 1: 高质量预测（正常 DAG）
        ("char_hero", "character", MockScheduleResult(
            quality_matrix={1: {"char_hero": 0.35}, 2: {"char_hero": 0.55}, 3: {"char_hero": 0.85}},
            reference_sources={"char_hero": 3}
        )),
        # Case 2: 边缘质量（低预算 DAG）
        ("char_villain", "character", MockScheduleResult(
            quality_matrix={1: {"char_villain": 0.3}, 2: {"char_villain": 0.4}, 3: {"char_villain": 0.45}},
            reference_sources={"char_villain": 3}
        )),
        # Case 3: 低质量预测（直接 T2I）
        ("char_mystery", "character", MockScheduleResult(
            quality_matrix={1: {"char_mystery": 0.1}, 2: {"char_mystery": 0.15}, 3: {"char_mystery": 0.2}},
            reference_sources={"char_mystery": 3}
        )),
    ]

    for entity_id, entity_type, schedule in test_cases:
        decision = agent.pre_execution_decision(
            entity_id=entity_id,
            entity_type=entity_type,
            entity_description=f"Test {entity_id}",
            schedule_result=schedule,
        )

        # 模拟执行过程
        if decision.strategy != FallbackStrategy.T2I_IMMEDIATE:
            for attempt in range(1, decision.max_retries + 2):
                # 模拟 DAG 尝试
                agent.log_attempt(entity_id, attempt, "dag_attempt",
                                  grounding_quality=0.3, failure_reason="质量不足")

                next_action = agent.post_failure_decision(entity_id, attempt, "质量不足")
                if next_action == "t2i":
                    agent.log_attempt(entity_id, attempt + 1, "t2i_fallback")
                    agent.record_final_outcome(entity_id, "t2i_success", "/path/to/t2i_ref.png")
                    break
        else:
            agent.log_attempt(entity_id, 1, "t2i_fallback")
            agent.record_final_outcome(entity_id, "t2i_success", "/path/to/t2i_ref.png")

    # 打印摘要
    agent.print_summary()

    # 导出日志
    agent.export_logs("./test_fallback_log.json")
