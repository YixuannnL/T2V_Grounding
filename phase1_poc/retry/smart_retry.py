"""
retry/smart_retry.py

智能重试执行器：基于根因分析执行针对性的重试策略

核心思想：
- 不同问题类别需要不同的修复策略
- 避免盲目换 seed（可能多次遇到同类问题）
- 每类问题有专属的参数调整方案

策略体系：
1. ENTITY_COUNT → 强化 prompt 中的数量描述 + 换 seed
2. IDENTITY → 增大 ip_adapter_scale + 可能换参考图
3. STYLE → 增大 ip_adapter_scale + 在 prompt 中强调风格
4. QUALITY → 增加推理步数（有上限）+ 换 seed
5. POSE_MOTION → 换 seed（姿态问题通常随机性导致）
6. SCENE → 添加光线描述到 prompt / 考虑加 location 参考
"""

import random
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retry.root_cause_analyzer import (
    IssueCategory,
    RootCause,
    DiagnosisResult,
    RootCauseAnalyzer,
)
from verification.video_critic import CritiqueResult


class RetryAction(Enum):
    """重试动作类型"""
    CHANGE_SEED = "change_seed"                      # 换随机种子
    INCREASE_IP_SCALE = "increase_ip_scale"          # 增大参考图权重
    DECREASE_IP_SCALE = "decrease_ip_scale"          # 减小参考图权重
    INCREASE_STEPS = "increase_steps"                # 增加推理步数
    ENHANCE_PROMPT_COUNT = "enhance_prompt_count"    # 强化 prompt 中的数量描述
    ENHANCE_PROMPT_STYLE = "enhance_prompt_style"    # 强化 prompt 中的风格描述
    ENHANCE_PROMPT_LIGHTING = "enhance_prompt_lighting"  # 添加光线描述
    CHANGE_REFERENCE = "change_reference"            # 更换参考图
    FALLBACK_T2V = "fallback_t2v"                    # 回退到 T2V 模式


@dataclass
class RetryStrategy:
    """单次重试策略"""
    actions: List[RetryAction]           # 执行的动作列表
    target_category: IssueCategory       # 针对的问题类别
    param_changes: Dict[str, Any]        # 参数变更
    prompt_additions: List[str]          # prompt 追加内容
    reasoning: str                       # 策略理由
    estimated_success_rate: float        # 预估成功率 0-1


@dataclass
class RetryExecutionResult:
    """重试执行结果"""
    attempt: int                         # 第几次重试
    strategy: RetryStrategy              # 使用的策略
    new_params: Dict[str, Any]           # 新参数
    new_prompt: str                      # 新 prompt
    success: bool                        # 是否成功
    critique_score: float                # 新的 critique 分数
    improvement: float                   # 相比上次的分数提升


# ── 策略模板 ─────────────────────────────────────────────────────────────────

# 每类问题的默认修复策略（按优先级排序）
DEFAULT_STRATEGIES: Dict[IssueCategory, List[Dict]] = {
    IssueCategory.ENTITY_COUNT: [
        {
            "actions": [RetryAction.ENHANCE_PROMPT_COUNT, RetryAction.CHANGE_SEED],
            "param_changes": {},
            "prompt_template": "Important: This scene contains exactly {count} {entity_type}(s). "
                              "Ensure the correct number of entities is generated.",
            "reasoning": "数量问题通常需要在 prompt 中明确强调 + 换 seed 避免同样错误",
            "success_rate": 0.6,
        },
        {
            "actions": [RetryAction.DECREASE_IP_SCALE, RetryAction.CHANGE_SEED],
            "param_changes": {"ip_adapter_scale_delta": -0.1},
            "reasoning": "降低参考图权重可能让模型更关注 prompt 中的数量描述",
            "success_rate": 0.4,
        },
    ],
    IssueCategory.IDENTITY: [
        {
            "actions": [RetryAction.INCREASE_IP_SCALE],
            "param_changes": {"ip_adapter_scale_delta": 0.15},
            "reasoning": "增大参考图权重以强化身份保持",
            "success_rate": 0.65,
        },
        {
            "actions": [RetryAction.INCREASE_IP_SCALE, RetryAction.CHANGE_SEED],
            "param_changes": {"ip_adapter_scale_delta": 0.2},
            "reasoning": "进一步增大参考图权重 + 换 seed",
            "success_rate": 0.5,
        },
        {
            "actions": [RetryAction.CHANGE_REFERENCE],
            "param_changes": {},
            "reasoning": "当前参考图可能不适合，尝试使用其他可用参考",
            "success_rate": 0.4,
        },
    ],
    IssueCategory.STYLE: [
        {
            "actions": [RetryAction.INCREASE_IP_SCALE, RetryAction.ENHANCE_PROMPT_STYLE],
            "param_changes": {"ip_adapter_scale_delta": 0.15},
            "prompt_template": "Photorealistic style. Real human appearance. "
                              "NOT anime, NOT cartoon, NOT CGI. Natural skin texture.",
            "reasoning": "增大参考图权重 + 在 prompt 中强调真实感风格",
            "success_rate": 0.55,
        },
        {
            "actions": [RetryAction.FALLBACK_T2V],
            "param_changes": {"force_t2v": True},
            "reasoning": "严重风格漂移时，回退 T2V 可能效果更好",
            "success_rate": 0.3,
        },
    ],
    IssueCategory.QUALITY: [
        {
            "actions": [RetryAction.INCREASE_STEPS, RetryAction.CHANGE_SEED],
            "param_changes": {"steps_delta": 10},
            "reasoning": "增加推理步数以提升画质 + 换 seed",
            "success_rate": 0.5,
        },
        {
            "actions": [RetryAction.CHANGE_SEED],
            "param_changes": {},
            "reasoning": "画质问题可能是 seed 导致，简单换 seed 重试",
            "success_rate": 0.4,
        },
    ],
    IssueCategory.POSE_MOTION: [
        {
            "actions": [RetryAction.CHANGE_SEED],
            "param_changes": {},
            "reasoning": "姿态问题通常由随机性导致，换 seed 是最有效方案",
            "success_rate": 0.6,
        },
        {
            "actions": [RetryAction.DECREASE_IP_SCALE, RetryAction.CHANGE_SEED],
            "param_changes": {"ip_adapter_scale_delta": -0.1},
            "reasoning": "降低参考图权重给模型更多自由度处理姿态",
            "success_rate": 0.45,
        },
    ],
    IssueCategory.SCENE: [
        {
            "actions": [RetryAction.ENHANCE_PROMPT_LIGHTING, RetryAction.CHANGE_SEED],
            "param_changes": {},
            "prompt_template": "Consistent lighting from the reference. "
                              "Natural daylight / warm indoor lighting. Soft shadows.",
            "reasoning": "在 prompt 中添加光线描述以引导生成",
            "success_rate": 0.5,
        },
        {
            "actions": [RetryAction.CHANGE_SEED],
            "param_changes": {},
            "reasoning": "场景问题可能是随机变化，换 seed 重试",
            "success_rate": 0.4,
        },
    ],
    IssueCategory.UNKNOWN: [
        {
            "actions": [RetryAction.CHANGE_SEED],
            "param_changes": {},
            "reasoning": "未知问题，尝试换 seed",
            "success_rate": 0.3,
        },
    ],
}

# 参数约束
PARAM_CONSTRAINTS = {
    "ip_adapter_scale": {"min": 0.3, "max": 1.0, "default": 0.6},
    "num_inference_steps": {"min": 30, "max": 50, "default": 50},  # 上限 50（UniPC solver）
    "guide_scale_text": {"min": 3.0, "max": 15.0, "default": 7.5},
}


class SmartRetryExecutor:
    """
    智能重试执行器

    根据根因分析结果选择最合适的重试策略，
    并管理多次重试的参数演进。

    Usage:
        executor = SmartRetryExecutor(max_retries=3)

        # 初始生成失败后
        diagnosis = analyzer.diagnose(critique_result)
        strategy = executor.select_strategy(diagnosis, attempt=1)

        # 应用策略
        new_params, new_prompt = executor.apply_strategy(
            strategy=strategy,
            current_params=current_params,
            current_prompt=current_prompt,
        )
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_seed: int = 42,
        verbose: bool = True,
    ):
        self.max_retries = max_retries
        self.base_seed = base_seed
        self.verbose = verbose

        # 策略历史：记录每次重试用过的策略，避免重复
        self.strategy_history: List[RetryStrategy] = []

        # 分数历史：追踪分数变化
        self.score_history: List[float] = []

    def reset(self):
        """重置状态（新 shot 开始时调用）"""
        self.strategy_history = []
        self.score_history = []

    def select_strategy(
        self,
        diagnosis: DiagnosisResult,
        attempt: int,
        current_score: float = 0.0,
    ) -> Optional[RetryStrategy]:
        """
        选择重试策略

        Args:
            diagnosis: 根因诊断结果
            attempt: 当前是第几次重试（1-based）
            current_score: 当前 critique 分数

        Returns:
            RetryStrategy 或 None（如果没有可用策略）
        """
        if attempt > self.max_retries:
            return None

        if not diagnosis.primary_cause:
            # 无明确根因，使用通用策略
            return self._create_generic_strategy(attempt)

        primary_category = diagnosis.primary_cause.category

        # 获取该类别的策略模板
        category_strategies = DEFAULT_STRATEGIES.get(
            primary_category,
            DEFAULT_STRATEGIES[IssueCategory.UNKNOWN]
        )

        # 选择尚未使用过的策略
        for i, template in enumerate(category_strategies):
            strategy = self._create_strategy_from_template(
                template=template,
                category=primary_category,
                diagnosis=diagnosis,
                attempt=attempt,
            )

            # 检查是否已经使用过类似策略
            if not self._is_strategy_used(strategy):
                if self.verbose:
                    print(f"[SmartRetry] 选择策略: {strategy.target_category.value} / "
                          f"actions={[a.value for a in strategy.actions]}")
                    print(f"[SmartRetry] 理由: {strategy.reasoning}")
                return strategy

        # 所有策略都用过了，返回最后一个（通常是最激进的）
        if category_strategies:
            return self._create_strategy_from_template(
                template=category_strategies[-1],
                category=primary_category,
                diagnosis=diagnosis,
                attempt=attempt,
            )

        return self._create_generic_strategy(attempt)

    def apply_strategy(
        self,
        strategy: RetryStrategy,
        current_params: Dict[str, Any],
        current_prompt: str,
        shot_id: int = 0,
        entity_counts: Optional[Dict[str, int]] = None,
    ) -> tuple:
        """
        应用重试策略

        Args:
            strategy: 选择的重试策略
            current_params: 当前生成参数
            current_prompt: 当前 prompt
            shot_id: 当前 shot ID（用于 seed 计算）
            entity_counts: 实体数量 {entity_type: count}（用于数量增强）

        Returns:
            (new_params, new_prompt)
        """
        new_params = current_params.copy()
        prompt_parts = [current_prompt]

        # 记录策略
        self.strategy_history.append(strategy)

        # 应用每个动作
        for action in strategy.actions:
            if action == RetryAction.CHANGE_SEED:
                new_seed = self._generate_new_seed(shot_id, len(self.strategy_history))
                new_params["seed"] = new_seed
                if self.verbose:
                    print(f"[SmartRetry] 换 seed: {new_seed}")

            elif action == RetryAction.INCREASE_IP_SCALE:
                delta = strategy.param_changes.get("ip_adapter_scale_delta", 0.15)
                old_val = new_params.get("ip_adapter_scale", PARAM_CONSTRAINTS["ip_adapter_scale"]["default"])
                new_val = min(old_val + delta, PARAM_CONSTRAINTS["ip_adapter_scale"]["max"])
                new_params["ip_adapter_scale"] = new_val
                if self.verbose:
                    print(f"[SmartRetry] ip_adapter_scale: {old_val:.2f} → {new_val:.2f}")

            elif action == RetryAction.DECREASE_IP_SCALE:
                delta = abs(strategy.param_changes.get("ip_adapter_scale_delta", -0.1))
                old_val = new_params.get("ip_adapter_scale", PARAM_CONSTRAINTS["ip_adapter_scale"]["default"])
                new_val = max(old_val - delta, PARAM_CONSTRAINTS["ip_adapter_scale"]["min"])
                new_params["ip_adapter_scale"] = new_val
                if self.verbose:
                    print(f"[SmartRetry] ip_adapter_scale: {old_val:.2f} → {new_val:.2f}")

            elif action == RetryAction.INCREASE_STEPS:
                delta = strategy.param_changes.get("steps_delta", 10)
                old_val = new_params.get("num_inference_steps", PARAM_CONSTRAINTS["num_inference_steps"]["default"])
                new_val = min(old_val + delta, PARAM_CONSTRAINTS["num_inference_steps"]["max"])
                if new_val != old_val:
                    new_params["num_inference_steps"] = new_val
                    if self.verbose:
                        print(f"[SmartRetry] steps: {old_val} → {new_val}")

            elif action == RetryAction.ENHANCE_PROMPT_COUNT:
                if entity_counts:
                    for entity_type, count in entity_counts.items():
                        count_prompt = f"[Important] This scene shows exactly {count} {entity_type}(s). "
                        prompt_parts.insert(0, count_prompt)
                        if self.verbose:
                            print(f"[SmartRetry] 添加数量强调: {count} {entity_type}")

            elif action == RetryAction.ENHANCE_PROMPT_STYLE:
                style_prompt = strategy.param_changes.get(
                    "prompt_template",
                    "Photorealistic style. Real human appearance. Natural skin texture."
                )
                prompt_parts.insert(0, f"[Style] {style_prompt}")
                if self.verbose:
                    print(f"[SmartRetry] 添加风格强调")

            elif action == RetryAction.ENHANCE_PROMPT_LIGHTING:
                lighting_prompt = strategy.param_changes.get(
                    "prompt_template",
                    "Consistent lighting. Natural illumination. Soft shadows."
                )
                prompt_parts.insert(0, f"[Lighting] {lighting_prompt}")
                if self.verbose:
                    print(f"[SmartRetry] 添加光线描述")

            elif action == RetryAction.FALLBACK_T2V:
                new_params["force_t2v"] = True
                if self.verbose:
                    print(f"[SmartRetry] 标记回退到 T2V 模式")

            elif action == RetryAction.CHANGE_REFERENCE:
                new_params["change_reference"] = True
                if self.verbose:
                    print(f"[SmartRetry] 标记需要更换参考图")

        # 添加策略中的 prompt 追加内容
        for addition in strategy.prompt_additions:
            prompt_parts.insert(0, addition)

        new_prompt = "\n\n".join(prompt_parts)

        return new_params, new_prompt

    def record_result(self, score: float, success: bool):
        """记录重试结果"""
        self.score_history.append(score)
        if self.verbose:
            if len(self.score_history) > 1:
                improvement = score - self.score_history[-2]
                trend = "📈" if improvement > 0 else ("📉" if improvement < 0 else "➡️")
                print(f"[SmartRetry] 分数: {score:.2f} ({trend} {improvement:+.2f})")
            else:
                print(f"[SmartRetry] 初始分数: {score:.2f}")

    def get_best_score(self) -> float:
        """获取历史最高分"""
        return max(self.score_history) if self.score_history else 0.0

    def _create_strategy_from_template(
        self,
        template: Dict,
        category: IssueCategory,
        diagnosis: DiagnosisResult,
        attempt: int,
    ) -> RetryStrategy:
        """从模板创建策略"""
        prompt_additions = []
        if "prompt_template" in template:
            prompt_additions.append(template["prompt_template"])

        return RetryStrategy(
            actions=template["actions"],
            target_category=category,
            param_changes=template.get("param_changes", {}),
            prompt_additions=prompt_additions,
            reasoning=template.get("reasoning", ""),
            estimated_success_rate=template.get("success_rate", 0.5),
        )

    def _create_generic_strategy(self, attempt: int) -> RetryStrategy:
        """创建通用策略（无明确根因时使用）"""
        return RetryStrategy(
            actions=[RetryAction.CHANGE_SEED],
            target_category=IssueCategory.UNKNOWN,
            param_changes={},
            prompt_additions=[],
            reasoning="无明确根因，尝试换 seed",
            estimated_success_rate=0.3,
        )

    def _is_strategy_used(self, strategy: RetryStrategy) -> bool:
        """检查策略是否已使用过"""
        for used in self.strategy_history:
            if (used.target_category == strategy.target_category and
                set(used.actions) == set(strategy.actions)):
                return True
        return False

    def _generate_new_seed(self, shot_id: int, attempt: int) -> int:
        """生成新的随机 seed"""
        # 确保不同 attempt 使用不同 seed
        random.seed(self.base_seed + shot_id * 1000 + attempt * 100)
        return random.randint(0, 2**31 - 1)


# ── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from retry.root_cause_analyzer import IssueCategory, RootCause, DiagnosisResult

    # 模拟一个诊断结果
    test_diagnosis = DiagnosisResult(
        primary_cause=RootCause(
            category=IssueCategory.IDENTITY,
            confidence=0.9,
            description="角色身份不一致",
            evidence=["胡子形状变了"],
            affected_entities=["char_bearded_man"],
            severity_score=0.8,
        ),
        secondary_causes=[
            RootCause(
                category=IssueCategory.STYLE,
                confidence=0.7,
                description="风格略有漂移",
                evidence=["轻微动画化"],
                affected_entities=[],
                severity_score=0.4,
            ),
        ],
        overall_severity=0.7,
        raw_issues=[],
        diagnosis_summary="主要问题是身份不一致",
    )

    executor = SmartRetryExecutor(max_retries=3, verbose=True)

    # 第一次重试
    print("\n=== 第 1 次重试 ===")
    strategy = executor.select_strategy(test_diagnosis, attempt=1, current_score=0.4)

    current_params = {
        "ip_adapter_scale": 0.6,
        "num_inference_steps": 50,
        "seed": 12345,
    }
    current_prompt = "A bearded man walks into the room"

    new_params, new_prompt = executor.apply_strategy(
        strategy=strategy,
        current_params=current_params,
        current_prompt=current_prompt,
        shot_id=1,
    )

    print(f"\n新参数: {new_params}")
    print(f"新 prompt 前缀: {new_prompt[:100]}...")

    # 记录结果
    executor.record_result(score=0.55, success=False)

    # 第二次重试
    print("\n=== 第 2 次重试 ===")
    strategy2 = executor.select_strategy(test_diagnosis, attempt=2, current_score=0.55)

    new_params2, new_prompt2 = executor.apply_strategy(
        strategy=strategy2,
        current_params=new_params,
        current_prompt=new_prompt,
        shot_id=1,
    )

    print(f"\n新参数: {new_params2}")
