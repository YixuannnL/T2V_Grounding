"""
experience/advisor.py

经验顾问：基于历史经验为当前生成任务提供建议

核心功能：
1. 在生成前提供参数建议（基于相似场景的历史经验）
2. 在遇到问题时推荐最有效的解决策略
3. 生成结束后记录经验供未来使用

设计原则：
- 不增加 LLM 调用：所有建议基于规则和历史统计
- 渐进式学习：经验越多，建议越准确
- 透明可解释：每条建议都有来源说明
"""

import uuid
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experience.database import (
    SceneFingerprint,
    GenerationExperience,
    ExperienceDatabase,
)
from retry.root_cause_analyzer import IssueCategory


@dataclass
class ExperienceAdvice:
    """经验建议"""
    # 参数建议
    suggested_ip_adapter_scale: Optional[float] = None
    suggested_steps: Optional[int] = None
    suggested_guide_scale_text: Optional[float] = None

    # 策略建议
    recommended_strategies: List[str] = field(default_factory=list)
    strategies_to_avoid: List[str] = field(default_factory=list)

    # prompt 建议
    prompt_hints: List[str] = field(default_factory=list)

    # 预期结果
    expected_attempts: float = 1.0           # 预期需要的尝试次数
    expected_success_rate: float = 0.5       # 预期成功率

    # 来源说明
    source_experiences: int = 0              # 基于多少条历史经验
    confidence: float = 0.5                  # 建议置信度 0-1
    reasoning: List[str] = field(default_factory=list)  # 建议理由

    def has_suggestions(self) -> bool:
        """是否有有效建议"""
        return (
            self.suggested_ip_adapter_scale is not None or
            self.suggested_steps is not None or
            self.recommended_strategies or
            self.prompt_hints
        )


class ExperienceAdvisor:
    """
    经验顾问

    基于历史生成经验提供智能建议，帮助提高首次生成成功率。

    Usage:
        advisor = ExperienceAdvisor(db_path="./experience.db")

        # 生成前获取建议
        fingerprint = advisor.create_fingerprint(parse_result, shot_info)
        advice = advisor.get_advice(fingerprint)

        if advice.has_suggestions():
            print(f"建议 ip_adapter_scale: {advice.suggested_ip_adapter_scale}")
            print(f"推荐策略: {advice.recommended_strategies}")

        # 生成后记录经验
        advisor.record_generation(
            fingerprint=fingerprint,
            generation_mode="phantom",
            params=final_params,
            attempts=total_attempts,
            issues=encountered_issues,
            strategies_used=strategies,
            final_score=score,
            success=passed,
        )
    """

    def __init__(
        self,
        db_path: str = "./experience.db",
        min_experiences_for_advice: int = 3,  # 至少需要这么多经验才给建议
        verbose: bool = True,
    ):
        self.db = ExperienceDatabase(db_path, verbose=verbose)
        self.min_experiences = min_experiences_for_advice
        self.verbose = verbose

        # 当前 session 的临时经验（尚未写入数据库）
        self._session_experiences: List[GenerationExperience] = []

    def create_fingerprint(
        self,
        entities: List[Dict],
        shot_text: str,
        shot_type: str = "medium",
    ) -> SceneFingerprint:
        """
        从解析结果创建场景指纹

        Args:
            entities: 实体列表 [{"entity_id": ..., "type": ...}, ...]
            shot_text: 镜头文本描述
            shot_type: 镜头类型 ("closeup" / "medium" / "wide")

        Returns:
            SceneFingerprint
        """
        # 统计实体类型
        entity_types = [e.get("type", "unknown") for e in entities]
        characters = [e for e in entities if e.get("type") == "character"]
        objects = [e for e in entities if e.get("type") == "object"]
        locations = [e for e in entities if e.get("type") == "location"]

        # 检测身体部位特写
        shot_lower = shot_text.lower()
        body_parts = ["hand", "hands", "finger", "foot", "feet", "leg", "arm", "back", "shoulder"]
        is_body_part = shot_type == "closeup" and any(bp in shot_lower for bp in body_parts)

        # 检测交互动作
        interaction_keywords = [
            "shakes hands", "hugs", "kisses", "talks to", "hands over",
            "gives", "takes", "touches", "holds", "握手", "拥抱", "交谈",
        ]
        has_interaction = any(kw in shot_lower for kw in interaction_keywords)

        return SceneFingerprint(
            shot_type=shot_type,
            entity_types=entity_types,
            entity_count=len(entities),
            has_character=len(characters) > 0,
            has_multiple_characters=len(characters) > 1,
            character_count=len(characters),
            has_object=len(objects) > 0,
            has_location=len(locations) > 0,
            is_body_part_closeup=is_body_part,
            has_interaction=has_interaction,
        )

    def get_advice(
        self,
        fingerprint: SceneFingerprint,
        current_issues: Optional[List[str]] = None,
    ) -> ExperienceAdvice:
        """
        获取生成建议

        Args:
            fingerprint: 场景指纹
            current_issues: 当前遇到的问题类别（用于获取问题特定建议）

        Returns:
            ExperienceAdvice 包含参数和策略建议
        """
        advice = ExperienceAdvice()
        reasoning = []

        # Step 1: 查找相似经验
        similar_exps = self.db.find_similar_experiences(
            fingerprint,
            top_k=10,
            min_similarity=0.5,
            success_only=False,
        )

        if not similar_exps:
            advice.reasoning = ["无历史经验，使用默认参数"]
            return advice

        advice.source_experiences = len(similar_exps)

        # Step 2: 从成功经验中提取参数建议
        successful_exps = [(e, s) for e, s in similar_exps if e.success]

        if len(successful_exps) >= self.min_experiences:
            # 计算加权平均参数（相似度越高权重越大）
            total_weight = sum(s for _, s in successful_exps)

            ip_scales = [e.ip_adapter_scale * s for e, s in successful_exps]
            advice.suggested_ip_adapter_scale = round(sum(ip_scales) / total_weight, 2)
            reasoning.append(f"基于 {len(successful_exps)} 条成功经验，"
                           f"建议 ip_adapter_scale={advice.suggested_ip_adapter_scale:.2f}")

            # 推理步数建议（取众数或加权平均）
            steps = [e.num_inference_steps for e, _ in successful_exps]
            from collections import Counter
            step_counter = Counter(steps)
            most_common_steps = step_counter.most_common(1)[0][0]
            advice.suggested_steps = most_common_steps
            reasoning.append(f"推理步数建议: {most_common_steps}")

            # 计算预期成功率和尝试次数
            total_attempts = sum(e.total_attempts for e, _ in successful_exps)
            advice.expected_attempts = total_attempts / len(successful_exps)
            advice.expected_success_rate = len(successful_exps) / len(similar_exps)

        # Step 3: 从失败经验中提取要避免的策略
        failed_exps = [(e, s) for e, s in similar_exps if not e.success]
        if failed_exps:
            # 收集失败的策略
            failed_strategies = []
            for exp, _ in failed_exps:
                failed_strategies.extend(exp.failed_strategies)

            from collections import Counter
            fail_counter = Counter(failed_strategies)
            common_failures = [s for s, c in fail_counter.most_common(3) if c >= 2]

            if common_failures:
                advice.strategies_to_avoid = common_failures
                reasoning.append(f"建议避免策略: {common_failures} (历史上多次失败)")

        # Step 4: 提取成功策略
        if successful_exps:
            successful_strategies = []
            for exp, sim in successful_exps:
                if exp.successful_strategy:
                    successful_strategies.append((exp.successful_strategy, sim))

            if successful_strategies:
                # 按相似度加权
                from collections import defaultdict
                strategy_scores = defaultdict(float)
                for strat, sim in successful_strategies:
                    strategy_scores[strat] += sim

                sorted_strategies = sorted(
                    strategy_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                advice.recommended_strategies = [s for s, _ in sorted_strategies[:3]]
                reasoning.append(f"推荐策略: {advice.recommended_strategies}")

        # Step 5: 基于场景特征的特定建议
        if fingerprint.has_multiple_characters:
            advice.prompt_hints.append(
                "多人场景建议：在 prompt 中明确人数，"
                "如 'exactly 2 people'"
            )
            if not advice.suggested_ip_adapter_scale or advice.suggested_ip_adapter_scale < 0.7:
                advice.suggested_ip_adapter_scale = 0.75
                reasoning.append("多人场景需要更高的参考图权重")

        if fingerprint.is_body_part_closeup:
            advice.prompt_hints.append(
                "身体部位特写：建议不传人脸参考图，"
                "通过 prompt 描述服装/配饰保持一致性"
            )

        if fingerprint.has_interaction:
            advice.prompt_hints.append(
                "交互动作场景：可能需要多次尝试，"
                "建议预期 2-3 次重试"
            )
            advice.expected_attempts = max(advice.expected_attempts, 2.0)

        # Step 6: 如果有当前问题，获取问题特定统计
        if current_issues:
            for issue in current_issues:
                stats = self.db.get_issue_stats(issue)
                if stats and stats["most_effective_strategy"]:
                    if stats["most_effective_strategy"] not in advice.recommended_strategies:
                        advice.recommended_strategies.insert(0, stats["most_effective_strategy"])
                        reasoning.append(
                            f"针对 {issue} 问题，历史最有效策略: "
                            f"{stats['most_effective_strategy']} "
                            f"(解决率: {stats['resolution_rate']:.1%})"
                        )

        # 计算置信度
        advice.confidence = min(1.0, len(similar_exps) / 10 * 0.5 +
                                (advice.expected_success_rate if advice.expected_success_rate else 0) * 0.5)
        advice.reasoning = reasoning

        if self.verbose and advice.has_suggestions():
            self._print_advice(advice, fingerprint)

        return advice

    def record_generation(
        self,
        fingerprint: SceneFingerprint,
        generation_mode: str,
        params: Dict[str, Any],
        attempts: int,
        issues: List[str],
        strategies_used: List[Tuple[str, bool]],  # [(strategy_name, success), ...]
        final_score: float,
        success: bool,
        lessons: Optional[List[str]] = None,
        project_hint: str = "",
    ):
        """
        记录生成经验

        Args:
            fingerprint: 场景指纹
            generation_mode: 生成模式 ("t2v" / "phantom")
            params: 最终使用的参数
            attempts: 总尝试次数
            issues: 遇到的问题类别列表
            strategies_used: 使用的策略及其成功/失败状态
            final_score: 最终 critique 分数
            success: 是否成功
            lessons: 学到的经验（可选，如果有的话）
            project_hint: 项目类型提示
        """
        # 提取成功和失败的策略
        successful_strategy = None
        failed_strategies = []
        for strat, succeeded in strategies_used:
            if succeeded:
                successful_strategy = strat
            else:
                failed_strategies.append(strat)

        # 自动生成 lessons（如果没有提供）
        if lessons is None:
            lessons = self._generate_lessons(
                fingerprint, issues, strategies_used, success, attempts
            )

        exp = GenerationExperience(
            experience_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            fingerprint=fingerprint,
            generation_mode=generation_mode,
            ip_adapter_scale=params.get("ip_adapter_scale", 0.6),
            num_inference_steps=params.get("num_inference_steps", 50),
            guide_scale_text=params.get("guide_scale_text", 7.5),
            total_attempts=attempts,
            final_attempt_params=params,
            encountered_issues=issues,
            successful_strategy=successful_strategy,
            failed_strategies=failed_strategies,
            final_score=final_score,
            success=success,
            lessons_learned=lessons,
            project_hint=project_hint,
        )

        # 写入数据库
        self.db.record_experience(exp)

        # 也保存到 session 缓存
        self._session_experiences.append(exp)

    def _generate_lessons(
        self,
        fingerprint: SceneFingerprint,
        issues: List[str],
        strategies_used: List[Tuple[str, bool]],
        success: bool,
        attempts: int,
    ) -> List[str]:
        """自动生成经验教训"""
        lessons = []

        # 基于场景特征
        if fingerprint.has_multiple_characters:
            if success and attempts <= 2:
                lessons.append("多人场景一次/两次通过，参数配置有效")
            elif success and attempts > 2:
                lessons.append(f"多人场景需要 {attempts} 次尝试，考虑提高初始 ip_adapter_scale")

        if fingerprint.is_body_part_closeup and success:
            lessons.append("身体部位特写成功，验证跳过人脸参考的策略有效")

        # 基于问题和策略
        for strat, succeeded in strategies_used:
            if succeeded:
                if issues:
                    lessons.append(f"'{strat}' 策略有效解决 {issues} 问题")
            else:
                lessons.append(f"'{strat}' 策略对此场景无效")

        # 基于结果
        if not success:
            lessons.append(f"最终未通过 critique，需要进一步优化策略")

        return lessons

    def get_session_summary(self) -> Dict:
        """获取当前 session 的经验总结"""
        if not self._session_experiences:
            return {"total": 0, "success_rate": 0}

        total = len(self._session_experiences)
        successes = sum(1 for e in self._session_experiences if e.success)
        avg_attempts = sum(e.total_attempts for e in self._session_experiences) / total

        # 统计问题
        issue_counts = {}
        for exp in self._session_experiences:
            for issue in exp.encountered_issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1

        return {
            "total": total,
            "success_rate": successes / total,
            "avg_attempts": avg_attempts,
            "common_issues": issue_counts,
        }

    def _print_advice(self, advice: ExperienceAdvice, fingerprint: SceneFingerprint):
        """打印建议"""
        print(f"\n[Advisor] ── 经验建议 ──")
        print(f"[Advisor] 场景: {fingerprint.shot_type} | "
              f"角色: {fingerprint.character_count} | "
              f"交互: {'是' if fingerprint.has_interaction else '否'}")
        print(f"[Advisor] 基于 {advice.source_experiences} 条历史经验 "
              f"(置信度: {advice.confidence:.2f})")

        if advice.suggested_ip_adapter_scale:
            print(f"[Advisor] 💡 建议 ip_adapter_scale: {advice.suggested_ip_adapter_scale:.2f}")

        if advice.suggested_steps:
            print(f"[Advisor] 💡 建议推理步数: {advice.suggested_steps}")

        if advice.recommended_strategies:
            print(f"[Advisor] ✅ 推荐策略: {advice.recommended_strategies}")

        if advice.strategies_to_avoid:
            print(f"[Advisor] ⛔ 建议避免: {advice.strategies_to_avoid}")

        if advice.prompt_hints:
            for hint in advice.prompt_hints:
                print(f"[Advisor] 📝 {hint}")

        print(f"[Advisor] 预期: {advice.expected_attempts:.1f} 次尝试, "
              f"{advice.expected_success_rate:.1%} 成功率")


# ── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    # 使用临时数据库测试
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        test_db_path = f.name

    advisor = ExperienceAdvisor(db_path=test_db_path, verbose=True)

    # 创建测试指纹
    test_entities = [
        {"entity_id": "char_a", "type": "character"},
        {"entity_id": "char_b", "type": "character"},
    ]
    fp = advisor.create_fingerprint(
        entities=test_entities,
        shot_text="Alex and Bob shake hands in the office",
        shot_type="medium",
    )

    print(f"\n创建的指纹:")
    print(f"  shot_type: {fp.shot_type}")
    print(f"  character_count: {fp.character_count}")
    print(f"  has_interaction: {fp.has_interaction}")
    print(f"  hash: {fp.to_hash()}")

    # 获取建议（此时应该没有历史经验）
    print("\n首次获取建议（无历史经验）:")
    advice = advisor.get_advice(fp)

    # 模拟记录一些经验
    print("\n模拟记录经验...")
    for i in range(5):
        advisor.record_generation(
            fingerprint=fp,
            generation_mode="phantom",
            params={"ip_adapter_scale": 0.7 + i * 0.05, "num_inference_steps": 50},
            attempts=2 if i < 3 else 1,
            issues=["identity"] if i < 2 else [],
            strategies_used=[("increase_ip_scale", i >= 2)],
            final_score=0.6 + i * 0.08,
            success=i >= 2,
        )

    # 再次获取建议
    print("\n再次获取建议（有历史经验）:")
    advice2 = advisor.get_advice(fp)

    # 打印 session 总结
    print("\n当前 session 总结:")
    summary = advisor.get_session_summary()
    print(f"  总数: {summary['total']}")
    print(f"  成功率: {summary['success_rate']:.1%}")
    print(f"  平均尝试: {summary['avg_attempts']:.1f}")
    print(f"  常见问题: {summary['common_issues']}")

    # 清理
    os.unlink(test_db_path)
