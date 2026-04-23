"""
retry/root_cause_analyzer.py

根因分析器：从 VLM Critique 结果诊断问题的根本原因

核心思想：
- 不只是知道"分数低"，而是要知道"为什么分数低"
- 将问题归类到 6 大类别，每类对应不同的修复策略
- 基于规则匹配，无需额外 LLM 调用（快速）

分类体系：
1. ENTITY_COUNT - 实体数量错误（人多/少了）
2. IDENTITY - 身份特征不一致（脸不像、体型变了）
3. STYLE - 风格漂移（动画化、写实度变化）
4. QUALITY - 画质问题（模糊、伪影）
5. POSE_MOTION - 姿态/动作问题（不自然、穿模）
6. SCENE - 场景/光线问题（背景不对、光线方向错）
"""

import re
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verification.video_critic import CritiqueResult, CritiqueIssue, IssueType, IssueSeverity


class IssueCategory(Enum):
    """问题大类"""
    ENTITY_COUNT = "entity_count"    # 实体数量错误
    IDENTITY = "identity"            # 身份/外观不一致
    STYLE = "style"                  # 风格漂移
    QUALITY = "quality"              # 画质问题
    POSE_MOTION = "pose_motion"      # 姿态/动作问题
    SCENE = "scene"                  # 场景/光线问题
    UNKNOWN = "unknown"              # 无法归类


@dataclass
class RootCause:
    """单个根因"""
    category: IssueCategory
    confidence: float               # 0-1，诊断置信度
    description: str                # 根因描述
    evidence: List[str]             # 支持该诊断的证据（来自 CritiqueIssue）
    affected_entities: List[str]    # 受影响的实体 ID
    severity_score: float           # 严重程度 0-1


@dataclass
class DiagnosisResult:
    """诊断结果"""
    primary_cause: Optional[RootCause]   # 主要根因（最应该优先修复的）
    secondary_causes: List[RootCause]    # 次要根因
    overall_severity: float              # 整体严重程度 0-1
    raw_issues: List[CritiqueIssue]      # 原始问题列表
    diagnosis_summary: str               # 诊断总结

    @property
    def all_causes(self) -> List[RootCause]:
        """获取所有根因（按严重程度排序）"""
        causes = []
        if self.primary_cause:
            causes.append(self.primary_cause)
        causes.extend(self.secondary_causes)
        return sorted(causes, key=lambda c: c.severity_score, reverse=True)

    @property
    def categories(self) -> List[IssueCategory]:
        """获取所有涉及的问题类别"""
        return list(set(c.category for c in self.all_causes))


# ── 问题模式定义 ─────────────────────────────────────────────────────────────

# IssueType -> IssueCategory 映射
ISSUE_TYPE_TO_CATEGORY: Dict[IssueType, IssueCategory] = {
    # 数量问题
    IssueType.ENTITY_MISSING: IssueCategory.ENTITY_COUNT,
    IssueType.ENTITY_EXTRA: IssueCategory.ENTITY_COUNT,
    IssueType.ENTITY_COUNT_MISMATCH: IssueCategory.ENTITY_COUNT,

    # 身份问题
    IssueType.IDENTITY_MISMATCH: IssueCategory.IDENTITY,
    IssueType.FACIAL_FEATURE_MISMATCH: IssueCategory.IDENTITY,
    IssueType.CLOTHING_MISMATCH: IssueCategory.IDENTITY,

    # 风格问题
    IssueType.STYLE_DRIFT: IssueCategory.STYLE,

    # 画质问题
    IssueType.QUALITY_DEGRADATION: IssueCategory.QUALITY,

    # 姿态问题
    IssueType.POSE_UNNATURAL: IssueCategory.POSE_MOTION,
    IssueType.MOTION_ARTIFACT: IssueCategory.POSE_MOTION,

    # 场景问题
    IssueType.LIGHTING_INCONSISTENT: IssueCategory.SCENE,
}

# 关键词模式匹配（用于 description 分析）
KEYWORD_PATTERNS: Dict[IssueCategory, List[str]] = {
    IssueCategory.ENTITY_COUNT: [
        r"人数", r"数量", r"多了", r"少了", r"缺少", r"多余",
        r"missing", r"extra", r"count", r"number of",
        r"\d+\s*个", r"\d+\s*person", r"\d+\s*people",
    ],
    IssueCategory.IDENTITY: [
        r"脸", r"面部", r"五官", r"长相", r"外貌", r"身份",
        r"face", r"facial", r"identity", r"appearance",
        r"胡子", r"胡须", r"发型", r"发色", r"眼睛", r"鼻子",
        r"beard", r"hair", r"eyes", r"nose", r"skin",
        r"衣服", r"服装", r"穿着", r"clothing", r"outfit",
    ],
    IssueCategory.STYLE: [
        r"动画", r"卡通", r"风格", r"写实", r"真实感",
        r"anime", r"cartoon", r"style", r"realistic", r"photorealistic",
        r"画风", r"渲染", r"CG", r"3D", r"2D",
    ],
    IssueCategory.QUALITY: [
        r"模糊", r"清晰度", r"噪点", r"伪影", r"画质",
        r"blur", r"sharp", r"noise", r"artifact", r"quality",
        r"分辨率", r"resolution", r"degradation",
    ],
    IssueCategory.POSE_MOTION: [
        r"姿态", r"姿势", r"动作", r"运动", r"穿模",
        r"pose", r"posture", r"motion", r"movement",
        r"关节", r"四肢", r"手", r"脚", r"变形",
        r"joint", r"limb", r"hand", r"foot", r"deform",
        r"不自然", r"unnatural", r"awkward",
    ],
    IssueCategory.SCENE: [
        r"背景", r"场景", r"环境", r"光线", r"光照",
        r"background", r"scene", r"environment", r"lighting", r"light",
        r"阴影", r"色温", r"色调", r"明暗",
        r"shadow", r"temperature", r"tone", r"brightness",
    ],
}

# 编译正则表达式
COMPILED_PATTERNS: Dict[IssueCategory, List[re.Pattern]] = {
    cat: [re.compile(p, re.IGNORECASE) for p in patterns]
    for cat, patterns in KEYWORD_PATTERNS.items()
}

# 严重程度权重
SEVERITY_WEIGHTS: Dict[IssueSeverity, float] = {
    IssueSeverity.CRITICAL: 1.0,
    IssueSeverity.HIGH: 0.7,
    IssueSeverity.MEDIUM: 0.4,
    IssueSeverity.LOW: 0.2,
}

# 问题类别优先级（数值越小优先级越高）
CATEGORY_PRIORITY: Dict[IssueCategory, int] = {
    IssueCategory.ENTITY_COUNT: 1,   # 最高优先级：人数错了必须修
    IssueCategory.IDENTITY: 2,       # 次高：身份不对很明显
    IssueCategory.STYLE: 3,          # 风格漂移影响大
    IssueCategory.SCENE: 4,          # 场景/光线
    IssueCategory.POSE_MOTION: 5,    # 姿态问题
    IssueCategory.QUALITY: 6,        # 画质问题（通常不是主因）
    IssueCategory.UNKNOWN: 99,
}


class RootCauseAnalyzer:
    """
    根因分析器

    将 VLM Critique 的问题列表归类为根本原因，
    确定最应该优先修复的问题类别。

    Usage:
        analyzer = RootCauseAnalyzer()
        diagnosis = analyzer.diagnose(critique_result)

        if diagnosis.primary_cause:
            print(f"主要问题: {diagnosis.primary_cause.category.value}")
            print(f"建议: {diagnosis.primary_cause.description}")
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def diagnose(self, critique_result: CritiqueResult) -> DiagnosisResult:
        """
        诊断问题根因

        Args:
            critique_result: VLM Critique 结果

        Returns:
            DiagnosisResult 包含根因分析
        """
        if not critique_result.issues:
            return DiagnosisResult(
                primary_cause=None,
                secondary_causes=[],
                overall_severity=0.0,
                raw_issues=[],
                diagnosis_summary="无问题，视频质量良好",
            )

        # Step 1: 将每个 issue 归类
        categorized_issues: Dict[IssueCategory, List[Tuple[CritiqueIssue, float]]] = {
            cat: [] for cat in IssueCategory
        }

        for issue in critique_result.issues:
            category = self._categorize_issue(issue)
            severity = SEVERITY_WEIGHTS.get(issue.severity, 0.5) * issue.confidence
            categorized_issues[category].append((issue, severity))

        # Step 2: 为每个类别计算总严重度和构建 RootCause
        root_causes: List[RootCause] = []

        for category, issues_with_severity in categorized_issues.items():
            if not issues_with_severity:
                continue

            # 计算该类别的总严重度
            total_severity = sum(s for _, s in issues_with_severity)
            avg_severity = total_severity / len(issues_with_severity)

            # 收集证据和受影响实体
            evidence = [issue.description for issue, _ in issues_with_severity]
            affected_entities = list(set(
                issue.affected_entity
                for issue, _ in issues_with_severity
                if issue.affected_entity
            ))

            # 生成根因描述
            description = self._generate_cause_description(category, issues_with_severity)

            root_causes.append(RootCause(
                category=category,
                confidence=min(1.0, len(issues_with_severity) * 0.3 + avg_severity),
                description=description,
                evidence=evidence,
                affected_entities=affected_entities,
                severity_score=total_severity,
            ))

        # Step 3: 按优先级和严重度排序，确定主因
        root_causes.sort(key=lambda c: (
            CATEGORY_PRIORITY.get(c.category, 99),
            -c.severity_score,
        ))

        primary_cause = root_causes[0] if root_causes else None
        secondary_causes = root_causes[1:] if len(root_causes) > 1 else []

        # Step 4: 计算整体严重度
        overall_severity = sum(c.severity_score for c in root_causes) / max(len(root_causes), 1)
        overall_severity = min(1.0, overall_severity)

        # Step 5: 生成诊断总结
        summary = self._generate_summary(primary_cause, secondary_causes, critique_result)

        result = DiagnosisResult(
            primary_cause=primary_cause,
            secondary_causes=secondary_causes,
            overall_severity=overall_severity,
            raw_issues=critique_result.issues,
            diagnosis_summary=summary,
        )

        if self.verbose:
            self._print_diagnosis(result)

        return result

    def _categorize_issue(self, issue: CritiqueIssue) -> IssueCategory:
        """将单个问题归类"""
        # 优先使用 IssueType 映射
        if issue.issue_type in ISSUE_TYPE_TO_CATEGORY:
            return ISSUE_TYPE_TO_CATEGORY[issue.issue_type]

        # 回退到关键词匹配
        description = issue.description.lower()

        match_scores: Dict[IssueCategory, int] = {cat: 0 for cat in IssueCategory}

        for category, patterns in COMPILED_PATTERNS.items():
            for pattern in patterns:
                if pattern.search(description):
                    match_scores[category] += 1

        # 返回匹配分数最高的类别
        best_category = max(match_scores.items(), key=lambda x: x[1])
        if best_category[1] > 0:
            return best_category[0]

        return IssueCategory.UNKNOWN

    def _generate_cause_description(
        self,
        category: IssueCategory,
        issues_with_severity: List[Tuple[CritiqueIssue, float]],
    ) -> str:
        """生成根因描述"""
        num_issues = len(issues_with_severity)

        descriptions = {
            IssueCategory.ENTITY_COUNT: f"实体数量不匹配 ({num_issues} 个相关问题)",
            IssueCategory.IDENTITY: f"角色身份/外观与参考图不一致 ({num_issues} 个相关问题)",
            IssueCategory.STYLE: f"视频风格发生漂移 ({num_issues} 个相关问题)",
            IssueCategory.QUALITY: f"视频画质存在问题 ({num_issues} 个相关问题)",
            IssueCategory.POSE_MOTION: f"角色姿态或动作不自然 ({num_issues} 个相关问题)",
            IssueCategory.SCENE: f"场景或光线与预期不符 ({num_issues} 个相关问题)",
            IssueCategory.UNKNOWN: f"其他未分类问题 ({num_issues} 个相关问题)",
        }

        base_desc = descriptions.get(category, f"未知问题 ({num_issues} 个)")

        # 添加具体问题示例
        if issues_with_severity:
            top_issue = max(issues_with_severity, key=lambda x: x[1])[0]
            base_desc += f" - 最严重: {top_issue.description[:50]}..."

        return base_desc

    def _generate_summary(
        self,
        primary: Optional[RootCause],
        secondary: List[RootCause],
        critique_result: CritiqueResult,
    ) -> str:
        """生成诊断总结"""
        if not primary:
            return "诊断完成，未发现明显问题"

        parts = [f"诊断发现 {len([primary] + secondary)} 类问题"]
        parts.append(f"主要问题: {primary.category.value} (严重度: {primary.severity_score:.2f})")

        if secondary:
            sec_cats = [c.category.value for c in secondary[:2]]
            parts.append(f"次要问题: {', '.join(sec_cats)}")

        # 添加建议
        suggestions = {
            IssueCategory.ENTITY_COUNT: "建议：调整 prompt 强调人数 / 换 seed 重试",
            IssueCategory.IDENTITY: "建议：增大 ip_adapter_scale / 使用更好的参考图",
            IssueCategory.STYLE: "建议：增大 ip_adapter_scale / 在 prompt 中强调风格",
            IssueCategory.QUALITY: "建议：增加推理步数 / 换 seed 重试",
            IssueCategory.POSE_MOTION: "建议：换 seed 重试 / 调整 prompt 描述",
            IssueCategory.SCENE: "建议：添加 location 参考图 / 在 prompt 中描述光线",
        }

        if primary.category in suggestions:
            parts.append(suggestions[primary.category])

        return " | ".join(parts)

    def _print_diagnosis(self, result: DiagnosisResult):
        """打印诊断结果"""
        print(f"\n[RootCause] ── 根因诊断 ──")
        print(f"[RootCause] 整体严重度: {result.overall_severity:.2f}")

        if result.primary_cause:
            cat = result.primary_cause.category.value
            sev = result.primary_cause.severity_score
            print(f"[RootCause] 🔴 主因: {cat} (严重度={sev:.2f})")
            print(f"[RootCause]    {result.primary_cause.description}")
            if result.primary_cause.affected_entities:
                print(f"[RootCause]    影响实体: {result.primary_cause.affected_entities}")

        for i, cause in enumerate(result.secondary_causes[:2], 1):
            print(f"[RootCause] 🟡 次因{i}: {cause.category.value} (严重度={cause.severity_score:.2f})")

        print(f"[RootCause] 总结: {result.diagnosis_summary}")


# ── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 构造测试数据
    from verification.video_critic import CritiqueIssue, CritiqueResult, IssueType, IssueSeverity

    test_issues = [
        CritiqueIssue(
            issue_type=IssueType.IDENTITY_MISMATCH,
            severity=IssueSeverity.CRITICAL,
            description="角色面部与参考图严重不符，胡子形状从络腮胡变成了山羊胡",
            affected_entity="char_bearded_man",
            affected_region="face",
            confidence=0.9,
        ),
        CritiqueIssue(
            issue_type=IssueType.STYLE_DRIFT,
            severity=IssueSeverity.HIGH,
            description="视频整体呈现动画风格，与真实感参考图不符",
            affected_entity=None,
            affected_region=None,
            confidence=0.85,
        ),
        CritiqueIssue(
            issue_type=IssueType.ENTITY_COUNT_MISMATCH,
            severity=IssueSeverity.CRITICAL,
            description="预期2人，实际检测到3人",
            affected_entity=None,
            affected_region=None,
            confidence=0.95,
        ),
    ]

    test_result = CritiqueResult(
        overall_score=0.4,
        passed=False,
        issues=test_issues,
        suggestions=[],
        analysis_summary="测试",
        frame_analyses=[],
    )

    analyzer = RootCauseAnalyzer(verbose=True)
    diagnosis = analyzer.diagnose(test_result)

    print(f"\n主因类别: {diagnosis.primary_cause.category if diagnosis.primary_cause else None}")
    print(f"所有类别: {[c.value for c in diagnosis.categories]}")
