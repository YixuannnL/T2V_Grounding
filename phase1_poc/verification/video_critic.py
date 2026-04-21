"""
verification/video_critic.py

Self-Critique & Reflection Loop for Video Generation

核心功能：
- 使用 VLM 深度分析生成视频与参考图的一致性
- 识别具体问题（而不只是给一个分数）
- 提供针对性的修复建议
- 支持多维度评估（身份、姿态、光线、数量等）

与简单 CLIP 分数的区别：
- CLIP: 只知道"分数低"，不知道为什么
- Critique: 知道"胡子形状变了"、"光线偏冷"等具体问题
"""

import os
import sys
import json
import base64
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.llm_client import LLMClient


class IssueSeverity(Enum):
    """问题严重程度"""
    CRITICAL = "critical"   # 必须修复，否则不可接受
    HIGH = "high"           # 严重影响一致性
    MEDIUM = "medium"       # 有明显差异但可接受
    LOW = "low"             # 轻微差异


class IssueType(Enum):
    """问题类型"""
    IDENTITY_MISMATCH = "identity_mismatch"         # 身份/外观不一致
    FACIAL_FEATURE_MISMATCH = "facial_feature_mismatch"  # 面部特征不一致
    CLOTHING_MISMATCH = "clothing_mismatch"         # 服装不一致
    POSE_UNNATURAL = "pose_unnatural"               # 姿态不自然
    LIGHTING_INCONSISTENT = "lighting_inconsistent" # 光线不一致
    ENTITY_MISSING = "entity_missing"               # 实体缺失
    ENTITY_EXTRA = "entity_extra"                   # 多余实体
    ENTITY_COUNT_MISMATCH = "entity_count_mismatch" # 数量不匹配
    STYLE_DRIFT = "style_drift"                     # 风格漂移（如动画化）
    MOTION_ARTIFACT = "motion_artifact"             # 动作伪影/不连贯
    QUALITY_DEGRADATION = "quality_degradation"     # 画质下降
    OTHER = "other"


@dataclass
class CritiqueIssue:
    """单个问题"""
    issue_type: IssueType
    severity: IssueSeverity
    description: str                    # 问题描述
    affected_entity: Optional[str]      # 影响的实体 ID
    affected_region: Optional[str]      # 影响的区域（如 "face", "clothing"）
    confidence: float                   # 置信度 0-1


@dataclass
class RepairSuggestion:
    """修复建议"""
    action: str                         # 建议的动作
    target: str                         # 修改目标（如 "ip_adapter_scale", "prompt", "reference"）
    detail: str                         # 具体细节
    priority: int                       # 优先级 1-5，1 最高


@dataclass
class CritiqueResult:
    """Critique 完整结果"""
    overall_score: float                # 综合评分 0-1
    passed: bool                        # 是否通过（可接受）
    issues: List[CritiqueIssue]         # 发现的问题列表
    suggestions: List[RepairSuggestion] # 修复建议列表
    analysis_summary: str               # 分析总结
    frame_analyses: List[Dict]          # 各帧的分析结果
    metadata: Dict = field(default_factory=dict)

    @property
    def critical_issues(self) -> List[CritiqueIssue]:
        return [i for i in self.issues if i.severity == IssueSeverity.CRITICAL]

    @property
    def high_issues(self) -> List[CritiqueIssue]:
        return [i for i in self.issues if i.severity == IssueSeverity.HIGH]

    def get_issues_by_entity(self, entity_id: str) -> List[CritiqueIssue]:
        return [i for i in self.issues if i.affected_entity == entity_id]

    def should_retry(self) -> bool:
        """判断是否应该重试"""
        return len(self.critical_issues) > 0 or len(self.high_issues) >= 2


# ── VLM Prompts ──────────────────────────────────────────────────────────────

CRITIQUE_SYSTEM_PROMPT = """你是一个专业的视频质量评审专家，专门评估 AI 生成视频与参考图的一致性。

## 你的任务
分析生成的视频帧，对比参考图，找出所有一致性问题，并提供针对性的修复建议。

## 评估维度

### 1. 身份一致性 (Identity)
- 面部特征：脸型、眼睛、鼻子、嘴巴、耳朵形状
- 发型发色：长度、颜色、卷曲度、分界线
- 肤色：整体色调、斑点、皱纹
- 体型：身材比例、身高感

### 2. 服装一致性 (Clothing)
- 服装款式：类型、剪裁、层次
- 颜色图案：主色调、条纹/印花
- 配饰：眼镜、帽子、首饰、包

### 3. 姿态自然度 (Pose)
- 人体比例：四肢长度、头身比
- 姿势合理性：关节角度、重心
- 动作连贯性：帧间过渡

### 4. 光线一致性 (Lighting)
- 光线方向：主光源位置
- 色温：冷暖色调
- 阴影：位置、强度

### 5. 风格一致性 (Style)
- 画面风格：真实感 vs 动画化
- 渲染质量：清晰度、细节层次
- 整体氛围：色彩饱和度、对比度

### 6. 实体完整性 (Entity)
- 实体数量：人数、物体数量
- 实体存在：预期的实体是否出现
- 额外实体：是否有不应出现的元素

## 问题严重程度定义

- **critical**: 严重破坏一致性，必须修复
  - 例：人脸完全不像、人数错误、风格完全动画化
- **high**: 明显影响观感，强烈建议修复
  - 例：发型大变、衣服颜色明显不同、光线方向相反
- **medium**: 有差异但可接受
  - 例：细微表情差异、轻微色调偏移
- **low**: 几乎不影响
  - 例：背景细节变化、极细微纹理差异

## 输出格式

```json
{
  "overall_score": 0.75,
  "passed": true,
  "issues": [
    {
      "issue_type": "facial_feature_mismatch",
      "severity": "high",
      "description": "胡子形状从络腮胡变成山羊胡",
      "affected_entity": "char_bearded_man",
      "affected_region": "lower_face",
      "confidence": 0.9
    }
  ],
  "suggestions": [
    {
      "action": "increase_ip_adapter_scale",
      "target": "ip_adapter_scale",
      "detail": "从 0.6 增加到 0.85，强化面部特征保持",
      "priority": 1
    },
    {
      "action": "add_prompt_detail",
      "target": "prompt",
      "detail": "在 prompt 中添加 'full thick beard, dark facial hair'",
      "priority": 2
    }
  ],
  "analysis_summary": "视频整体质量尚可，但面部毛发特征有明显偏差..."
}
```

## 修复建议类型

1. **increase_ip_adapter_scale**: 增大参考图权重
2. **decrease_ip_adapter_scale**: 减小参考图权重（当姿态受限时）
3. **change_reference**: 更换参考图
4. **add_prompt_detail**: 在 prompt 中添加细节描述
5. **remove_prompt_conflict**: 移除 prompt 中的冲突描述
6. **adjust_lighting_prompt**: 调整光线描述
7. **increase_inference_steps**: 增加推理步数（提升质量）
8. **change_seed**: 更换随机种子
9. **use_t2v_fallback**: 切换到纯文本生成模式

请基于以上标准，仔细分析提供的视频帧和参考图。
"""


class VideoQualityCritic:
    """
    视频质量评审专家

    使用 VLM 深度分析生成视频与参考图的一致性，
    识别具体问题并提供针对性修复建议。

    Usage:
        critic = VideoQualityCritic()
        result = critic.critique(
            video_path="output/shot_01.mp4",
            reference_images={"char_alex": ["crops/char_alex_shot1.jpg"]},
            expected_entities=[{"entity_id": "char_alex", "type": "character", ...}],
            shot_text="Alex walks into the room"
        )

        if result.should_retry():
            print(f"发现问题: {result.issues}")
            print(f"修复建议: {result.suggestions}")
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6-Anthropic",
        pass_threshold: float = 0.7,
        sample_frames: int = 5,          # 采样帧数
        max_image_size: int = 512,       # 图片最大边长（节省 token）
        verbose: bool = True,
    ):
        self.llm = LLMClient(model=model)
        self.pass_threshold = pass_threshold
        self.sample_frames = sample_frames
        self.max_image_size = max_image_size
        self.verbose = verbose

    def critique(
        self,
        video_path: str,
        reference_images: Dict[str, List[str]],  # entity_id -> [crop_paths]
        expected_entities: List[Dict],            # 预期的实体列表
        shot_text: str,
        entity_counts: Optional[List[Dict]] = None,  # 预期实体数量
    ) -> CritiqueResult:
        """
        对生成的视频进行深度 Critique

        Args:
            video_path: 生成的视频路径
            reference_images: 参考图，按实体 ID 组织
            expected_entities: 预期出现的实体信息
            shot_text: 镜头文本描述
            entity_counts: 预期实体数量（可选）

        Returns:
            CritiqueResult 包含问题列表和修复建议
        """
        if self.verbose:
            print(f"[Critic] 开始分析视频: {video_path}")

        # Step 1: 从视频采样关键帧
        frames = self._sample_video_frames(video_path)
        if not frames:
            return self._empty_result("无法读取视频帧")

        # Step 2: 准备参考图
        ref_images = self._load_reference_images(reference_images)

        # Step 3: 构建 VLM 请求
        critique_result = self._call_vlm_critique(
            frames=frames,
            ref_images=ref_images,
            expected_entities=expected_entities,
            shot_text=shot_text,
            entity_counts=entity_counts,
        )

        if self.verbose:
            self._print_result(critique_result)

        return critique_result

    def _sample_video_frames(self, video_path: str) -> List[np.ndarray]:
        """从视频均匀采样帧"""
        if not os.path.exists(video_path):
            print(f"[Critic] 视频文件不存在: {video_path}")
            return []

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            cap.release()
            return []

        # 均匀采样
        indices = np.linspace(0, total_frames - 1, self.sample_frames, dtype=int)
        frames = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # BGR -> RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize to save tokens
                frame_rgb = self._resize_image(frame_rgb)
                frames.append(frame_rgb)

        cap.release()
        if self.verbose:
            print(f"[Critic] 采样 {len(frames)} 帧 (总帧数: {total_frames})")

        return frames

    def _load_reference_images(
        self,
        reference_images: Dict[str, List[str]]
    ) -> Dict[str, List[np.ndarray]]:
        """加载参考图"""
        result = {}
        for entity_id, paths in reference_images.items():
            images = []
            for path in paths[:2]:  # 每个实体最多 2 张参考图
                if os.path.exists(path):
                    img = cv2.imread(path)
                    if img is not None:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img_rgb = self._resize_image(img_rgb)
                        images.append(img_rgb)
            if images:
                result[entity_id] = images

        if self.verbose:
            print(f"[Critic] 加载参考图: {list(result.keys())}")

        return result

    def _resize_image(self, img: np.ndarray) -> np.ndarray:
        """缩放图片以节省 token"""
        h, w = img.shape[:2]
        if max(h, w) <= self.max_image_size:
            return img

        scale = self.max_image_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _image_to_base64(self, img: np.ndarray) -> str:
        """将图片转为 base64"""
        pil_img = Image.fromarray(img)
        import io
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _call_vlm_critique(
        self,
        frames: List[np.ndarray],
        ref_images: Dict[str, List[np.ndarray]],
        expected_entities: List[Dict],
        shot_text: str,
        entity_counts: Optional[List[Dict]],
    ) -> CritiqueResult:
        """调用 VLM 进行 Critique"""

        # 构建消息内容
        content_parts = []

        # 文本描述部分
        text_content = f"""## 任务
请分析以下生成的视频帧，对比参考图，评估一致性并找出问题。

## 镜头描述
{shot_text}

## 预期实体
"""
        for entity in expected_entities:
            text_content += f"- {entity.get('entity_id', 'unknown')}: {entity.get('text_description', '')} (type: {entity.get('type', 'unknown')})\n"

        if entity_counts:
            text_content += "\n## 预期实体数量\n"
            for ec in entity_counts:
                text_content += f"- {ec.get('entity_type', 'unknown')}: {ec.get('expected_count', '?')} 个\n"

        content_parts.append({"type": "text", "text": text_content})

        # 添加参考图
        content_parts.append({"type": "text", "text": "\n## 参考图\n"})
        for entity_id, images in ref_images.items():
            content_parts.append({"type": "text", "text": f"\n### {entity_id} 的参考图:\n"})
            for i, img in enumerate(images):
                b64 = self._image_to_base64(img)
                content_parts.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": b64
                    }
                })

        # 添加生成的视频帧
        content_parts.append({"type": "text", "text": "\n## 生成的视频帧（按时间顺序）\n"})
        for i, frame in enumerate(frames):
            content_parts.append({"type": "text", "text": f"\n### 帧 {i+1}/{len(frames)}:\n"})
            b64 = self._image_to_base64(frame)
            content_parts.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": b64
                }
            })

        content_parts.append({
            "type": "text",
            "text": "\n请仔细对比参考图和生成的视频帧，输出 JSON 格式的分析结果。"
        })

        # 调用 LLM
        try:
            response = self.llm.chat_with_images(
                content_parts=content_parts,
                system=CRITIQUE_SYSTEM_PROMPT,
                max_tokens=4096,
            )
            return self._parse_critique_response(response)
        except Exception as e:
            print(f"[Critic] VLM 调用失败: {e}")
            # 回退到简单评分
            return self._fallback_critique(frames, ref_images)

    def _parse_critique_response(self, response: str) -> CritiqueResult:
        """解析 VLM 响应"""
        # 提取 JSON
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
            response = response.strip()

        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            print(f"[Critic] JSON 解析失败: {e}")
            return self._empty_result(f"JSON 解析失败: {e}")

        # 解析 issues
        issues = []
        for issue_data in data.get("issues", []):
            try:
                issue_type = IssueType(issue_data.get("issue_type", "other"))
            except ValueError:
                issue_type = IssueType.OTHER

            try:
                severity = IssueSeverity(issue_data.get("severity", "medium"))
            except ValueError:
                severity = IssueSeverity.MEDIUM

            issues.append(CritiqueIssue(
                issue_type=issue_type,
                severity=severity,
                description=issue_data.get("description", ""),
                affected_entity=issue_data.get("affected_entity"),
                affected_region=issue_data.get("affected_region"),
                confidence=float(issue_data.get("confidence", 0.5)),
            ))

        # 解析 suggestions
        suggestions = []
        for sug_data in data.get("suggestions", []):
            suggestions.append(RepairSuggestion(
                action=sug_data.get("action", ""),
                target=sug_data.get("target", ""),
                detail=sug_data.get("detail", ""),
                priority=int(sug_data.get("priority", 3)),
            ))

        # 按优先级排序
        suggestions.sort(key=lambda s: s.priority)

        overall_score = float(data.get("overall_score", 0.5))
        passed = data.get("passed", overall_score >= self.pass_threshold)

        return CritiqueResult(
            overall_score=overall_score,
            passed=passed,
            issues=issues,
            suggestions=suggestions,
            analysis_summary=data.get("analysis_summary", ""),
            frame_analyses=data.get("frame_analyses", []),
            metadata={"raw_response": response[:500]},
        )

    def _fallback_critique(
        self,
        frames: List[np.ndarray],
        ref_images: Dict[str, List[np.ndarray]],
    ) -> CritiqueResult:
        """
        VLM 失败时的回退方案：使用 CLIP 简单评分

        这是降级策略，只提供分数，不提供具体问题分析
        """
        try:
            import clip
            import torch

            model, preprocess = clip.load("ViT-B/32", device="cuda")

            # 计算帧与参考图的平均相似度
            scores = []
            with torch.no_grad():
                for frame in frames[:3]:  # 只用前 3 帧
                    frame_pil = Image.fromarray(frame)
                    frame_tensor = preprocess(frame_pil).unsqueeze(0).cuda()
                    frame_feat = model.encode_image(frame_tensor)

                    for entity_id, images in ref_images.items():
                        for ref_img in images:
                            ref_pil = Image.fromarray(ref_img)
                            ref_tensor = preprocess(ref_pil).unsqueeze(0).cuda()
                            ref_feat = model.encode_image(ref_tensor)
                            cos_sim = torch.nn.functional.cosine_similarity(
                                frame_feat, ref_feat
                            ).item()
                            scores.append(cos_sim)

            overall_score = float(np.mean(scores)) if scores else 0.5
            passed = overall_score >= self.pass_threshold

            return CritiqueResult(
                overall_score=overall_score,
                passed=passed,
                issues=[],
                suggestions=[],
                analysis_summary=f"[Fallback] CLIP 相似度评分: {overall_score:.3f}",
                frame_analyses=[],
                metadata={"fallback": True},
            )

        except Exception as e:
            print(f"[Critic] Fallback CLIP 评分也失败: {e}")
            return self._empty_result(f"评分失败: {e}")

    def _empty_result(self, reason: str) -> CritiqueResult:
        """返回空结果"""
        return CritiqueResult(
            overall_score=0.5,
            passed=True,  # 失败时默认通过，避免无限重试
            issues=[],
            suggestions=[],
            analysis_summary=f"[Error] {reason}",
            frame_analyses=[],
            metadata={"error": reason},
        )

    def _print_result(self, result: CritiqueResult):
        """打印 Critique 结果"""
        status = "✅ 通过" if result.passed else "❌ 需要修复"
        print(f"[Critic] 评分: {result.overall_score:.2f} | {status}")

        if result.issues:
            print(f"[Critic] 发现 {len(result.issues)} 个问题:")
            for issue in result.issues:
                icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(
                    issue.severity.value, "⚪"
                )
                print(f"         {icon} [{issue.severity.value}] {issue.description}")

        if result.suggestions and not result.passed:
            print(f"[Critic] 修复建议:")
            for i, sug in enumerate(result.suggestions[:3], 1):
                print(f"         {i}. {sug.action}: {sug.detail}")


# ── Repair Strategy Generator ────────────────────────────────────────────────

class RepairStrategyGenerator:
    """
    根据 Critique 结果生成修复策略

    将抽象的修复建议转化为具体的参数调整
    """

    def __init__(self):
        # 默认参数范围
        self.param_ranges = {
            "ip_adapter_scale": (0.3, 1.0),
            "guide_scale_text": (3.0, 15.0),
            "guide_scale_img": (3.0, 10.0),
            "num_inference_steps": (30, 100),
        }

    def generate_repair_params(
        self,
        critique_result: CritiqueResult,
        current_params: Dict,
    ) -> Dict:
        """
        根据 Critique 结果生成修复参数

        Args:
            critique_result: Critique 结果
            current_params: 当前使用的参数

        Returns:
            修复后的参数字典
        """
        new_params = current_params.copy()

        for suggestion in critique_result.suggestions:
            action = suggestion.action

            if action == "increase_ip_adapter_scale":
                current = new_params.get("ip_adapter_scale", 0.6)
                new_params["ip_adapter_scale"] = min(current + 0.15, 1.0)

            elif action == "decrease_ip_adapter_scale":
                current = new_params.get("ip_adapter_scale", 0.6)
                new_params["ip_adapter_scale"] = max(current - 0.15, 0.3)

            elif action == "increase_inference_steps":
                current = new_params.get("num_inference_steps", 50)
                new_params["num_inference_steps"] = min(current + 20, 100)

            elif action == "change_seed":
                import random
                new_params["seed"] = random.randint(0, 2**31 - 1)

            elif action == "add_prompt_detail":
                # 将建议添加到 prompt 增强列表
                if "prompt_additions" not in new_params:
                    new_params["prompt_additions"] = []
                new_params["prompt_additions"].append(suggestion.detail)

            elif action == "use_t2v_fallback":
                new_params["force_t2v"] = True

            elif action == "change_reference":
                new_params["change_reference_for"] = suggestion.detail

        return new_params

    def get_reference_change_suggestion(
        self,
        critique_result: CritiqueResult,
    ) -> Optional[Tuple[str, str]]:
        """
        获取参考图更换建议

        Returns:
            (entity_id, reason) 或 None
        """
        for suggestion in critique_result.suggestions:
            if suggestion.action == "change_reference":
                # 从 detail 中提取实体 ID
                # 格式通常是 "使用 Shot X 的参考图替代" 或 "entity_id: reason"
                return (suggestion.target, suggestion.detail)

        return None


# ── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 测试用例
    critic = VideoQualityCritic(verbose=True)

    # Mock 测试（需要实际视频和参考图）
    test_video = "./data/test/video.mp4"
    if os.path.exists(test_video):
        result = critic.critique(
            video_path=test_video,
            reference_images={},
            expected_entities=[
                {"entity_id": "char_main", "type": "character", "text_description": "main character"}
            ],
            shot_text="A person walks through the scene",
        )
        print(f"\n综合评分: {result.overall_score}")
        print(f"通过: {result.passed}")
        print(f"问题数: {len(result.issues)}")
    else:
        print(f"测试视频不存在: {test_video}")
