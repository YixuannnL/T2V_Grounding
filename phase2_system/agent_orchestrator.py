"""
phase2_system/agent_orchestrator.py
职责: Phase 2 Agentic Pipeline — LLM 作为 Orchestrator，通过工具调用循环
      自动决策 reference 选择策略、检测一致性失败并 Retry
"""

import os
import sys
import json
from typing import List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase1_poc"))
from utils.llm_client import (
    LLMClient,
    assistant_message_with_tool_calls,
    tool_result_message,
)

# ── 工具定义（供 LLM 调用）────────────────────────────────────────────────────
TOOLS = [
    {
        "name": "query_references",
        "description": "查询指定实体的历史参考图列表，返回质量排序结果",
        "input_schema": {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string", "description": "实体唯一 ID"},
                "top_k": {"type": "integer", "description": "返回数量", "default": 3},
                "prefer_recent": {"type": "boolean", "description": "是否优先最近镜头"}
            },
            "required": ["entity_id"]
        }
    },
    {
        "name": "run_grounding",
        "description": "在指定历史视频中对实体做 Visual Grounding，返回找到的参考帧",
        "input_schema": {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string"},
                "entity_text": {"type": "string", "description": "实体文本描述"},
                "video_paths": {"type": "array", "items": {"type": "string"}, "description": "历史视频路径列表"},
                "max_results": {"type": "integer", "default": 3}
            },
            "required": ["entity_id", "entity_text", "video_paths"]
        }
    },
    {
        "name": "generate_video",
        "description": "调用 Reference2Video 模型生成视频镜头",
        "input_schema": {
            "type": "object",
            "properties": {
                "shot_text": {"type": "string", "description": "镜头文本描述"},
                "reference_paths": {"type": "array", "items": {"type": "string"}, "description": "参考图路径列表"},
                "ip_adapter_scale": {"type": "number", "description": "reference conditioning 强度 0-1", "default": 0.6},
                "output_path": {"type": "string"}
            },
            "required": ["shot_text", "reference_paths", "output_path"]
        }
    },
    {
        "name": "check_consistency",
        "description": "检测生成视频与参考图的主体一致性分数",
        "input_schema": {
            "type": "object",
            "properties": {
                "video_path": {"type": "string"},
                "reference_paths": {"type": "array", "items": {"type": "string"}},
                "entity_type": {"type": "string", "enum": ["character", "object", "location"]}
            },
            "required": ["video_path", "reference_paths"]
        }
    },
    {
        "name": "bootstrap_reference",
        "description": "当实体无历史参考时，调用 T2I 生成参考图（冷启动）",
        "input_schema": {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string"},
                "entity_description": {"type": "string"}
            },
            "required": ["entity_id", "entity_description"]
        }
    },
    {
        "name": "finish_shot",
        "description": "确认当前镜头生成完成，返回最终视频路径",
        "input_schema": {
            "type": "object",
            "properties": {
                "video_path": {"type": "string"},
                "consistency_score": {"type": "number"},
                "notes": {"type": "string"}
            },
            "required": ["video_path", "consistency_score"]
        }
    }
]


SYSTEM_PROMPT = """你是一个视频生成 Agent，负责决策如何生成当前镜头以保持跨镜头主体一致性。

你的工作流程：
1. 分析当前镜头需要哪些主体参考图
2. 查询 registry 获取历史参考，或触发 grounding / 冷启动
3. 调用视频生成
4. 检查一致性分数：
   - 若 ≥ 0.7：直接完成
   - 若 0.5-0.7：调整参数后重试一次
   - 若 < 0.5：分析失败原因，换策略重试（最多 3 次）
5. 调用 finish_shot 结束

失败时的调整策略：
- 参考图质量太低 → 换用不同历史镜头的参考
- 姿态差异大 → 降低 ip_adapter_scale，增加文本权重
- 多实体参考冲突 → 只保留最重要实体的参考
- 完全找不到参考 → bootstrap_reference 冷启动
"""


class AgentOrchestrator:
    """
    Phase 2 Agentic Orchestrator
    使用 Claude 工具调用循环自主决策每个镜头的生成策略
    """

    def __init__(
        self,
        pipeline,
        model: str = "claude-sonnet-4-6-jh",
        max_retries: int = 3,
        consistency_threshold: float = 0.7,
    ):
        self.pipeline = pipeline
        self.max_retries = max_retries
        self.consistency_threshold = consistency_threshold
        self.llm = LLMClient(model=model)

    def run_shot(self, shot_id: int, shot_text: str, entity_list: list) -> dict:
        """用 Agent 循环处理单个镜头"""
        context = json.dumps({
            "shot_id": shot_id,
            "shot_text": shot_text,
            "entities": entity_list,
            "history_video_paths": self.pipeline._generated_videos,
            "max_retries": self.max_retries,
            "consistency_threshold": self.consistency_threshold,
        }, ensure_ascii=False)

        messages = [{"role": "user", "content": f"请处理以下镜头：\n{context}"}]
        retry_count = 0

        while retry_count <= self.max_retries:
            reply_text, tool_calls, stop_reason, raw_msg = self.llm.chat_with_tools(
                messages=messages,
                tools=TOOLS,
                system=SYSTEM_PROMPT,
                max_tokens=4096,
            )

            # 没有工具调用 → Agent 认为完成
            if stop_reason == "end_turn":
                print(f"[Agent] Shot {shot_id} 完成（无工具调用）")
                break

            # 将 assistant 消息（含 tool_calls）加入历史
            messages.append(assistant_message_with_tool_calls(raw_msg))

            # 处理每个工具调用，收集结果
            finished = False
            for tc in (tool_calls or []):
                tool_name = tc.function.name
                tool_input = json.loads(tc.function.arguments)

                result = self._dispatch_tool(tool_name, tool_input)

                # 将 tool result 加入对话历史
                messages.append(tool_result_message(tc.id, json.dumps(result, ensure_ascii=False)))

                if tool_name == "finish_shot":
                    finished = True
                    print(f"[Agent] Shot {shot_id} 完成，一致性分={result.get('consistency_score')}")

            if finished:
                break
            retry_count += 1

        return {"shot_id": shot_id, "messages": len(messages)}

    def _dispatch_tool(self, tool_name: str, tool_input: dict) -> dict:
        """将 Agent 工具调用路由到具体实现"""
        print(f"[Agent] 调用工具: {tool_name}({list(tool_input.keys())})")

        if tool_name == "query_references":
            refs = self.pipeline.registry.query(
                tool_input["entity_id"],
                top_k=tool_input.get("top_k", 3),
                prefer_recent=tool_input.get("prefer_recent", True),
            )
            return {"references": [{"path": r.crop_path, "score": r.quality_score, "shot_id": r.shot_id} for r in refs]}

        elif tool_name == "run_grounding":
            # 调用 grounding 模块（简化版）
            from visual_grounding.grounder import extract_frames
            crops = []
            for vp in tool_input["video_paths"][-2:]:
                frames = extract_frames(vp, f"/tmp/frames_{tool_input['entity_id']}", fps=1.0)
                results = self.pipeline.grounder.ground_in_frames(
                    frames, tool_input["entity_text"], tool_input["entity_id"],
                    f"/tmp/crops_{tool_input['entity_id']}", max_results=2
                )
                crops.extend([r.crop_path for r in results])
            return {"crop_paths": crops, "count": len(crops)}

        elif tool_name == "generate_video":
            from PIL import Image
            refs = [Image.open(p).convert("RGB") for p in tool_input["reference_paths"] if os.path.exists(p)]
            if not refs:
                refs = [Image.new("RGB", (848, 480), (50, 50, 50))]
            self.pipeline.generator.generate(
                text_prompt=tool_input["shot_text"],
                references=refs,
                output_path=tool_input["output_path"],
            )
            return {"video_path": tool_input["output_path"], "success": True}

        elif tool_name == "check_consistency":
            # 简化版：用 CLIP-I 评分
            score = self._compute_consistency(
                tool_input["video_path"],
                tool_input["reference_paths"],
                tool_input.get("entity_type", "character"),
            )
            return {"consistency_score": score, "passed": score >= self.consistency_threshold}

        elif tool_name == "bootstrap_reference":
            from reference_manager.registry import ReferenceEntry
            entry = self.pipeline.registry.bootstrap_entity(
                tool_input["entity_id"],
                tool_input["entity_description"],
                output_dir=os.path.join(self.pipeline.output_dir, "crops", "bootstrap"),
            )
            return {"crop_path": entry.crop_path if entry else None, "success": entry is not None}

        elif tool_name == "finish_shot":
            return tool_input  # 直接透传

        return {"error": f"未知工具: {tool_name}"}

    def _compute_consistency(self, video_path: str, ref_paths: List[str], entity_type: str) -> float:
        """计算视频帧与参考图的 CLIP-I 相似度（简化版）"""
        try:
            import clip
            import torch
            import cv2
            import numpy as np
            from PIL import Image

            model, preprocess = clip.load("ViT-B/32", device="cpu")

            # 取视频第 1 帧
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return 0.5

            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_tensor = preprocess(frame_pil).unsqueeze(0)

            scores = []
            with torch.no_grad():
                frame_feat = model.encode_image(frame_tensor)
                for rp in ref_paths:
                    if not os.path.exists(rp):
                        continue
                    ref_pil = Image.open(rp).convert("RGB")
                    ref_tensor = preprocess(ref_pil).unsqueeze(0)
                    ref_feat = model.encode_image(ref_tensor)
                    cos_sim = torch.nn.functional.cosine_similarity(frame_feat, ref_feat).item()
                    scores.append(cos_sim)

            return float(np.mean(scores)) if scores else 0.5
        except Exception as e:
            print(f"[Agent] 一致性评分失败: {e}，返回默认分 0.6")
            return 0.6
