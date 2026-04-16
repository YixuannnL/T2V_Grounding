"""
entity_parser/parser.py
职责: 调用 LLM 从镜头文本中提取结构化实体列表，并做跨镜头共指消解
"""

import json
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.llm_client import LLMClient


@dataclass
class Entity:
    entity_id: str            # 跨镜头唯一 ID，如 "char_alex"
    type: str                 # character / object / location / style
    text_description: str     # 原文描述
    attributes: dict          # 附加属性，如 {"gender": "male", "age": "30s"}
    is_new: bool              # 是否首次出现
    grounding_priority: str   # high / medium / low
    aliases: List[str] = field(default_factory=list)  # 别名列表


@dataclass
class GlobalContext:
    """
    从 global_caption 中提取的全局语义信息，
    这些信息与具体实体无关，但需要体现在每个 shot 中。
    """
    visual_style: str = ""      # 视觉风格：如 "cinematic, dramatic lighting"
    mood: str = ""              # 情绪氛围：如 "tense, suspenseful"
    setting: str = ""           # 场景设定：如 "sandy desolate desert environment"
    narrative_context: str = "" # 叙事背景：如 "a standoff between two characters"


@dataclass
class ParseResult:
    shot_id: int
    raw_text: str
    entities: List[Entity]
    new_elements: List[str]   # 本镜头新增的无需 reference 的元素（如天气、特效）
    style_ref_shot: Optional[int] = None  # 参考哪个镜头的整体风格


SYSTEM_PROMPT = """你是一个专业的影视脚本分析器。
给定一个镜头的文本描述，以及已知的实体列表（跨镜头上下文），你需要：
1. 提取该镜头中出现的所有实体（人物、重要物体、场景/地点、整体风格）
2. 判断每个实体是否已在历史镜头中出现（共指消解）
3. 为新实体生成唯一 ID（格式: char_xxx / obj_xxx / loc_xxx / style_xxx）
4. 评估每个实体的 grounding 优先级：
   - high: 人物角色（character）、核心互动物体
   - medium: 场景/地点（location）、重要道具 —— 注意：location 类型必须是 medium，不能是 low
   - low: 纯背景元素、不需要跨镜头保持一致的物体

**重要**：每个镜头必须包含至少一个 location 类型的实体（type="location"），
用于描述当前场景环境。location 的 grounding_priority 必须设为 "medium"，
这对于保持跨镜头的场景一致性至关重要。

如果提供了 global_caption（全局背景描述），请以它为准来理解人物外观和设定。
当镜头中出现代词或简短指代（如 "the man"、"he"、"the masked figure"）时，
请根据 global_caption 和已知实体做共指消解，映射到正确的 entity_id。

输出严格遵循以下 JSON 格式，不要输出任何其他内容：
{
  "entities": [
    {
      "entity_id": "char_alex",
      "type": "character",
      "text_description": "Alex, a detective in trench coat",
      "attributes": {"gender": "male", "age": "30s", "clothing": "trench coat"},
      "is_new": false,
      "grounding_priority": "high",
      "aliases": ["the detective", "he", "Alex Chen"]
    },
    {
      "entity_id": "loc_police_station",
      "type": "location",
      "text_description": "police station interior",
      "attributes": {"lighting": "fluorescent", "atmosphere": "busy"},
      "is_new": true,
      "grounding_priority": "medium",
      "aliases": ["the station", "inside the precinct"]
    }
  ],
  "new_elements": ["heavy rain", "neon signs"],
  "style_ref_shot": 1
}
"""


GLOBAL_CONTEXT_PROMPT = """你是一个专业的影视脚本分析器。
给定一段全局背景描述（global_caption），请从中提取与具体角色/物体无关的全局语义信息。

这些信息将用于指导每个镜头的视频生成，确保风格和氛围一致。

请提取以下四个维度的信息（如果描述中没有相关信息，留空字符串）：

1. visual_style: 视觉风格描述（如 "cinematic", "dramatic lighting", "high contrast"）
2. mood: 情绪氛围（如 "tense", "suspenseful", "melancholic", "action-packed"）
3. setting: 整体场景设定（如 "sandy desolate desert", "rainy urban night"）
4. narrative_context: 叙事背景/情节概要（如 "a standoff", "a chase sequence", "a reunion"）

注意：
- 不要包含具体角色名称或外观描述（那些会单独处理）
- 提取的是跨越所有镜头的共性信息
- 用简洁的英文短语描述，便于注入视频生成 prompt

输出严格遵循以下 JSON 格式：
{
  "visual_style": "cinematic, dramatic lighting, high contrast",
  "mood": "tense, suspenseful",
  "setting": "sandy desolate desert environment with sandbags and ruins",
  "narrative_context": "a standoff and pursuit sequence"
}
"""


class EntityParser:
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-6-Anthropic"):
        self.llm = LLMClient(api_key=api_key, model=model)
        self._known_entities: List[Entity] = []  # 跨镜头已知实体
        self._global_caption: Optional[str] = None  # 全局背景描述
        self._global_context: Optional[GlobalContext] = None  # 全局语义上下文

    def set_global_caption(self, global_caption: str):
        """
        设置全局背景描述（整个视频的叙事背景）。
        会在解析 Shot 1 之前：
        1. 从 global_caption 提取所有实体，建立实体图谱
        2. 提取全局语义上下文（风格、氛围、叙事背景），用于注入每个 shot 的 prompt
        """
        self._global_caption = global_caption.strip()

        # Step 1: 提取全局语义上下文（风格、氛围、叙事背景）
        print("[Parser] 从 global_caption 提取全局语义...")
        self._global_context = self._extract_global_context(global_caption)
        print(f"[Parser] 全局语义: style='{self._global_context.visual_style}' | "
              f"mood='{self._global_context.mood}' | setting='{self._global_context.setting}'")

        # Step 2: 用 shot_id=0 解析全局实体，建立实体图谱
        print("[Parser] 从 global_caption 预建实体图谱...")
        result = self._parse_internal(global_caption, shot_id=0)
        print(f"[Parser] global_caption 解析出 {len(result.entities)} 个实体: "
              f"{[e.entity_id for e in result.entities]}")

    def _extract_global_context(self, global_caption: str) -> GlobalContext:
        """
        调用 LLM 从 global_caption 中提取全局语义信息
        """
        try:
            raw_json = self.llm.chat(
                user_message=f"请分析以下全局背景描述：\n\n{global_caption}",
                system=GLOBAL_CONTEXT_PROMPT,
                max_tokens=512,
            ).strip()

            # 兼容 LLM 输出 ```json ``` 包裹
            if raw_json.startswith("```"):
                raw_json = raw_json.split("```")[1]
                if raw_json.startswith("json"):
                    raw_json = raw_json[4:]
                raw_json = raw_json.strip()

            data = json.loads(raw_json)
            return GlobalContext(
                visual_style=data.get("visual_style", ""),
                mood=data.get("mood", ""),
                setting=data.get("setting", ""),
                narrative_context=data.get("narrative_context", ""),
            )
        except Exception as e:
            print(f"[Parser] 提取全局语义失败: {e}，使用空上下文")
            return GlobalContext()

    def get_global_context(self) -> Optional[GlobalContext]:
        """获取全局语义上下文"""
        return self._global_context

    def parse(self, shot_text: str, shot_id: int) -> ParseResult:
        """解析单个镜头文本，返回结构化实体列表"""
        return self._parse_internal(shot_text, shot_id)

    def _parse_internal(self, shot_text: str, shot_id: int) -> ParseResult:
        known_ctx = self._build_known_context()

        global_section = ""
        if self._global_caption and shot_id > 0:
            global_section = f"\n全局背景描述（以此为准理解人物外观和设定）:\n{self._global_caption}\n"

        user_message = f"""当前镜头 ID: {shot_id}
镜头描述: {shot_text}
{global_section}
已知实体（历史镜头中出现过的）:
{known_ctx}

请分析该镜头并输出 JSON。"""

        raw_json = self.llm.chat(
            user_message=user_message,
            system=SYSTEM_PROMPT,
            max_tokens=1024,  # 实体 JSON 响应通常 <500 tokens，4096 是浪费
        ).strip()
        # 兼容 LLM 偶尔输出 ```json ``` 包裹
        if raw_json.startswith("```"):
            raw_json = raw_json.split("```")[1]
            if raw_json.startswith("json"):
                raw_json = raw_json[4:]
            raw_json = raw_json.strip()

        # 截断兜底：提取第一个完整 JSON 对象，防止模型多余输出或截断
        brace_count = 0
        end_idx = -1
        for i, ch in enumerate(raw_json):
            if ch == '{':
                brace_count += 1
            elif ch == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break
        if end_idx > 0:
            raw_json = raw_json[:end_idx]

        data = json.loads(raw_json)
        entities = [Entity(**e) for e in data["entities"]]

        # 更新已知实体库（仅新实体加入）
        for entity in entities:
            if entity.is_new:
                self._known_entities.append(entity)
            else:
                # 更新别名
                existing = self._get_entity(entity.entity_id)
                if existing:
                    existing.aliases = list(set(existing.aliases + entity.aliases))

        return ParseResult(
            shot_id=shot_id,
            raw_text=shot_text,
            entities=entities,
            new_elements=data.get("new_elements", []),
            style_ref_shot=data.get("style_ref_shot")
        )

    def _build_known_context(self) -> str:
        if not self._known_entities:
            return "（无，这是第一个镜头）"
        lines = []
        for e in self._known_entities:
            lines.append(f"  - [{e.entity_id}] {e.text_description} (aliases: {e.aliases})")
        return "\n".join(lines)

    def _get_entity(self, entity_id: str) -> Optional[Entity]:
        for e in self._known_entities:
            if e.entity_id == entity_id:
                return e
        return None

    def build_shot_context(self, parse_result: "ParseResult") -> str:
        """
        为当前 shot 构建精准生成上下文，仅包含本 shot 实际出现的实体描述。

        原则：
          · 不使用原始 global_caption 文本，因为它包含整个视频的叙事，
            会把后续 shot 才出现的实体（如 masked figure）引入当前 shot 的生成 prompt。
          · 转而从 entity graph 中逐实体查找属性描述，只取本 shot 出现的实体。
          · 格式化为 "Character / Scene / Object: 描述." 便于视频模型理解。

        示例（Shot 1 仅含 char_bearded_man + loc_desert）:
          Character: bearded man (gender: male, age: 40s, weapon: sniper rifle).
          Scene: sandy desolate environment (lighting: harsh sunlight).
        """
        TYPE_LABEL = {
            "character": "Character",
            "object":    "Object",
            "location":  "Scene",
            "style":     "Style",
        }
        parts = []
        for entity in parse_result.entities:
            if entity.grounding_priority == "low":
                continue
            # 优先用 global_caption 解析时存入 known_entities 的完整属性
            source = self._get_entity(entity.entity_id) or entity
            desc = source.text_description.strip()
            if source.attributes:
                attr_str = ", ".join(
                    f"{k}: {v}" for k, v in source.attributes.items() if v
                )
                if attr_str:
                    desc = f"{desc} ({attr_str})"
            label = TYPE_LABEL.get(source.type, source.type.capitalize())
            parts.append(f"{label}: {desc}.")
        return "\n".join(parts)

    def build_global_context_prompt(self) -> str:
        """
        构建全局语义上下文的 prompt 部分。

        这部分信息来自 global_caption，但不包含具体实体描述，
        而是风格、氛围、叙事背景等跨越所有镜头的共性信息。

        返回示例：
          [Global Context]
          Visual style: cinematic, dramatic lighting.
          Mood: tense, suspenseful.
          Setting: sandy desolate desert environment.
          Narrative: a standoff and pursuit sequence.
        """
        if self._global_context is None:
            return ""

        ctx = self._global_context
        parts = []

        if ctx.visual_style:
            parts.append(f"Visual style: {ctx.visual_style}.")
        if ctx.mood:
            parts.append(f"Mood: {ctx.mood}.")
        if ctx.setting:
            parts.append(f"Setting: {ctx.setting}.")
        if ctx.narrative_context:
            parts.append(f"Narrative: {ctx.narrative_context}.")

        if not parts:
            return ""

        return "[Global Context]\n" + "\n".join(parts)

    def get_known_entities(self) -> List[Entity]:
        return self._known_entities

    def save_state(self, path: str):
        """持久化实体图谱"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump([asdict(e) for e in self._known_entities], f, ensure_ascii=False, indent=2)

    def load_state(self, path: str):
        """加载实体图谱"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._known_entities = [Entity(**e) for e in data]


# ── 快速测试 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = EntityParser()

    shots = [
        (1, "Alex Chen, a detective in his 30s wearing a trench coat, enters the police station."),
        (2, "Inside the interrogation room, Alex sits across from a nervous suspect."),
        (3, "Alex walks into the rain-soaked alley at night, his coat drenched.")
    ]

    for shot_id, text in shots:
        result = parser.parse(text, shot_id)
        print(f"\n=== Shot {shot_id} ===")
        for e in result.entities:
            print(f"  [{e.entity_id}] {e.text_description} | is_new={e.is_new} | priority={e.grounding_priority}")
        print(f"  new_elements: {result.new_elements}")
