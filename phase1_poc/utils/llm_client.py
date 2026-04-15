"""
utils/llm_client.py
统一 LLM 客户端封装 — 对接公司内部 OpenAI 兼容 API
base_url: http://yy.dbh.baidu-int.com/v1
auth:     Bearer sk-...

提供两个接口:
  - chat(messages, system, model)         → str   (普通对话)
  - chat_with_tools(messages, tools, ...)  → (content_text, tool_calls)
"""

import os
import time
import yaml
from pathlib import Path
from typing import List, Optional
from openai import OpenAI, RateLimitError

# ── 默认配置 ──────────────────────────────────────────────────────────────────
INTERNAL_BASE_URL = "http://yy.dbh.baidu-int.com/v1"

# config.yaml 路径（resolve() 保证绝对路径，避免相对路径层级偏移）
_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "config.yaml"


def _load_config() -> dict:
    """加载 configs/config.yaml，失败时打印原因并返回空 dict"""
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[LLMClient] 警告: 读取 config.yaml 失败 ({_CONFIG_PATH}): {e}")
        return {}

# 公司内部可用的 Claude 模型映射（标准名 → 内部渠道 ID）
MODEL_ALIASES = {
    # ── Haiku ──────────────────────────────────────────────────────────────────
    "claude-haiku-4-5":                     "claude-haiku-4-5-20251001",
    "claude-haiku-4-5-20251001":            "claude-haiku-4-5-20251001",
    "claude-haiku-4-5-agent":               "claude-haiku-4-5-20251001-agent",
    # ── Sonnet 4.5 ────────────────────────────────────────────────────────────
    "claude-sonnet-4-5":                    "claude-sonnet-4-5-20250929-Anthropic",
    "claude-sonnet-4-5-20250929":           "claude-sonnet-4-5-20250929",
    "claude-sonnet-4-5-20250929-Anthropic": "claude-sonnet-4-5-20250929-Anthropic",
    # ── Sonnet 4.6 ────────────────────────────────────────────────────────────
    "claude-sonnet-4-6":                    "claude-sonnet-4-6-Anthropic",
    "claude-sonnet-4-6-Anthropic":          "claude-sonnet-4-6-Anthropic",
    "claude-sonnet-4-6-chuangzuo":          "claude-sonnet-4-6-chuangzuo",
    "claude-sonnet-4-6-chuangzuo-2":        "claude-sonnet-4-6-chuangzuo-2",
    # ── Opus 4.5 ──────────────────────────────────────────────────────────────
    "claude-opus-4-5":                      "claude-opus-4-5-20251101-Anthropic",
    "claude-opus-4-5-20251101":             "claude-opus-4-5-20251101",
    "claude-opus-4-5-20251101-Anthropic":   "claude-opus-4-5-20251101-Anthropic",
    "claude-opus-4-5-chuangzuo":            "claude-opus-4-5-20251101-chuangzuo",
    "claude-opus-4-5-chuangzuo-2":          "claude-opus-4-5-20251101-chuangzuo-2",
    # ── Opus 4.6 ──────────────────────────────────────────────────────────────
    "claude-opus-4-6":                      "claude-opus-4-6",
    "claude-opus-4-6-Anthropic":            "claude-opus-4-6-Anthropic",
    "claude-opus-4-6-chuangzuo":            "claude-opus-4-6-chuangzuo",
    "claude-opus-4-6-chuangzuo-2":          "claude-opus-4-6-chuangzuo-2",
}

def _default_model() -> str:
    """从 config.yaml llm.model 读取默认模型，读不到则用 opus"""
    return _load_config().get("llm", {}).get("model", "claude-opus-4-6")


def _resolve_model(model: str) -> str:
    """将标准/别名模型名解析为内部模型 ID"""
    return MODEL_ALIASES.get(model, model)


def _get_client(api_key: Optional[str] = None) -> OpenAI:
    """创建 OpenAI 兼容客户端，key 优先级: 参数 > 环境变量 > config.yaml"""
    cfg = _load_config().get("api", {})
    key = (
        api_key
        or os.environ.get("INTERNAL_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or cfg.get("api_key")
    )
    if not key:
        raise ValueError(
            "未找到 API Key。请在 configs/config.yaml 的 api.api_key 中填写，\n"
            "或设置环境变量: export INTERNAL_API_KEY=sk-..."
        )
    base_url = cfg.get("base_url", INTERNAL_BASE_URL)
    return OpenAI(api_key=key, base_url=base_url)


class LLMClient:
    """
    统一 LLM 客户端，内部使用 OpenAI SDK 对接公司 API

    使用示例:
        client = LLMClient()
        reply = client.chat("帮我分析这段话...", system="你是...")
        text, calls = client.chat_with_tools(messages, tools=TOOLS)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.client = _get_client(api_key)
        self.model = _resolve_model(model or _default_model())

    # ── 普通对话 ──────────────────────────────────────────────────────────────
    def chat(
        self,
        user_message: str,
        system: str = "",
        model: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_message})

        for attempt in range(5):
            try:
                response = self.client.chat.completions.create(
                    model=_resolve_model(model) if model else self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response.choices[0].message.content
            except RateLimitError as e:
                wait = 15 * (2 ** attempt)  # 15s, 30s, 60s, 120s, 240s
                print(f"[LLMClient] 429 限流，{wait}s 后重试 (attempt {attempt+1}/5): {e}")
                time.sleep(wait)
        raise RuntimeError("[LLMClient] 达到最大重试次数，API 持续限流")

    # ── 多轮对话 ──────────────────────────────────────────────────────────────
    def chat_multi(
        self,
        messages: List[dict],
        system: str = "",
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> str:
        """
        多轮对话，messages 格式: [{"role": "user/assistant", "content": "..."}]
        system 会自动插入到首条消息前。
        """
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        response = self.client.chat.completions.create(
            model=_resolve_model(model) if model else self.model,
            messages=full_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content

    # ── 工具调用（Function Calling）────────────────────────────────────────────
    def chat_with_tools(
        self,
        messages: List[dict],
        tools: List[dict],
        system: str = "",
        model: Optional[str] = None,
        max_tokens: int = 4096,
    ):
        """
        支持工具调用的多轮对话（OpenAI function calling 格式）

        Args:
            messages: 对话历史（OpenAI 格式，含 tool_calls / tool 角色）
            tools:    工具定义列表（Anthropic 格式，内部自动转换为 OpenAI 格式）
            system:   系统提示词

        Returns:
            (reply_text: str | None, tool_calls: list | None, stop_reason: str)
            - stop_reason: "end_turn" | "tool_use"
        """
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        openai_tools = [_convert_tool_to_openai(t) for t in tools]

        response = self.client.chat.completions.create(
            model=_resolve_model(model) if model else self.model,
            messages=full_messages,
            tools=openai_tools,
            tool_choice="auto",
            max_tokens=max_tokens,
        )

        choice = response.choices[0]
        msg = choice.message
        finish_reason = choice.finish_reason  # "stop" | "tool_calls"

        stop_reason = "tool_use" if finish_reason == "tool_calls" else "end_turn"
        tool_calls = msg.tool_calls if msg.tool_calls else None
        reply_text = msg.content  # 可能为 None（纯工具调用时）

        return reply_text, tool_calls, stop_reason, msg


# ── 工具格式转换（Anthropic → OpenAI）───────────────────────────────────────
def _convert_tool_to_openai(tool: dict) -> dict:
    """
    将 Anthropic tool 格式转换为 OpenAI function calling 格式

    Anthropic:
      {"name": "...", "description": "...", "input_schema": {...}}

    OpenAI:
      {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
    """
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
        },
    }


def tool_result_message(tool_call_id: str, content: str) -> dict:
    """构造 tool 结果消息（OpenAI 格式）"""
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": content,
    }


def assistant_message_with_tool_calls(msg) -> dict:
    """将 OpenAI response message 转换为对话历史中的 assistant 消息"""
    return {
        "role": "assistant",
        "content": msg.content,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in (msg.tool_calls or [])
        ],
    }


# ── 快速测试 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 模型和 key 均从 configs/config.yaml 自动读取
    client = LLMClient()
    print(f"使用模型: {client.model}")

    print("=== 测试普通对话 ===")
    reply = client.chat(
        user_message="用一句话介绍你自己",
        system="你是一个影视制作助手",
    )
    print(f"回复: {reply}\n")

    print("=== 测试工具调用 ===")
    tools = [{
        "name": "get_weather",
        "description": "获取城市天气",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名"}
            },
            "required": ["city"]
        }
    }]
    _, tool_calls, stop_reason, _ = client.chat_with_tools(
        messages=[{"role": "user", "content": "北京今天天气怎么样？"}],
        tools=tools,
    )
    print(f"stop_reason: {stop_reason}")
    if tool_calls:
        for tc in tool_calls:
            print(f"  工具调用: {tc.function.name}({tc.function.arguments})")
