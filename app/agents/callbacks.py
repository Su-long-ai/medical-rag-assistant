"""
Agent回调处理器
"""
from typing import Dict, Any
from langchain_core.callbacks import BaseCallbackHandler


class StreamingCallbackHandler(BaseCallbackHandler):
    """流式输出回调处理器"""

    def __init__(self):
        self.tokens = []
        self.current_step = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """处理新生成的token"""
        self.tokens.append(token)

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        """工具开始执行"""
        self.current_step = f"🔍 正在查询知识图谱: {input_str[:50]}..."

    def on_tool_end(self, output: str, **kwargs) -> None:
        """工具执行结束"""
        self.current_step = "💭 正在整理答案..."

    def on_agent_action(self, action, **kwargs) -> None:
        """Agent执行动作"""
        self.current_step = f"🤖 {action.tool}: {action.tool_input[:50]}..."