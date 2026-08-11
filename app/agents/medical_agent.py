"""
医疗Agent核心逻辑
"""
from typing import Optional
from langchain.agents import AgentExecutor, AgentType, initialize_agent
from .prompts import MEDICAL_AGENT_SYSTEM_PROMPT
from ..deps import get_llm, get_graph_tool, get_tavily_search_tool


class MedicalAgent:
    """医疗Agent"""

    def __init__(self):
        self.llm = get_llm()
        self.graph_tool = get_graph_tool()
        self.search_tool = get_tavily_search_tool()
        self._base_agent: Optional[AgentExecutor] = None

    def get_base_agent(self) -> AgentExecutor:
        """获取基础Agent（无记忆）"""
        if self._base_agent is None:
            # 创建Agent工具列表
            tools = [self.graph_tool, self.search_tool]

            # 创建Agent - 使用更稳定的类型
            agent = initialize_agent(
                tools=tools,
                llm=self.llm,
                agent_type=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=3,  # 减少迭代次数，避免无限循环
                early_stopping_method="generate",
                agent_kwargs={
                    "system_message": MEDICAL_AGENT_SYSTEM_PROMPT
                }
            )

            # 包装为AgentExecutor
            if hasattr(agent, 'agent') and hasattr(agent, 'tools'):
                self._base_agent = AgentExecutor(
                    agent=agent.agent,
                    tools=agent.tools,
                    verbose=True,
                    handle_parsing_errors=True,
                    max_iterations=5,  # 增加迭代次数以支持更复杂的搜索
                    early_stopping_method="generate"
                )
            else:
                self._base_agent = agent

        return self._base_agent

    def invoke_with_fallback(self, input_text: str, callback_handler=None) -> dict:
        """
        执行Agent，带降级处理

        Args:
            input_text: 输入文本
            callback_handler: 回调处理器

        Returns:
            包含output的字典
        """
        try:
            agent = self.get_base_agent()
            config = {"callbacks": [callback_handler]} if callback_handler else {}
            return agent.invoke({"input": input_text}, config=config)
        except Exception as e:
            print(f"Agent execution failed, falling back to LLM: {e}")
            return self._fallback_to_llm(input_text)

    def _fallback_to_llm(self, input_text: str) -> dict:
        """降级到直接使用LLM"""
        fallback_prompt = f"""你是一个专业的医疗AI助手。请基于你的医学知识回答以下问题。

{input_text}

请提供专业、准确的医疗信息。"""

        llm_response = self.llm.invoke(fallback_prompt)
        content = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
        return {"output": content}

    @staticmethod
    def extract_response_text(response) -> str:
        """从agent响应中提取文本内容"""
        if isinstance(response, str):
            return response
        elif isinstance(response, dict):
            # 尝试多种可能的键
            for key in ['output', 'result', 'answer']:
                if key in response:
                    return str(response[key])

            # 查找第一个长文本值
            for value in response.values():
                if isinstance(value, str) and len(value) > 10:
                    return value

            return str(response)
        else:
            return str(response)