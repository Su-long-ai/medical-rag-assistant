"""
智能医疗Agent - 优化版本，减少token消耗
"""
from typing import Optional, Dict, Any
from langchain.agents import AgentExecutor, AgentType, initialize_agent
from langchain.schema import HumanMessage, AIMessage
from .prompts import MEDICAL_AGENT_SYSTEM_PROMPT
from ..deps import get_llm, get_graph_tool, get_tavily_search_tool


class SmartMedicalAgent:
    """智能医疗Agent - 优化版本"""

    def __init__(self):
        self.llm = get_llm()
        self.graph_tool = get_graph_tool()
        self.search_tool = get_tavily_search_tool()
        self._base_agent: Optional[AgentExecutor] = None

    def get_base_agent(self) -> AgentExecutor:
        """获取基础Agent（优化版本）"""
        if self._base_agent is None:
            # 创建Agent工具列表
            tools = [self.graph_tool, self.search_tool]

            # 创建优化的Agent - 使用更高效的类型
            agent = initialize_agent(
                tools=tools,
                llm=self.llm,
                agent_type=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                verbose=False,  # 减少日志输出
                handle_parsing_errors=True,
                max_iterations=2,  # 进一步减少迭代次数
                early_stopping_method="generate",
                agent_kwargs={
                    "system_message": self._get_optimized_prompt()
                }
            )

            # 包装为AgentExecutor
            if hasattr(agent, 'agent') and hasattr(agent, 'tools'):
                self._base_agent = AgentExecutor(
                    agent=agent.agent,
                    tools=agent.tools,
                    verbose=False,  # 减少日志输出
                    handle_parsing_errors=True,
                    max_iterations=2,  # 减少迭代次数
                    early_stopping_method="generate"
                )
            else:
                self._base_agent = agent

        return self._base_agent

    def _get_optimized_prompt(self) -> str:
        """获取优化的系统提示词"""
        return f"""
{MEDICAL_AGENT_SYSTEM_PROMPT}

### ⚡ 重要优化指令 ###
1. **优先使用文档内容**：如果提供了文档，直接基于文档回答，不要使用工具
2. **减少工具调用**：仅在文档信息不足时才使用工具
3. **避免重复搜索**：不要进行相似的网络搜索
4. **简洁回答**：直接给出答案，避免冗长的解释

### 智能决策流程：
- 有文档相关内容？→ 直接回答，不使用工具
- 文档内容不足？→ 使用知识图谱工具
- 需要最新信息？→ 使用网络搜索工具
- 都不需要？→ 直接基于专业知识回答

**记住：减少不必要的工具调用，节省token消耗！**
"""

    def invoke_with_fallback(self, input_text: str, callback_handler=None) -> Dict[str, Any]:
        """
        智能执行Agent，带优化的降级处理
        """
        try:
            # 智能决策：分析问题类型和文档相关性
            decision = self._analyze_question_and_decide(input_text)

            if decision["use_direct_llm"]:
                # 使用直接LLM回答
                return self._smart_llm_response(input_text, decision)
            else:
                # 使用Agent工具
                agent = self.get_base_agent()
                config = {"callbacks": [callback_handler]} if callback_handler else {}
                return agent.invoke({"input": input_text}, config=config)

        except Exception as e:
            print(f"Smart Agent execution failed, falling back to LLM: {e}")
            # 保存原始决策信息，不要丢失
            return self._smart_llm_response(input_text, decision)

    def _analyze_question_and_decide(self, input_text: str) -> Dict[str, Any]:
        """
        智能分析问题类型并决定处理策略
        """
        # 提取用户问题（去除RAG检索到的文档内容）
        user_question = self._extract_user_question(input_text)

        # 检查是否有相关文档内容
        has_docs = self._has_document_context(input_text)

        # 分析问题类型关键词 - 调整优先级
        search_keywords = [
            "最新", "新闻", "研究", "进展", "新药", "临床试验",
            "最新治疗", "新方法", "突破", "近期", "今年",
            # 时间相关关键词
            "今天", "现在", "当前", "目前", "几月几号", "几点",
            "日期", "时间", "上网搜索", "搜索一下", "百度",
            "谷歌", "搜一下", "网上查", "网络搜索"
        ]

        knowledge_graph_keywords = [
            "查询", "知识图谱", "定义", "什么是", "原理", "机制",
            "分类", "症状", "病因", "诊断标准", "检测方法",
            "治疗方法", "预防措施", "流行病学"
        ]

        document_specific_keywords = [
            "文档中", "资料中", "根据文档", "文件中说",
            "报告中", "根据你提供", "文档提到"
        ]

        # 判断问题类型 - 调整判断顺序
        is_search_question = any(keyword in user_question for keyword in search_keywords)
        is_knowledge_question = any(keyword in user_question for keyword in knowledge_graph_keywords)
        is_document_specific = any(keyword in user_question for keyword in document_specific_keywords)

        # 决策逻辑 - 调整优先级，文档特定问题优先级最高
        if is_document_specific:
            # 用户明确要求基于文档回答（优先级最高）
            return {
                "use_direct_llm": True,
                "question_type": "document_specific",
                "reasoning": "用户明确要求基于文档回答"
            }
        elif is_search_question:
            # 需要最新信息的搜索类问题（优先级高）
            return {
                "use_direct_llm": False,
                "question_type": "search_query",
                "reasoning": "需要获取最新医疗信息"
            }
        elif is_knowledge_question:
            # 知识图谱查询类问题
            return {
                "use_direct_llm": False,
                "question_type": "knowledge_query",
                "reasoning": "需要查询知识图谱获取权威医学信息"
            }
        elif has_docs and not (is_knowledge_question or is_search_question):
            # 有文档且问题不涉及需要外部工具的查询
            return {
                "use_direct_llm": True,
                "question_type": "document_general",
                "reasoning": "有相关文档内容，且问题不涉及外部知识查询"
            }
        else:
            # 默认情况：如果有文档就用文档，没有就用Agent
            return {
                "use_direct_llm": has_docs,
                "question_type": "general",
                "reasoning": f"有文档内容: {has_docs}"
            }

    def _extract_user_question(self, input_text: str) -> str:
        """
        从输入中提取用户原始问题（去除RAG检索的文档内容）
        """
        # 对于聊天服务，input_text就是用户的问题，不需要提取
        # 这个函数主要用于处理Agent输入中可能包含的文档内容
        if input_text and len(input_text) < 500:  # 假设用户问题不会太长
            return input_text.strip()

        # 如果输入很长，尝试提取问题部分
        lines = input_text.split('\n')
        question_lines = []

        for line in lines:
            line = line.strip()
            # 跳过文档内容行
            if any(indicator in line for indicator in [
                "相关文档内容：", "来源:", "uploaded file",
                "文件内容", "基于以下相关文档", "文档片段"
            ]):
                continue
            # 跳过看起来像文档内容的行（通常较长且包含具体信息）
            if len(line) > 200 and any(char in line for char in ['。', '，', '、']):
                continue
            # 跳过文件名行
            if '.' in line and any(line.endswith(ext) for ext in ['.pdf', '.docx', '.txt']):
                continue

            question_lines.append(line)

        # 返回前面的部分作为用户问题
        return '\n'.join(question_lines[:3]) if question_lines else input_text

    def _has_document_context(self, input_text: str) -> bool:
        """检查输入是否包含文档上下文"""
        document_indicators = [
            "相关文档内容：",
            "基于以下相关文档",
            "来源:",
            "uploaded file",
            "文件内容"
        ]
        return any(indicator in input_text for indicator in document_indicators)

    def _smart_llm_response(self, input_text: str, decision: Dict[str, Any] = None) -> Dict[str, Any]:
        """智能LLM响应 - 降级处理"""
        if decision is None:
            decision = {"question_type": "unknown"}

        question_type = decision.get("question_type", "unknown")

        # 根据问题类型构建不同的提示
        if question_type in ["document_specific", "document_general"]:
            # 基于文档回答
            smart_prompt = f"""你是一个专业的医疗AI助手。请基于以下提供的文档内容准确回答问题。

{input_text}

回答要求：
1. 优先使用文档中的信息回答
2. 如果文档信息完整，不要进行额外搜索
3. 提供专业、准确的医疗信息
4. 如果文档中没有相关信息，诚实地说明，然后基于你的医疗知识补充回答"""
        elif question_type == "knowledge_query":
            # 知识查询类问题，即使没有工具也要基于专业知识回答
            smart_prompt = f"""你是一个专业的医疗AI助手。用户询问的是医学知识查询类问题。请基于你的专业知识回答。

{input_text}

请提供权威、准确的医学知识回答。如果涉及具体疾病的定义、症状、诊断方法等，请给出详细的专业解释。"""
        elif question_type == "search_query":
            # 需要最新信息的搜索类问题
            # 检查输入是否包含搜索结果
            if "根据网络搜索结果" in input_text or "搜索结果" in input_text:
                smart_prompt = f"""你是一个专业的医疗AI助手。用户询问了需要最新信息的问题，系统已经进行了网络搜索。请基于以下搜索结果准确回答用户问题。

{input_text}

请基于搜索结果提供准确、有用的信息。如果搜索结果不完整或需要更多专业解释，可以基于你的知识进行补充。"""
            else:
                smart_prompt = f"""你是一个专业的医疗AI助手。用户询问的是需要最新信息的问题。请基于你的知识库回答。

{input_text}

请注意：由于无法进行实时网络搜索，我的信息可能不是最新的。对于需要最新治疗指南、研究进展的问题，建议用户咨询专业医疗机构或查阅最新文献。"""
        else:
            # 通用问题
            smart_prompt = f"""你是一个专业的医疗AI助手。请基于你的医学知识回答以下问题。

{input_text}

请提供专业、准确的医疗信息。"""

        try:
            llm_response = self.llm.invoke(smart_prompt)
            content = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)

            # 添加决策信息日志（用于调试）
            print(f"Decision: {decision.get('reasoning', 'unknown')}")

            return {"output": content}
        except Exception as e:
            print(f"Smart LLM response failed: {e}")
            # 最后的降级处理
            return {"output": "抱歉，我遇到了技术问题。请重新提问或咨询专业医疗人员。"}

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