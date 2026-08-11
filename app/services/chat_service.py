"""
聊天服务 - 集成RAG功能
"""
from typing import AsyncGenerator, Dict, Any
import asyncio
import concurrent.futures

from ..agents import MedicalAgent, StreamingCallbackHandler
from ..agents.smart_medical_agent import SmartMedicalAgent
from .session_service import SessionManager
from ..deps import search_relevant_documents


class MedicalChatService:
    """医疗聊天服务"""

    def __init__(self):
        self.agent = SmartMedicalAgent()  # 使用优化版本
        self.session_manager = SessionManager()

    @property
    def sessions(self):
        """为了向后兼容，返回session_manager的store"""
        return self.session_manager.store

    def _build_agent_input(self, message: str, session_id: str) -> str:
        """构建Agent输入，智能集成RAG检索"""
        chat_history_str = self.session_manager.format_chat_history(session_id)

        # 让Agent先决定是否需要RAG检索
        decision = self.agent._analyze_question_and_decide(message)
        print(f"Agent决策: {decision['reasoning']} (类型: {decision['question_type']})")

        # 根据决策结果决定是否进行RAG检索
        rag_context = ""
        if decision["question_type"] in ["document_specific", "document_general"]:
            # 只有在Agent认为需要文档时才进行RAG检索
            try:
                search_result = search_relevant_documents(message)
                if search_result["success"] and search_result["documents"]:
                    rag_context = "\n\n相关文档内容：\n"
                    for doc, score in search_result["documents"]:
                        rag_context += f"- [来源: {doc.metadata.get('filename', '未知文件')}]: {doc.page_content[:300]}...\n"
            except Exception as e:
                print(f"RAG检索失败: {e}")

        # 构建输入，根据决策类型调整提示
        if decision["question_type"] in ["document_specific", "document_general"] and rag_context:
            # 文档相关问题
            if chat_history_str:
                return f"""【文档相关问题】基于以下文档内容和对话历史回答问题：

相关文档内容：
{rag_context}

对话历史：
{chat_history_str}

当前问题：{message}

请基于文档内容准确回答问题。"""
            else:
                return f"""【文档相关问题】基于以下文档内容回答问题：

相关文档内容：
{rag_context}

当前问题：{message}

请基于文档内容准确回答问题。"""
        elif decision["question_type"] in ["knowledge_query", "search_query"]:
            # 需要工具的问题，不提供文档内容
            if chat_history_str:
                return f"""【工具查询问题】基于对话历史，使用工具查询准确信息：

对话历史：
{chat_history_str}

当前问题：{message}

请使用医疗知识图谱或网络搜索工具获取权威信息，不要依赖可能不相关的文档内容。"""
            else:
                return f"""【工具查询问题】请使用工具查询权威信息：

当前问题：{message}

请使用医疗知识图谱或网络搜索工具获取准确信息。"""
        else:
            # 一般问题
            if chat_history_str:
                return f"""【一般咨询】基于对话历史回答问题：

对话历史：
{chat_history_str}

当前问题：{message}

请提供专业的医疗建议。如需要权威信息，可使用工具查询。"""
            else:
                return f"""【一般咨询】回答问题：

当前问题：{message}

请提供专业的医疗建议。"""

    async def chat_stream(self, message: str, session_id: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        流式聊天响应

        Args:
            message: 用户消息
            session_id: 会话ID

        Yields:
            流式响应块
        """
        try:
            # 构建输入
            full_input = self._build_agent_input(message, session_id)

            # 创建回调处理器
            callback_handler = StreamingCallbackHandler()

            # 发送思考状态
            yield {
                "type": "status",
                "content": "🔍 正在检索相关文档..."
            }

            # 在线程池中执行Agent
            def run_agent():
                return self.agent.invoke_with_fallback(full_input, callback_handler)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_agent)

                # 定期检查进度并发送状态更新
                while not future.done():
                    if callback_handler.current_step:
                        yield {
                            "type": "status",
                            "content": callback_handler.current_step
                        }
                        callback_handler.current_step = ""

                    await asyncio.sleep(0.1)

                # 获取最终结果
                response = future.result()

            # 提取响应文本
            response_text = self.agent.extract_response_text(response)

            # 保存到聊天历史
            self.session_manager.add_exchange(session_id, message, response_text)

            # 调试信息
            print(f"Response: {response_text[:100]}...")
            print(f"Chat history length: {len(self.session_manager.get_history_list(session_id))}")

            # 流式发送最终响应
            if callback_handler.tokens:
                # 如果有token级别的流式输出
                for token in callback_handler.tokens:
                    yield {
                        "type": "token",
                        "content": token
                    }
            else:
                # 分块发送响应
                chunk_size = 5
                for i in range(0, len(response_text), chunk_size):
                    chunk = response_text[i:i + chunk_size]
                    yield {
                        "type": "chunk",
                        "content": chunk
                    }
                    await asyncio.sleep(0.1)

            # 发送完成信号
            yield {
                "type": "done",
                "content": ""
            }

        except Exception as e:
            print(f"Chat service error: {e}")
            import traceback
            traceback.print_exc()

            yield {
                "type": "error",
                "content": f"处理消息时出错: {str(e)}"
            }

    def get_chat_history(self, session_id: str) -> list:
        """获取聊天历史"""
        return self.session_manager.get_history_list(session_id)

    def clear_session(self, session_id: str) -> bool:
        """清除会话记忆"""
        return self.session_manager.clear_session(session_id)