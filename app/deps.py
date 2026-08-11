import os
import logging
from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_deepseek import ChatDeepSeek
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from zhipuai import ZhipuAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document

from .config import config


@lru_cache()
def get_llm():
    """获取 DeepSeek LLM 实例"""
    return  ChatDeepSeek(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        model="deepseek-chat",
        temperature=0.1,
    )

"""
ChatDeepSeek(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        model="deepseek-chat",
        temperature=0.1,
    )

ChatOpenAI(
        base_url="https://api.moonshot.cn/v1",  # 末尾别留空格
        api_key=os.getenv("MOONSHOT_API_KEY"),  # 显式再传一次更保险
        model="kimi-k2-0905-preview",
        temperature=0.1,
    )
"""


@lru_cache()
def get_neo4j_graph():
    """获取 Neo4j 图数据库连接"""
    return Neo4jGraph(
        url=config.NEO4J_URI,
        username=config.NEO4J_USERNAME,
        password=config.NEO4J_PASSWORD,
        refresh_schema=True,
    )


@lru_cache()
def get_graph_chain():
    """获取图数据库查询链"""
    from .agents.prompts import CYPHER_GENERATION_PROMPT

    llm = get_llm()
    graph = get_neo4j_graph()

    return GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        allow_dangerous_requests=True,
        verbose=True,
        cypher_llm_kwargs={
            "temperature": 0.1,
        },
        cypher_prompt_template=CYPHER_GENERATION_PROMPT
    )


def safe_graph_query(query_text: str) -> str:
    """安全的图数据库查询包装器"""
    try:
        chain = get_graph_chain()
        result = chain.invoke({"query": query_text})
        return result
    except Exception as e:
        error_msg = str(e)
        print(f"Graph query error: {error_msg}")

        # 如果是UNION错误，返回友好的错误信息
        if "UNION" in error_msg and "same return column names" in error_msg:
            return "抱歉，知识图谱查询遇到了技术问题。这通常是由于查询语法复杂导致的。请尝试更具体或更简单的问题，我会尽力使用我的医学知识来回答。"

        # 其他Neo4j错误
        elif "Neo4j" in error_msg or "Cypher" in error_msg:
            return "知识图谱暂时不可用，但我可以基于我的医学知识来回答您的问题。请告诉我您想了解什么医疗信息？"

        # 其他错误
        else:
            return f"查询过程中遇到问题：{error_msg}。让我尝试用其他方式回答您的问题。"


@lru_cache()
def get_graph_tool():
    """获取知识图谱查询工具"""
    return Tool(
        name="KnowledgeGraphQA",
        func=safe_graph_query,
        description="""使用知识图谱回答医疗相关问题。

适用场景：
- 查询疾病信息和症状
- 查询治疗方法和药物信息
- 查询医疗实体之间的关系
- 医学知识的结构化查询

使用建议：
- 问题要具体明确
- 一次只查询一个主要概念
- 避免过于复杂的组合查询"""
    )


@lru_cache()
def get_tavily_search_tool():
    """获取Tavily网络搜索工具"""
    if not config.TAVILY_API_KEY:
        # 如果没有配置API密钥，返回一个模拟工具
        return Tool(
            name="WebSearch",
            func=lambda x: {"results": [], "error": "Tavily API密钥未配置，无法使用网络搜索功能"},
            description="网络搜索工具（当前未配置API密钥）"
        )

    try:
        search = TavilySearchResults(
            api_key=config.TAVILY_API_KEY,
            max_results=5,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=False,
            include_domains=[],
            exclude_domains=[]
        )

        return Tool(
            name="WebSearch",
            func=search.run,
            description="""使用网络搜索获取最新的医疗健康信息。

适用场景：
- 查询最新的医学研究进展
- 搜索最新的药物信息和临床试验结果
- 获取最新的医疗指南和专家建议
- 查询特定疾病的最新治疗方法
- 搜索医疗新闻和健康资讯

使用建议：
- 搜索词要具体且与医疗健康相关
- 优先搜索权威医疗网站的内容
- 注意信息的时效性和可靠性
- 结合知识图谱信息进行综合判断"""
        )
    except Exception as e:
        print(f"Failed to initialize Tavily search tool: {e}")
        return Tool(
            name="WebSearch",
            func=lambda x: {"results": [], "error": f"网络搜索工具初始化失败: {str(e)}"},
            description="网络搜索工具（初始化失败）"
        )


class ZhipuEmbeddings:
    """智谱AI嵌入模型封装"""

    def __init__(self, api_key: str, model: str = "embedding-3"):
        self.client = ZhipuAI(api_key=api_key)
        self.model = model

    def embed_documents(self, texts: list) -> list:
        """批量嵌入文档"""
        embeddings = []
        for text in texts:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            embeddings.append(response.data[0].embedding)
        return embeddings

    def embed_query(self, text: str) -> list:
        """嵌入单个查询"""
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding


@lru_cache()
def get_embedding_model():
    """获取嵌入模型实例"""
    # 使用智谱AI的嵌入模型
    return ZhipuEmbeddings(
        api_key=config.ZHIPU_API_KEY,
        model=config.EMBEDDING_MODEL
    )


@lru_cache()
def get_vector_store():
    """获取向量数据库实例"""
    embeddings = get_embedding_model()

    # 确保向量数据库目录存在
    os.makedirs(config.VECTOR_DB_PATH, exist_ok=True)

    # 初始化Chroma向量数据库
    vector_store = Chroma(
        persist_directory=config.VECTOR_DB_PATH,
        embedding_function=embeddings
    )

    return vector_store


def reset_vector_store():
    """强制重新初始化向量数据库实例（用于清空后）"""
    logger = logging.getLogger(__name__)

    # 清除lru_cache缓存
    get_vector_store.cache_clear()

    logger.info("🔄 向量数据库实例缓存已清除，下次调用将重新初始化")

    # 返回新的实例
    return get_vector_store()


def get_text_splitter():
    """获取文本分割器"""
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )


def add_documents_to_vector_store(text_content: str, filename: str, file_path: str):
    """将文档添加到向量数据库"""
    import logging
    logger = logging.getLogger(__name__)

    try:
        logger.info(f"🧠 开始向量化处理: {filename}")
        logger.info(f"📄 文本长度: {len(text_content)} 字符")

        # 创建文档对象
        logger.info("📝 创建文档对象...")
        document = Document(
            page_content=text_content,
            metadata={
                "filename": filename,
                "file_path": file_path,
                "source": f"{filename} (uploaded file)"
            }
        )

        # 分割文档
        logger.info("✂️ 开始文档分块...")
        text_splitter = get_text_splitter()
        chunks = text_splitter.split_documents([document])

        logger.info(f"📦 文档分块完成: {len(chunks)} 个块")

        # 添加到向量数据库
        logger.info("💾 添加到向量数据库...")
        vector_store = get_vector_store()
        vector_store.add_documents(chunks)

        # 持久化
        logger.info("💿 持久化向量数据库...")
        vector_store.persist()

        result = {
            "success": True,
            "chunks_count": len(chunks),
            "message": f"成功添加 {len(chunks)} 个文档块到向量数据库"
        }

        logger.info(f"✅ 向量化完成: {result}")
        return result

    except Exception as e:
        logger.error(f"❌ 向量化失败: {str(e)}")
        import traceback
        logger.error(f"📋 错误详情: {traceback.format_exc()}")
        return {
            "success": False,
            "error": str(e),
            "message": "文档向量化失败"
        }


def search_relevant_documents(query: str, k: int = None):
    """搜索相关文档"""
    try:
        if k is None:
            k = config.MAX_RETRIEVED_DOCS

        vector_store = get_vector_store()

        # 搜索相似文档
        results = vector_store.similarity_search_with_score(
            query,
            k=k
        )

        return {
            "success": True,
            "documents": results,
            "count": len(results)
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "documents": []
        }