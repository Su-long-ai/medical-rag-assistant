import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # DeepSeek API
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

    # Moonshot API (当前使用的)
    MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY")

    # 智谱AI API
    ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")

    # Tavily搜索API
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

    # Neo4j 配置
    NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    # Never keep credentials in source code. Set this in .env locally.
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

    # 文件上传配置
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.txt'}
    UPLOAD_DIR = "uploads"

    # RAG配置
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embedding-3")
    VECTOR_DB_PATH = "vector_db"
    CHUNK_SIZE = 1000  # 文本分块大小
    CHUNK_OVERLAP = 200  # 分块重叠
    MAX_RETRIEVED_DOCS = 5  # 检索的最大文档数


config = Config()
