import os

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_csv(name: str, default: str = "") -> tuple[str, ...]:
    value = os.getenv(name, default)
    return tuple(item.strip() for item in value.split(",") if item.strip())


class Config:
    """Environment-backed application configuration.

    Secrets are intentionally read only from the environment. Public API
    responses must never echo credential values or internal filesystem paths.
    """

    # LLM provider
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek").strip().lower()
    LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat").strip()
    LLM_API_KEY = os.getenv("LLM_API_KEY")
    LLM_BASE_URL = os.getenv("LLM_BASE_URL")
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

    # Embeddings
    ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embedding-3").strip()

    # Optional tools
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    ENABLE_WEB_SEARCH = _env_bool("ENABLE_WEB_SEARCH", False)
    ENABLE_GRAPH_TOOL = _env_bool("ENABLE_GRAPH_TOOL", False)
    ENABLE_ADMIN_ENDPOINTS = _env_bool("ENABLE_ADMIN_ENDPOINTS", False)

    # Neo4j (only used when ENABLE_GRAPH_TOOL=true)
    NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

    # Upload and retrieval policy
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_BYTES", str(10 * 1024 * 1024)))
    MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "50"))
    ALLOWED_EXTENSIONS = frozenset({".pdf", ".docx", ".txt"})
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "vector_db")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
    MAX_RETRIEVED_DOCS = int(os.getenv("MAX_RETRIEVED_DOCS", "5"))
    MAX_CONTEXT_CHARS_PER_CHUNK = int(os.getenv("MAX_CONTEXT_CHARS_PER_CHUNK", "800"))
    OCR_LANGUAGES = os.getenv("OCR_LANGUAGES", "chi_sim+eng")

    # Browser clients allowed to call the API in development.
    CORS_ORIGINS = _env_csv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://localhost:3000",
    )

    @classmethod
    def llm_configured(cls) -> bool:
        if cls.LLM_PROVIDER == "deepseek":
            return bool(cls.DEEPSEEK_API_KEY or cls.LLM_API_KEY)
        if cls.LLM_PROVIDER == "openai_compatible":
            return bool(cls.LLM_API_KEY and cls.LLM_BASE_URL)
        return False

    @classmethod
    def embeddings_configured(cls) -> bool:
        return bool(cls.ZHIPU_API_KEY)


config = Config()
