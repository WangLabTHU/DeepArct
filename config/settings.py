"""
config/settings.py
系统配置管理
"""

import os
from typing import Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    """系统设置"""
    # LLM配置
    llm_model: str = "gpt-5.1"
    code_model: str = "gpt-5.1"
    openai_api_key: Optional[str] = None
    
    # RAG配置
    rag_top_k: int = 5  # 减少到5，降低token使用
    rag_similarity_threshold: float = 0.3
    rag_default_strategy: str = "hybrid"
    
    # 向量数据库配置
    vector_db_type: str = "chromadb"
    chroma_persist_directory: str = "./RAG/chroma_db"
    
    # Embedding模型
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # 知识源配置
    knowledge_sources_config: Optional[str] = None
    
    # 工作流配置
    max_iterations: int = 8
    default_temperature: float = 0.6
    code_generator_temperature: float = 0.3  # 代码生成专用，更聚焦

    # 消融实验：是否启用RAG（False=仅智能体讨论，无知识库检索）
    use_rag: bool = True



def _load_env_files() -> None:
    """
    主动从项目根目录和Agents目录加载 .env 文件到环境变量中
    （仅在变量尚未设置时写入，避免覆盖外部配置）。
    """
    root_dir = Path(__file__).resolve().parent.parent
    candidate_paths = [
        root_dir / ".env",
        root_dir / "Agents" / ".env",
    ]

    for env_path in candidate_paths:
        if not env_path.exists():
            continue
        try:
            with env_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip("\"'")  # 去掉可能的引号
                    if key and key not in os.environ:
                        os.environ[key] = value
        except Exception as e:
            # 加载.env失败时仅打印警告，不中断程序
            print(f"⚠️ 加载环境文件失败: {env_path} ({e})")


# 模块加载时先尝试加载 .env
_load_env_files()


# 全局设置实例
_settings = None

def get_settings() -> Settings:
    """获取设置单例"""
    global _settings
    if _settings is None:
        _settings = Settings()
        # 从环境变量加载配置（此时 .env 已尝试加载）
        _settings.openai_api_key = os.getenv("OPENAI_API_KEY")
        # USE_RAG: "false"/"0" 表示消融实验（仅智能体讨论，无RAG）
        use_rag_env = os.getenv("USE_RAG", "").lower()
        if use_rag_env in ("false", "0", "no", "off"):
            _settings.use_rag = False
    return _settings

