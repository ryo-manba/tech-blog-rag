"""定数・設定"""
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
EMBEDDING_MODEL = "gemini-embedding-001"
GENERATION_MODEL = "gemini-2.5-flash"
CHROMA_DB_PATH = "data/chroma_db"
COLLECTION_NAME = "tech_blog_articles"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 5
CONTENT_DIR = os.environ.get("CONTENT_DIR", "../zenn-content")
ARTICLES_DIR = os.path.join(CONTENT_DIR, "articles")
ZENN_BASE_URL = "https://zenn.dev/ryo_manba/articles"
