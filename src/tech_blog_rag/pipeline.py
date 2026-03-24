"""記事取り込み・質問応答のパイプライン。"""
import logging

import chromadb

from tech_blog_rag.chunker import chunk_articles
from tech_blog_rag.collector import load_articles
from tech_blog_rag.config import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    CONTENT_DIR,
    TOP_K,
)
from tech_blog_rag.embedder import embed_chunks, get_client, store_chunks
from tech_blog_rag.generator import Answer, generate
from tech_blog_rag.retriever import search

logger = logging.getLogger(__name__)


def ingest(
    content_dir: str = CONTENT_DIR, db_path: str = CHROMA_DB_PATH
) -> dict:
    """記事取り込みの全フロー。"""
    import os

    articles_dir = os.path.join(content_dir, "articles")
    articles = load_articles(articles_dir)
    logger.info("記事数: %d", len(articles))

    chunks = chunk_articles(articles)
    logger.info("チャンク数: %d", len(chunks))

    client = get_client()
    embeddings = embed_chunks(client, chunks)
    stored = store_chunks(chunks, embeddings, db_path=db_path)
    logger.info("格納数: %d", stored)

    return {"articles": len(articles), "chunks": len(chunks), "stored": stored}


def query(
    question: str, top_k: int = TOP_K, db_path: str = CHROMA_DB_PATH
) -> Answer:
    """質問→検索→回答の全フロー。"""
    client = get_client()
    results = search(client, question, top_k=top_k, db_path=db_path)
    answer = generate(client, question, results)
    return answer


def get_status(db_path: str = CHROMA_DB_PATH) -> dict:
    """ChromaDB のステータスを返す。"""
    try:
        chroma_client = chromadb.PersistentClient(path=db_path)
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
        return {
            "collection_name": COLLECTION_NAME,
            "chunk_count": collection.count(),
        }
    except Exception:
        return {"collection_name": COLLECTION_NAME, "chunk_count": 0}
