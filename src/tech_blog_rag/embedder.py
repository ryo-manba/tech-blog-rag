"""チャンクを Gemini Embedding API でベクトル化し、ChromaDB に格納する。"""
import time

import chromadb
from google import genai

from tech_blog_rag.chunker import Chunk
from tech_blog_rag.config import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    GEMINI_API_KEY,
)


def get_client() -> genai.Client:
    return genai.Client(api_key=GEMINI_API_KEY)


def embed_text(
    client: genai.Client, text: str, task_type: str = "RETRIEVAL_DOCUMENT"
) -> list[float]:
    """Gemini Embedding API で 1 テキストをベクトル化。"""
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config={"task_type": task_type},
    )
    return result.embeddings[0].values


def embed_chunks(
    client: genai.Client,
    chunks: list[Chunk],
    batch_size: int = 100,
    sleep_interval: float = 0.5,
) -> list[list[float]]:
    """複数チャンクをバッチで Embedding。Rate Limit 対策でバッチ間に sleep を挟む。"""
    embeddings: list[list[float]] = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.text for c in batch]
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts,
            config={"task_type": "RETRIEVAL_DOCUMENT"},
        )
        embeddings.extend([e.values for e in result.embeddings])
        if i + batch_size < len(chunks):
            time.sleep(sleep_interval)
    return embeddings


def store_chunks(
    chunks: list[Chunk],
    embeddings: list[list[float]],
    db_path: str = CHROMA_DB_PATH,
) -> int:
    """ChromaDB に格納。既存コレクションは削除して再作成。戻り値は格納件数。"""
    client = chromadb.PersistentClient(path=db_path)

    # 既存コレクションがあれば削除
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [c.chunk_id for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [
        {
            "article_title": c.article_title,
            "article_url": c.article_url,
            "article_slug": c.article_slug,
            "topics": ",".join(c.topics),
            "chunk_index": c.chunk_index,
        }
        for c in chunks
    ]

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return collection.count()
