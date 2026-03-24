"""ユーザーの質問を Embedding し、ChromaDB から類似チャンクを検索する。"""
from dataclasses import dataclass

import chromadb
from google import genai

from tech_blog_rag.config import CHROMA_DB_PATH, COLLECTION_NAME, TOP_K
from tech_blog_rag.embedder import embed_text


@dataclass
class SearchResult:
    chunk_text: str
    article_title: str
    article_url: str
    topics: list[str]
    distance: float


def search(
    client: genai.Client,
    query: str,
    top_k: int = TOP_K,
    db_path: str = CHROMA_DB_PATH,
) -> list[SearchResult]:
    """query を Embedding し、ChromaDB でコサイン類似度検索する。"""
    query_vec = embed_text(client, query, task_type="RETRIEVAL_QUERY")

    chroma_client = chromadb.PersistentClient(path=db_path)
    try:
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
    except chromadb.errors.NotFoundError:
        print("エラー: データが未取り込みです。先に ingest を実行してください:")
        print("  uv run python scripts/ingest.py --content-dir ../zenn-content")
        return []

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=min(top_k, collection.count()),
    )

    search_results: list[SearchResult] = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        search_results.append(
            SearchResult(
                chunk_text=results["documents"][0][i],
                article_title=meta["article_title"],
                article_url=meta["article_url"],
                topics=meta["topics"].split(",") if meta["topics"] else [],
                distance=results["distances"][0][i],
            )
        )

    return search_results
