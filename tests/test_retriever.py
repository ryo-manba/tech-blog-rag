import pytest

from tech_blog_rag.chunker import Chunk
from tech_blog_rag.embedder import embed_chunks, get_client, store_chunks
from tech_blog_rag.retriever import search


@pytest.mark.api
def test_search_returns_results(tmp_chroma_dir):
    client = get_client()

    chunks = [
        Chunk(
            chunk_id="react-aria_chunk_0000",
            text="React Aria は Adobe が開発するヘッドレス UI ライブラリです。アクセシビリティを重視した設計になっています。",
            article_slug="react-aria",
            article_title="React Aria のコンポーネント設計",
            article_url="https://zenn.dev/ryo_manba/articles/react-aria",
            topics=["react", "reactaria"],
            chunk_index=0,
        ),
        Chunk(
            chunk_id="nextjs-bundle_chunk_0000",
            text="Next.js のバンドルサイズを分析するには @next/bundle-analyzer を使います。",
            article_slug="nextjs-bundle",
            article_title="Next.js バンドル分析",
            article_url="https://zenn.dev/ryo_manba/articles/nextjs-bundle",
            topics=["nextjs"],
            chunk_index=0,
        ),
    ]

    embeddings = embed_chunks(client, chunks, sleep_interval=1.0)
    store_chunks(chunks, embeddings, db_path=tmp_chroma_dir)

    results = search(client, "React Aria とは？", top_k=2, db_path=tmp_chroma_dir)
    assert len(results) > 0
    assert results[0].article_title == "React Aria のコンポーネント設計"
