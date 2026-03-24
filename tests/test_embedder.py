import pytest

from tech_blog_rag.chunker import Chunk
from tech_blog_rag.embedder import embed_text, get_client, store_chunks


@pytest.mark.api
def test_embed_text_returns_vector():
    client = get_client()
    vec = embed_text(client, "React Aria のコンポーネント設計")
    assert isinstance(vec, list)
    assert len(vec) > 0
    assert all(isinstance(v, float) for v in vec)


def test_store_and_count(tmp_chroma_dir):
    chunks = [
        Chunk(
            chunk_id="test_chunk_0000",
            text="テスト記事の本文です。",
            article_slug="test-article",
            article_title="テスト記事",
            article_url="https://zenn.dev/ryo_manba/articles/test-article",
            topics=["react"],
            chunk_index=0,
        ),
        Chunk(
            chunk_id="test_chunk_0001",
            text="もう一つのチャンクです。",
            article_slug="test-article",
            article_title="テスト記事",
            article_url="https://zenn.dev/ryo_manba/articles/test-article",
            topics=["react"],
            chunk_index=1,
        ),
    ]
    # ダミー Embedding（768 次元）
    embeddings = [[0.1] * 768, [0.2] * 768]
    count = store_chunks(chunks, embeddings, db_path=tmp_chroma_dir)
    assert count == 2
