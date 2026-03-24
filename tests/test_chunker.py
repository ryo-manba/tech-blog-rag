from tech_blog_rag.chunker import chunk_article, chunk_articles
from tech_blog_rag.collector import Article, load_articles


def _make_article(body: str, slug: str = "test-slug") -> Article:
    return Article(
        slug=slug,
        title="Test Title",
        url=f"https://zenn.dev/ryo_manba/articles/{slug}",
        topics=["react"],
        body=body,
        published=True,
    )


def test_short_article_becomes_one_chunk():
    article = _make_article("短い記事です。")
    chunks = chunk_article(article)
    assert len(chunks) == 1


def test_chunk_id_format():
    article = _make_article("短い記事です。")
    chunks = chunk_article(article)
    assert chunks[0].chunk_id == "test-slug_chunk_0000"


def test_chunk_preserves_metadata():
    article = _make_article("テスト本文", slug="my-article")
    chunks = chunk_article(article)
    c = chunks[0]
    assert c.article_slug == "my-article"
    assert c.article_title == "Test Title"
    assert c.topics == ["react"]
    assert c.chunk_index == 0


def test_code_block_not_split():
    body = "前文です。\n\n" + "あ" * 300 + "\n\n```python\nprint('hello')\nprint('world')\n```\n\n後文です。"
    article = _make_article(body)
    chunks = chunk_article(article, chunk_size=200, overlap=50)
    for chunk in chunks:
        count = chunk.text.count("```")
        assert count % 2 == 0, f"コードブロックが不完全: {chunk.text[:100]}"


def test_chunk_articles_processes_all(sample_md_dir):
    articles = load_articles(sample_md_dir)
    chunks = chunk_articles(articles)
    slugs = {c.article_slug for c in chunks}
    assert len(slugs) == 2  # published 記事は 2 つ


def test_overlap_exists():
    # 長い記事で複数チャンクを生成し、オーバーラップを確認
    paragraphs = [f"段落{i}。" + "あ" * 200 for i in range(10)]
    body = "\n\n".join(paragraphs)
    article = _make_article(body)
    chunks = chunk_article(article, chunk_size=300, overlap=100)
    assert len(chunks) >= 2
    # 隣接チャンク間でテキストの重複がある
    for i in range(len(chunks) - 1):
        tail = chunks[i].text[-50:]
        assert tail in chunks[i + 1].text, "隣接チャンクにオーバーラップがない"
