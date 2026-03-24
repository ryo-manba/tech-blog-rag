from tech_blog_rag.collector import load_articles
from tech_blog_rag.config import ZENN_BASE_URL


def test_load_articles_returns_published_only(sample_md_dir):
    articles = load_articles(sample_md_dir)
    assert len(articles) == 2


def test_article_has_correct_fields(sample_md_dir):
    articles = load_articles(sample_md_dir)
    article = next(a for a in articles if a.slug == "test-article-1")
    assert article.title == "拡張性に優れた React Aria のコンポーネント設計"
    assert article.topics == ["react", "reactaria", "フロントエンド"]
    assert article.url == f"{ZENN_BASE_URL}/test-article-1"
    assert "React Aria" in article.body
    assert article.published is True


def test_draft_articles_are_skipped(sample_md_dir):
    articles = load_articles(sample_md_dir)
    slugs = [a.slug for a in articles]
    assert "draft-article" not in slugs


def test_empty_directory(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    articles = load_articles(str(empty_dir))
    assert articles == []
