"""zenn-content/articles/*.md を読み込み、Article のリストを返す。"""
import glob
import logging
import os
from dataclasses import dataclass

import frontmatter

from tech_blog_rag.config import ZENN_BASE_URL

logger = logging.getLogger(__name__)


@dataclass
class Article:
    slug: str
    title: str
    url: str
    topics: list[str]
    body: str
    published: bool


def load_articles(articles_dir: str) -> list[Article]:
    """articles_dir/*.md を読み込み、published=true の Article リストを返す。"""
    articles: list[Article] = []
    md_files = sorted(glob.glob(os.path.join(articles_dir, "*.md")))

    for filepath in md_files:
        try:
            post = frontmatter.load(filepath)
        except Exception:
            logger.warning("frontmatter パース失敗: %s", filepath)
            continue

        published = post.metadata.get("published", False)
        if not published:
            continue

        slug = os.path.splitext(os.path.basename(filepath))[0]
        title = post.metadata.get("title", "") or slug
        topics = post.metadata.get("topics", [])
        url = f"{ZENN_BASE_URL}/{slug}"

        articles.append(
            Article(
                slug=slug,
                title=title,
                url=url,
                topics=topics,
                body=post.content,
                published=published,
            )
        )

    return articles
