"""記事本文を固定サイズ + オーバーラップでチャンク分割する。"""
import re
from dataclasses import dataclass

from tech_blog_rag.collector import Article
from tech_blog_rag.config import CHUNK_OVERLAP, CHUNK_SIZE


@dataclass
class Chunk:
    chunk_id: str
    text: str
    article_slug: str
    article_title: str
    article_url: str
    topics: list[str]
    chunk_index: int


def _protect_code_blocks(text: str) -> tuple[str, dict[str, str]]:
    """コードブロック（```...```）をプレースホルダに置換して保護する。"""
    replacements: dict[str, str] = {}
    counter = 0

    def replacer(match: re.Match[str]) -> str:
        nonlocal counter
        placeholder = f"__CODE_BLOCK_{counter}__"
        replacements[placeholder] = match.group(0)
        counter += 1
        return placeholder

    protected = re.sub(r"```[\s\S]*?```", replacer, text)
    return protected, replacements


def _restore_code_blocks(text: str, replacements: dict[str, str]) -> str:
    """プレースホルダをコードブロックに戻す。"""
    for placeholder, original in replacements.items():
        text = text.replace(placeholder, original)
    return text


def chunk_article(
    article: Article,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """1 記事をチャンク分割する。"""
    text = article.body.strip()
    if not text:
        return []

    protected, replacements = _protect_code_blocks(text)

    paragraphs = protected.split("\n\n")

    chunks_text: list[str] = []
    current = ""

    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para.strip()
        if len(candidate) > chunk_size and current:
            chunks_text.append(current)
            # overlap: 前チャンクの末尾 overlap 文字分を次チャンクの先頭に
            tail = current[-overlap:] if len(current) > overlap else current
            current = f"{tail}\n\n{para}".strip()
        else:
            current = candidate

    if current:
        # 最後のチャンクが overlap 以下の長さなら前のチャンクに結合
        if chunks_text and len(current) <= overlap:
            chunks_text[-1] = f"{chunks_text[-1]}\n\n{current}".strip()
        else:
            chunks_text.append(current)

    # コードブロックを復元してチャンクオブジェクトを生成
    result: list[Chunk] = []
    for i, chunk_text in enumerate(chunks_text):
        restored = _restore_code_blocks(chunk_text, replacements)
        result.append(
            Chunk(
                chunk_id=f"{article.slug}_chunk_{i:04d}",
                text=restored,
                article_slug=article.slug,
                article_title=article.title,
                article_url=article.url,
                topics=article.topics,
                chunk_index=i,
            )
        )

    return result


def chunk_articles(
    articles: list[Article],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """複数記事をチャンク分割する。"""
    chunks: list[Chunk] = []
    for article in articles:
        chunks.extend(chunk_article(article, chunk_size, overlap))
    return chunks
