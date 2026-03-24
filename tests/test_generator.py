import pytest

from tech_blog_rag.generator import build_user_prompt, generate
from tech_blog_rag.embedder import get_client
from tech_blog_rag.retriever import SearchResult


def _make_results() -> list[SearchResult]:
    return [
        SearchResult(
            chunk_text="React Aria は Adobe が開発するヘッドレス UI ライブラリです。",
            article_title="React Aria のコンポーネント設計",
            article_url="https://zenn.dev/ryo_manba/articles/react-aria",
            topics=["react", "reactaria"],
            distance=0.1,
        ),
    ]


def test_build_user_prompt_contains_context_and_question():
    results = _make_results()
    prompt = build_user_prompt("React Aria とは？", results)
    assert "React Aria は Adobe" in prompt
    assert "React Aria とは？" in prompt
    assert "https://zenn.dev/ryo_manba/articles/react-aria" in prompt
    assert "React Aria のコンポーネント設計" in prompt


@pytest.mark.api
def test_generate_returns_answer():
    client = get_client()
    results = _make_results()
    answer = generate(client, "React Aria とは何ですか？", results)
    assert answer.text
    assert len(answer.sources) > 0
