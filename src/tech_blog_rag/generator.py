"""検索結果をコンテキストとしてプロンプトを組み立て、Gemini で回答を生成する。"""
from dataclasses import dataclass

from google import genai
from google.genai.types import GenerateContentConfig

from tech_blog_rag.config import GENERATION_MODEL
from tech_blog_rag.retriever import SearchResult

SYSTEM_PROMPT = """あなたは技術ブログに関する Q&A アシスタントです。

## ルール
- 提供されたコンテキスト（記事の抜粋）に基づいてのみ回答してください
- コンテキストに情報がない場合は「該当する記事が見つかりませんでした」と正直に答えてください
- 回答の最後に「出典」として参照した記事のタイトルと URL を箇条書きで記載してください
- 技術的に正確で簡潔な回答を心がけてください
- コードが含まれる場合は適切にフォーマットして提示してください
- 日本語で回答してください
"""


@dataclass
class Answer:
    text: str
    sources: list[SearchResult]


def build_user_prompt(query: str, results: list[SearchResult]) -> str:
    """検索結果からユーザープロンプトを組み立てる。"""
    context_parts: list[str] = []
    for r in results:
        context_parts.append(
            f"---\n記事: {r.article_title}\nURL: {r.article_url}\n"
            f"トピック: {', '.join(r.topics)}\n\n{r.chunk_text}\n---"
        )

    context = "\n\n".join(context_parts)

    return f"""{context}

## 質問
{query}

## 回答
上記のコンテキストに基づいて回答してください。出典も記載してください。"""


def generate(
    client: genai.Client, query: str, results: list[SearchResult]
) -> Answer:
    """Gemini で回答を生成する。"""
    user_prompt = build_user_prompt(query, results)

    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=user_prompt,
        config=GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=1024,
            system_instruction=SYSTEM_PROMPT,
        ),
    )

    return Answer(
        text=response.text,
        sources=results,
    )
