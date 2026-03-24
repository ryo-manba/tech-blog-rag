# tech-blog-rag

## プロジェクト概要

`ryo-manba/zenn-content` リポジトリの技術記事（Markdown）を RAG で検索・回答する Python CLI ツール。

## 技術スタック

- Python 3.12+, uv（パッケージ管理）
- Gemini API: `gemini-2.5-flash`（回答生成）, `gemini-embedding-001`（Embedding）
- ChromaDB（ローカルベクトル DB）
- python-frontmatter（Zenn 記事の YAML frontmatter パース）

## ディレクトリ構成

```
tech-blog-rag/
├── pyproject.toml
├── README.md
├── CLAUDE.md
├── .env.example
├── .gitignore
├── src/tech_blog_rag/
│   ├── __init__.py
│   ├── config.py
│   ├── collector.py
│   ├── chunker.py
│   ├── embedder.py
│   ├── retriever.py
│   ├── generator.py
│   └── pipeline.py
├── scripts/
│   ├── ingest.py
│   └── query.py
├── tests/
│   ├── conftest.py
│   ├── test_collector.py
│   ├── test_chunker.py
│   ├── test_embedder.py
│   ├── test_retriever.py
│   ├── test_generator.py
│   └── test_pipeline.py
└── data/chroma_db/
```

## コマンド

```bash
# 依存インストール
uv sync

# テスト（API なし、高速）
uv run pytest tests/ -m "not api" -v

# テスト（API あり、全実行）
uv run pytest tests/ -v

# データ取り込み
uv run python scripts/ingest.py --content-dir ../zenn-content

# 質問
uv run python scripts/query.py -q "質問文"
```

## コーディング規約

- Type hints を全ての関数に付ける
- dataclass を使ったデータ構造定義
- テストは `@pytest.mark.api` で API テストを分離
- エラーは `logging.warning()` で出して処理を継続（1 記事の失敗で全体が止まらない）
