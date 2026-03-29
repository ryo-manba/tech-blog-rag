# tech-blog-rag

技術記事（Markdown）を対象に、自然言語で検索・Q&A ができる RAG ツールです。
Zenn や Qiita など、任意の Markdown 記事ディレクトリをデータソースとして利用できます。

## 特徴

- Gemini API による高精度な意味検索＆回答生成
- ChromaDB によるローカルベクトル DB 管理
- コードブロックや日本語記事にも最適化
- データソースは .env で自由に指定可能（Zenn 以外も OK）

## データソースのカスタマイズ

デフォルトでは [ryo-manba/zenn-content](https://github.com/ryo-manba/zenn-content) を例にしていますが、
`.env` の `CONTENT_DIR` や `ZENN_BASE_URL` を変更すれば、他の Markdown 記事ディレクトリも利用できます。

例:

```
CONTENT_DIR=/path/to/your/markdown-articles
ZENN_BASE_URL=https://your.site/articles
```

## アーキテクチャ

- **Embedding**: Gemini Embedding API (`gemini-embedding-001`)
- **Vector DB**: ChromaDB（ローカル）
- **LLM**: Gemini 2.5 Flash (`gemini-2.5-flash`)
- **データソース**: 任意の Markdown 記事ディレクトリ
- **コスト**: ¥0（Gemini API 無料枠内で完結）

## セットアップ

```bash
git clone https://github.com/ryo-manba/tech-blog-rag.git
cd tech-blog-rag
uv sync


# API キー設定
cp .env.example .env
# .env の GEMINI_API_KEY を設定


# zenn-content を取得
git clone https://github.com/ryo-manba/zenn-content.git ../zenn-content
```

## 使い方

```bash
# 記事のベクトル化
uv run python scripts/ingest.py --content-dir ../zenn-content

# ステータス確認
uv run python scripts/ingest.py --status

# ワンショット質問
uv run python scripts/query.py -q "React Aria のコンポーネント設計の特徴は？"

# 対話モード
uv run python scripts/query.py
```

## 技術的なポイント

- **task_type 非対称設定**: Embedding 時に Document / Query で `task_type` を使い分け、検索精度を向上
- **コードブロック保護チャンキング**: コードブロックを保護して分割することで技術記事の文脈を保持
- **日本語最適化チャンクサイズ**: 500 文字 / 100 文字オーバーラップで日本語記事に最適化

## テスト

```bash
# API を呼ばないテスト（高速）
uv run pytest tests/ -m "not api" -v

# 全テスト（API キーが必要）
uv run pytest tests/ -v
```

## 性能評価

`eval/questions.json` に評価用の質問・期待出典・期待キーワードを定義しています。

```bash
# 検索精度のみ評価（Embedding API のみ使用）
uv run python scripts/evaluate.py --retrieval-only

# 完全評価（検索 + 回答生成、Rate limit 対策で数分かかる）
uv run python scripts/evaluate.py
```
