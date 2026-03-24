# tech-blog-rag

自分の技術ブログ記事（Zenn）に対して自然言語で質問できる RAG Q&A ボット。

## アーキテクチャ

- **Embedding**: Gemini Embedding API (`gemini-embedding-001`)
- **Vector DB**: ChromaDB（ローカル）
- **LLM**: Gemini 2.5 Flash (`gemini-2.5-flash`)
- **データソース**: [ryo-manba/zenn-content](https://github.com/ryo-manba/zenn-content)
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
uv run python scripts/query.py -q "React Ariaのコンポーネント設計の特徴は？"

# 対話モード
uv run python scripts/query.py
```

## 技術的なポイント

- **task_type 非対称設定**: Embedding 時に Document / Query で `task_type` を使い分け、検索精度を向上
- **コードブロック保護チャンキング**: コードブロックを保護して分割することで技術記事の文脈を保持
- **日本語最適化チャンクサイズ**: 500文字 / 100文字オーバーラップで日本語記事に最適化

## テスト

```bash
# API を呼ばないテスト（高速）
uv run pytest tests/ -m "not api" -v

# 全テスト（API キーが必要）
uv run pytest tests/ -v
```
