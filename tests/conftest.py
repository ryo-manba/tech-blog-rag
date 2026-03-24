import pytest


@pytest.fixture
def sample_md_dir(tmp_path):
    """テスト用 Markdown ファイルを含むディレクトリを作成する。"""
    articles_dir = tmp_path / "articles"
    articles_dir.mkdir()

    # published=true の通常記事
    (articles_dir / "test-article-1.md").write_text(
        """\
---
title: "拡張性に優れた React Aria のコンポーネント設計"
emoji: "😺"
type: "tech"
topics: ["react", "reactaria", "フロントエンド"]
published: true
publication_name: "cybozu_frontend"
---

## はじめに

React Aria は Adobe が開発するヘッドレス UI ライブラリです。

### コンポーネント設計

アクセシビリティを重視した設計になっています。
""",
        encoding="utf-8",
    )

    # published=false のドラフト
    (articles_dir / "draft-article.md").write_text(
        """\
---
title: "下書き記事"
emoji: "📝"
type: "tech"
topics: ["draft"]
published: false
---

これは下書きです。
""",
        encoding="utf-8",
    )

    # published=true のコードブロックが多い記事
    (articles_dir / "code-heavy.md").write_text(
        """\
---
title: "Next.js でバンドルサイズを分析する"
emoji: "📦"
type: "tech"
topics: ["nextjs", "performance"]
published: true
---

## バンドルサイズの分析

以下のコマンドで分析できます。

```bash
npx next build
npx @next/bundle-analyzer
```

## 結果の見方

生成されたレポートを確認します。

```javascript
const config = {
  webpack: (config) => {
    config.plugins.push(new BundleAnalyzerPlugin());
    return config;
  },
};
```
""",
        encoding="utf-8",
    )

    return str(articles_dir)


@pytest.fixture
def tmp_chroma_dir(tmp_path):
    db_dir = tmp_path / "chroma_db"
    db_dir.mkdir()
    return str(db_dir)
