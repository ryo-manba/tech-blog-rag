# RAGAS 評価の仕組み

このドキュメントでは、`tech-blog-rag` に導入した [RAGAS](https://docs.ragas.io/en/stable/)（Retrieval Augmented Generation Assessment）による自動評価の仕組みを説明します。

---

## 1. なぜ RAGAS を導入したのか？

既存の評価（`scripts/evaluate.py`）はキーワードマッチと Retrieval Hit Rate で測定していました。

```
既存の評価:
  ✅ Retrieval Hit Rate: 期待する記事が検索結果に含まれるか
  ✅ Keyword Coverage: 回答に期待するキーワードが含まれるか
```

これだと「キーワードは含まれているが意味的に間違っている回答」や「検索結果にノイズが多い」といった問題を検出できません。RAGAS は LLM を審判（judge）として使い、意味レベルで評価します。

---

## 2. RAGAS とは？

RAGAS は RAG パイプラインを自動評価するフレームワークです。人手でスコアをつける代わりに、LLM（この場合 Gemini）が回答の質を判定します。

参考: [RAGAS 公式ドキュメント](https://docs.ragas.io/en/stable/)、[論文（arXiv:2309.15217）](https://arxiv.org/abs/2309.15217)

---

## 3. 評価メトリクス

このシステムでは以下の 2 つ（+ オプション 1 つ）のメトリクスを使っています。

### Faithfulness（忠実性）

**「回答がコンテキスト（検索結果）に基づいているか？」** を測定します。

```
スコア 1.0: 回答の全ての主張がコンテキストに裏付けられている
スコア 0.5: 半分の主張のみ裏付けあり
スコア 0.0: コンテキストと無関係な内容を回答している
```

RAGAS は内部で以下の処理を行います:

1. 回答から個々の「主張（claim）」を抽出する
2. 各主張がコンテキスト内に根拠があるか LLM で判定する
3. `根拠あり主張数 / 全主張数` をスコアとして返す

**なぜ重要か**: Faithfulness が低い = ハルシネーション（コンテキストにない情報の捏造）が発生している。

### ContextRecall（コンテキスト再現率）

**「正解に必要な情報が検索で取得できているか？」** を測定します。

```
スコア 1.0: 正解に含まれる全ての情報がコンテキストに存在する
スコア 0.5: 正解の半分の情報しか検索で取得できていない
スコア 0.0: 正解に必要な情報がコンテキストに全く含まれない
```

このメトリクスは ground truth（期待される正解 = `reference`）が必要です。

**なぜ重要か**: ContextRecall が低い = チャンク分割や Embedding の質に問題がある（検索段階で情報を取りこぼしている）。

### AnswerRelevancy（回答関連性）- オプション

**「回答が質問に対して的確か？」** を測定します。`--include-relevancy` フラグで有効化できますが、追加の API コールが必要です。

---

## 4. 評価データセット

`eval/questions.json` に 10 問の評価データがあります。

```json
{
  "question": "React Ariaのコンポーネント設計の特徴は？",
  "expected_slugs": ["react-aria-component-design"],
  "expected_keywords": ["useContextProps", "mergeProps", "コンポジション"],
  "reference": "React Aria Componentsはコンポジションを中心に設計されており..."
}
```

| フィールド | 用途 | 使用する評価 |
|---|---|---|
| `question` | 質問文 | 両方 |
| `expected_slugs` | 期待する記事の slug | 既存評価（Hit Rate） |
| `expected_keywords` | 期待するキーワード | 既存評価（Coverage） |
| `reference` | 期待される正解（ground truth） | RAGAS（ContextRecall） |

`reference` は実際の Zenn 記事を読んで手動で作成しています。この正解の質が ContextRecall の精度に直結します。

---

## 5. 処理フロー

```
eval/questions.json から質問を読み込み
↓
各質問で RAG パイプラインを実行
  ├─ search(): 質問をベクトル化 → ChromaDB で類似チャンク検索
  └─ generate(): チャンクをコンテキストとして Gemini で回答生成
↓
結果を RAGAS の SingleTurnSample に変換
  ├─ user_input: 質問文
  ├─ response: 生成された回答
  ├─ retrieved_contexts: 検索で取得したチャンクのテキスト
  └─ reference: ground truth（期待される正解）
↓
RAGAS がメトリクスを計算
  ├─ Gemini を judge LLM として各主張を検証
  └─ 質問ごと + 全体の平均スコアを出力
↓
結果を表示 + eval/ragas_results.csv に保存
```

---

## 6. Gemini を judge LLM として使う

RAGAS はデフォルトで OpenAI を使いますが、このシステムでは Gemini を judge LLM として使用しています。

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from ragas.llms import LangchainLLMWrapper

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.0,  # 評価の一貫性のため低く設定
)
ragas_llm = LangchainLLMWrapper(llm)
```

[`LangchainLLMWrapper`](https://docs.ragas.io/en/stable/) を使って、RAGAS が期待する LangChain 互換インターフェースに変換しています。

---

## 7. Rate Limit 対策

Gemini の無料枠は 5 リクエスト/分 です。RAGAS は内部で複数の LLM コールを行うため、以下の対策をしています。

```python
from ragas import RunConfig

run_config = RunConfig(
    max_workers=1,      # 並列実行しない（デフォルトは並列）
    max_wait=180,       # 429 エラー時に最大 3 分待機
    max_retries=5,      # 最大 5 回リトライ
    timeout=120,        # 1 コールあたり 2 分タイムアウト
)
```

また、RAG パイプライン実行時にも各質問の間に 15 秒の待機を入れています。10 問の評価で約 14 分かかります。

---

## 8. 使い方

```bash
# 依存インストール（ragas + langchain-google-genai）
uv sync --extra eval

# RAGAS 評価を実行
uv run python scripts/evaluate_ragas.py

# AnswerRelevancy も含める場合（API コール増）
uv run python scripts/evaluate_ragas.py --include-relevancy

# オプション
uv run python scripts/evaluate_ragas.py --questions eval/questions.json --top-k 5
```

### 出力例

```
=== Per-Question Scores ===

Q: React Ariaのコンポーネント設計の特徴は？
  faithfulness: 0.857
  context_recall: 1.000

Q: Next.js のバンドルサイズを分析する方法は？
  faithfulness: 1.000
  context_recall: 0.750

...

=== Aggregate Scores ===
  faithfulness: 0.920 (avg)
  context_recall: 0.850 (avg)
```

詳細な結果は `eval/ragas_results.csv` に保存されます。

---

## 9. 既存評価との使い分け

| | 既存評価（evaluate.py） | RAGAS 評価（evaluate_ragas.py） |
|---|---|---|
| 速度 | 高速（キーワードマッチ） | 低速（LLM judge） |
| 精度 | キーワードの有無のみ | 意味レベルで評価 |
| コスト | Embedding + 生成の API コール | 上記 + judge LLM の API コール |
| 用途 | CI / 簡易チェック | 改善施策の効果測定 |

両方を使い分けるのが効果的です。日常的には `evaluate.py --retrieval-only` で素早くチェックし、チャンク分割やプロンプトを変更した際に `evaluate_ragas.py` で詳細に測定します。

---

## 10. スコアが低い場合の改善指針

| スコア | 原因の可能性 | 改善方向 |
|---|---|---|
| Faithfulness が低い | プロンプト設計の問題 | system_instruction の制約を強化、temperature を下げる |
| ContextRecall が低い | チャンク分割・検索の問題 | チャンクサイズの調整、オーバーラップの増加、リランキングの導入 |
| 両方低い | Embedding 品質の問題 | Embedding モデルの変更、Contextual Retrieval の導入 |
