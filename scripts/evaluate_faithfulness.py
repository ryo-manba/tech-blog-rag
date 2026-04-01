"""Faithfulness 評価の独自実装。

RAGAS と同じ 2 ステップ方式（主張抽出 → 検証）を独自プロンプトで実装し、
RAGAS の結果と比較する。

使い方:
  uv run python scripts/evaluate_faithfulness.py
  uv run python scripts/evaluate_faithfulness.py --use-cache
"""

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import dataclass

from openai import OpenAI

sys.path.insert(0, "src")

from tech_blog_rag.config import GEMINI_API_KEY
from tech_blog_rag.embedder import get_client
from tech_blog_rag.generator import generate
from tech_blog_rag.retriever import search

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "gemma3:12b"
SAMPLES_CACHE_PATH = "eval/samples_cache.json"


@dataclass
class CollectedSample:
    question: str
    response: str
    retrieved_contexts: list[str]
    reference: str


@dataclass
class ClaimVerdict:
    claim: str
    supported: bool
    reason: str


def get_ollama_client() -> OpenAI:
    return OpenAI(api_key="ollama", base_url=OLLAMA_BASE_URL, timeout=600.0)


MAX_RESPONSE_LEN = 500


def extract_claims(client: OpenAI, question: str, response: str) -> list[str]:
    """回答から個々の主張を抽出する。"""
    truncated_response = response[:MAX_RESPONSE_LEN]
    prompt = f"""以下の質問と回答を読んで、回答に含まれる事実の主張を一つずつ抽出してください。

ルール:
- 主張は1文で完結させ、代名詞（「それ」「これ」など）は使わず具体的に書く
- 「〜です」「〜します」のような丁寧語は不要。簡潔な体言止めや常体でよい
- 意見や感想ではなく、事実として検証可能な主張のみ抽出する

例:
  質問: このライブラリの特徴は何ですか？
  回答: このライブラリはMITライセンスで公開されています。設定ファイルはYAML形式で記述し、プラグインで機能を拡張できます。月間ダウンロード数は100万を超えています。

  抽出結果:
  {{"claims": [
    "このライブラリはMITライセンスで公開されている",
    "設定ファイルはYAML形式で記述する",
    "プラグインで機能を拡張できる",
    "月間ダウンロード数は100万を超えている"
  ]}}

では、以下の質問と回答について同様に抽出してください。

質問: {question}

回答: {truncated_response}

JSON 形式で出力してください:
{{"claims": ["主張1", "主張2", ...]}}"""

    resp = client.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    try:
        data = json.loads(resp.choices[0].message.content)
        return data.get("claims", [])
    except (json.JSONDecodeError, KeyError):
        logger.warning(f"  主張抽出の JSON パースに失敗: {resp.choices[0].message.content[:200]}")
        return []


def verify_single_claim(
    client: OpenAI, claim: str, context_text: str
) -> ClaimVerdict:
    """1 件の主張がコンテキストに裏付けられるか検証する。"""
    prompt = f"""以下のコンテキストを読んで、主張がコンテキストから直接裏付けられるか判定してください。

判定基準:
- supported: コンテキストに書いてある内容から直接推論できる
- not_supported: コンテキストに書いていない、または矛盾する

例:
  コンテキスト: このライブラリはMITライセンスで公開されている。設定はYAMLファイルに記述する。
  主張: 設定はJSON形式で記述する
  判定: {{"supported": false, "reason": "コンテキストでは「YAMLファイルに記述」とあり、JSONではない"}}

では、以下について判定してください。

コンテキスト:
{context_text}

主張: {claim}

JSON 形式で出力してください:
{{"supported": true/false, "reason": "判定理由"}}"""

    resp = client.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    try:
        data = json.loads(resp.choices[0].message.content)
        return ClaimVerdict(
            claim=claim,
            supported=bool(data.get("supported", False)),
            reason=data.get("reason", ""),
        )
    except (json.JSONDecodeError, KeyError):
        logger.warning(f"  検証の JSON パースに失敗: {resp.choices[0].message.content[:200]}")
        return ClaimVerdict(claim=claim, supported=False, reason="パース失敗")


def verify_claims(
    client: OpenAI, claims: list[str], contexts: list[str]
) -> list[ClaimVerdict]:
    """各主張を 1 件ずつ検証する。

    コンテキストが長すぎるとタイムアウトするため、
    各チャンクを先頭 300 文字に切り詰めてから渡す。
    """
    truncated = [c[:300] for c in contexts]
    context_text = "\n---\n".join(truncated)
    verdicts = []
    for claim in claims:
        verdict = verify_single_claim(client, claim, context_text)
        verdicts.append(verdict)
    return verdicts


def compute_faithfulness(verdicts: list[ClaimVerdict]) -> float:
    """Faithfulness スコアを計算する。"""
    if not verdicts:
        return float("nan")
    supported = sum(1 for v in verdicts if v.supported)
    return supported / len(verdicts)


def load_samples_cache() -> list[CollectedSample] | None:
    try:
        with open(SAMPLES_CACHE_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return [CollectedSample(**d) for d in data]
    except FileNotFoundError:
        return None


def collect_samples(questions: list[dict], top_k: int) -> list[CollectedSample]:
    gemini_client = get_client()
    samples = []

    for i, q in enumerate(questions, 1):
        question = q["question"]
        reference = q.get("reference", "")

        logger.info(f"\n[{i}/{len(questions)}] パイプライン実行: {question}")

        results = search(gemini_client, question, top_k=top_k)
        if not results:
            logger.warning("  検索結果なし")
            continue

        answer = None
        for attempt in range(5):
            try:
                answer = generate(gemini_client, question, results)
                break
            except Exception as e:
                if ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "503" in str(e)) and attempt < 4:
                    wait = 30 * (attempt + 1)
                    logger.info(f"  Rate limit - {wait}秒待機...")
                    time.sleep(wait)
                else:
                    raise

        if answer is None:
            logger.warning("  生成失敗")
            continue

        samples.append(CollectedSample(
            question=question,
            response=answer.text,
            retrieved_contexts=[r.chunk_text for r in results],
            reference=reference,
        ))
        logger.info(f"  OK ({len(answer.text)} chars)")

        if i < len(questions):
            time.sleep(5)

    # キャッシュ保存
    data = [
        {
            "question": s.question,
            "response": s.response,
            "retrieved_contexts": s.retrieved_contexts,
            "reference": s.reference,
        }
        for s in samples
    ]
    with open(SAMPLES_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"\nキャッシュ保存: {SAMPLES_CACHE_PATH}")

    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Faithfulness 独自評価")
    parser.add_argument("--questions", default="eval/questions.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--use-cache", action="store_true",
                        help="前回のパイプライン結果キャッシュを使用する")
    args = parser.parse_args()

    # Step 1: サンプル準備
    samples = None
    if args.use_cache:
        samples = load_samples_cache()
        if samples:
            logger.info(f"キャッシュから {len(samples)} 件のサンプルを読み込みました")
        else:
            logger.warning("キャッシュなし。パイプラインを実行します。")

    if samples is None:
        with open(args.questions, encoding="utf-8") as f:
            questions = json.load(f)
        samples = collect_samples(questions, top_k=args.top_k)

    if not samples:
        logger.error("サンプルが0件です。")
        sys.exit(1)

    # Step 2: Faithfulness 評価
    ollama = get_ollama_client()
    results: list[dict] = []

    logger.info("\n" + "=" * 60)
    logger.info("Faithfulness 評価を実行中（Ollama）...")
    logger.info("=" * 60)

    for i, sample in enumerate(samples, 1):
        logger.info(f"\n[{i}/{len(samples)}] {sample.question}")

        # 主張抽出
        try:
            claims = extract_claims(ollama, sample.question, sample.response)
        except Exception as e:
            logger.warning(f"  主張抽出でエラー（スキップ）: {e}")
            results.append({"question": sample.question, "score": float("nan"), "claims": [], "verdicts": []})
            continue

        logger.info(f"  抽出した主張: {len(claims)} 件")
        for j, c in enumerate(claims, 1):
            logger.info(f"    {j}. {c}")

        if not claims:
            results.append({"question": sample.question, "score": float("nan"), "claims": [], "verdicts": []})
            continue

        # 検証
        try:
            verdicts = verify_claims(ollama, claims, sample.retrieved_contexts)
        except Exception as e:
            logger.warning(f"  検証でエラー（スキップ）: {e}")
            results.append({"question": sample.question, "score": float("nan"), "claims": claims, "verdicts": []})
            continue

        score = compute_faithfulness(verdicts)

        logger.info(f"  検証結果:")
        for v in verdicts:
            mark = "✓" if v.supported else "✗"
            logger.info(f"    {mark} {v.claim}")
            logger.info(f"      理由: {v.reason}")

        logger.info(f"  Faithfulness: {score:.3f} ({sum(1 for v in verdicts if v.supported)}/{len(verdicts)})")

        results.append({
            "question": sample.question,
            "score": score,
            "claims": claims,
            "verdicts": [
                {"claim": v.claim, "supported": v.supported, "reason": v.reason}
                for v in verdicts
            ],
        })

    # サマリー
    scores = [r["score"] for r in results if r["score"] == r["score"]]  # NaN 除外
    logger.info("\n" + "=" * 60)
    logger.info("=== 結果サマリー ===")
    for r in results:
        s = f"{r['score']:.3f}" if r["score"] == r["score"] else "N/A"
        logger.info(f"  {s}  {r['question']}")
    if scores:
        avg = sum(scores) / len(scores)
        logger.info(f"\n  平均 Faithfulness: {avg:.3f} (n={len(scores)})")
    logger.info("=" * 60)

    # 詳細結果を JSON で保存
    output_path = "eval/faithfulness_custom_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"\n詳細結果を保存しました: {output_path}")

    # CSV でも保存（RAGAS との比較用）
    csv_path = "eval/faithfulness_custom_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "faithfulness_custom", "num_claims", "num_supported"])
        for r in results:
            supported = sum(1 for v in r["verdicts"] if v["supported"])
            writer.writerow([r["question"], r["score"], len(r["claims"]), supported])
    logger.info(f"CSV 結果を保存しました: {csv_path}")


if __name__ == "__main__":
    main()
