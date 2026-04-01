"""RAGAS による RAG パイプラインの評価スクリプト。

使い方:
  uv sync --extra eval
  uv run python scripts/evaluate_ragas.py
  uv run python scripts/evaluate_ragas.py --include-relevancy
"""

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import dataclass, field

sys.path.insert(0, "src")

from tech_blog_rag.config import GEMINI_API_KEY, GENERATION_MODEL
from tech_blog_rag.embedder import get_client
from tech_blog_rag.generator import generate
from tech_blog_rag.retriever import search

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "gemma3:27b"


@dataclass
class SampleResult:
    question: str
    faithfulness: float | None = None
    context_recall: float | None = None
    answer_relevancy: float | None = None


def load_questions(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def create_ragas_llm():
    """Ollama を RAGAS の judge LLM としてセットアップする。

    Ollama の OpenAI 互換エンドポイント経由で llm_factory を使用する。
    ローカル実行のためクォータ制限なし。
    """
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    client = AsyncOpenAI(
        api_key="ollama",
        base_url=OLLAMA_BASE_URL,
    )
    return llm_factory(OLLAMA_MODEL, client=client)


SAMPLES_CACHE_PATH = "eval/samples_cache.json"


@dataclass
class CollectedSample:
    question: str
    response: str
    retrieved_contexts: list[str]
    reference: str


def save_samples_cache(samples: list[CollectedSample]) -> None:
    """パイプライン結果をキャッシュに保存する。"""
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
    logger.info(f"パイプライン結果をキャッシュしました: {SAMPLES_CACHE_PATH}")


def load_samples_cache() -> list[CollectedSample] | None:
    """キャッシュからパイプライン結果を読み込む。"""
    try:
        with open(SAMPLES_CACHE_PATH, encoding="utf-8") as f:
            data = json.load(f)
        samples = [CollectedSample(**d) for d in data]
        logger.info(f"キャッシュから {len(samples)} 件のサンプルを読み込みました")
        return samples
    except FileNotFoundError:
        return None


def collect_samples(questions: list[dict], top_k: int) -> list[CollectedSample]:
    """各質問で RAG パイプラインを実行し、結果を収集する。"""
    client = get_client()
    samples = []

    for i, q in enumerate(questions, 1):
        question = q["question"]
        reference = q.get("reference", "")

        logger.info(f"\n[{i}/{len(questions)}] パイプライン実行: {question}")

        # Retrieval
        results = search(client, question, top_k=top_k)
        if not results:
            logger.warning("  検索結果なし（ingest 未実行の可能性があります）")
            continue

        retrieved_titles = [r.article_title for r in results[:3]]
        logger.info(f"  検索 Top-3: {retrieved_titles}")

        # Generation（Rate limit 対策: リトライ付き）
        answer = None
        for attempt in range(5):
            try:
                answer = generate(client, question, results)
                break
            except Exception as e:
                if ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)) and attempt < 4:
                    wait = 30 * (attempt + 1)
                    logger.info(f"  Rate limit - {wait}秒待機...")
                    time.sleep(wait)
                else:
                    raise

        logger.info(f"  回答: {answer.text[:80]}...")

        samples.append(CollectedSample(
            question=question,
            response=answer.text,
            retrieved_contexts=[r.chunk_text for r in results],
            reference=reference,
        ))

        # Rate limit 対策
        if i < len(questions):
            time.sleep(5)

    return samples


def run_evaluation(
    samples: list[CollectedSample],
    include_relevancy: bool = False,
) -> list[SampleResult]:
    """各サンプルに対して RAGAS メトリクスを個別にスコアリングする。"""
    from ragas.metrics.collections import ContextRecall, Faithfulness

    ragas_llm = create_ragas_llm()
    faithfulness = Faithfulness(llm=ragas_llm)
    context_recall = ContextRecall(llm=ragas_llm)

    answer_relevancy = None
    if include_relevancy:
        from ragas.metrics.collections import AnswerRelevancy
        answer_relevancy = AnswerRelevancy(llm=ragas_llm)

    logger.info("\n" + "=" * 60)
    logger.info("RAGAS 評価を実行中...")
    logger.info("=" * 60)

    results: list[SampleResult] = []

    for i, sample in enumerate(samples, 1):
        logger.info(f"\n[{i}/{len(samples)}] 評価中: {sample.question}")
        result = SampleResult(question=sample.question)

        # Faithfulness
        try:
            faith_result = faithfulness.score(
                user_input=sample.question,
                response=sample.response,
                retrieved_contexts=sample.retrieved_contexts,
            )
            result.faithfulness = faith_result.value
            logger.info(f"  Faithfulness:   {result.faithfulness:.3f}")
        except Exception as e:
            logger.warning(f"  Faithfulness エラー: {e}")

        # ContextRecall
        try:
            recall_result = context_recall.score(
                user_input=sample.question,
                retrieved_contexts=sample.retrieved_contexts,
                reference=sample.reference,
            )
            result.context_recall = recall_result.value
            logger.info(f"  ContextRecall:  {result.context_recall:.3f}")
        except Exception as e:
            logger.warning(f"  ContextRecall エラー: {e}")

        # AnswerRelevancy (optional)
        if answer_relevancy is not None:
            try:
                rel_result = answer_relevancy.score(
                    user_input=sample.question,
                    response=sample.response,
                )
                result.answer_relevancy = rel_result.value
                logger.info(f"  AnswerRelevancy: {result.answer_relevancy:.3f}")
            except Exception as e:
                logger.warning(f"  AnswerRelevancy エラー: {e}")

        results.append(result)

    return results


def print_and_save_results(results: list[SampleResult]) -> None:
    """評価結果を表示し CSV に保存する。"""
    logger.info("\n" + "=" * 60)
    logger.info("=== Per-Question Scores ===")

    for r in results:
        logger.info(f"\nQ: {r.question}")
        if r.faithfulness is not None:
            logger.info(f"  Faithfulness:    {r.faithfulness:.3f}")
        if r.context_recall is not None:
            logger.info(f"  ContextRecall:   {r.context_recall:.3f}")
        if r.answer_relevancy is not None:
            logger.info(f"  AnswerRelevancy: {r.answer_relevancy:.3f}")

    # Aggregate scores
    logger.info("\n" + "=" * 60)
    logger.info("=== Aggregate Scores ===")

    for metric_name in ("faithfulness", "context_recall", "answer_relevancy"):
        values = [getattr(r, metric_name) for r in results if getattr(r, metric_name) is not None]
        if values:
            avg = sum(values) / len(values)
            logger.info(f"  {metric_name}: {avg:.3f} (avg, n={len(values)})")

    logger.info("=" * 60)

    # CSV 保存
    output_path = "eval/ragas_results.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "faithfulness", "context_recall", "answer_relevancy"])
        for r in results:
            writer.writerow([r.question, r.faithfulness, r.context_recall, r.answer_relevancy])

    logger.info(f"\n詳細結果を保存しました: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS による RAG 評価")
    parser.add_argument(
        "--questions",
        default="eval/questions.json",
        help="評価データセットのパス",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="検索結果の件数",
    )
    parser.add_argument(
        "--include-relevancy",
        action="store_true",
        help="AnswerRelevancy メトリクスを含める（API コール増）",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="前回のパイプライン結果キャッシュを使用する（Gemini API を呼ばない）",
    )
    args = parser.parse_args()

    questions = load_questions(args.questions)

    # reference フィールドの確認
    missing_ref = [q["question"] for q in questions if not q.get("reference")]
    if missing_ref:
        logger.warning("警告: 以下の質問に reference がありません（ContextRecall が不正確になります）:")
        for q in missing_ref:
            logger.warning(f"  - {q}")

    logger.info(f"評価開始: {len(questions)} 件の質問\n")

    # Step 1: RAG パイプライン実行（キャッシュがあればスキップ）
    samples = None
    if args.use_cache:
        samples = load_samples_cache()
        if samples is None:
            logger.warning("キャッシュが見つかりません。パイプラインを実行します。")

    if samples is None:
        samples = collect_samples(questions, top_k=args.top_k)
        save_samples_cache(samples)

    if not samples:
        logger.error("サンプルが0件です。ingest を先に実行してください。")
        sys.exit(1)

    # Step 2: RAGAS 評価
    results = run_evaluation(samples, include_relevancy=args.include_relevancy)

    # Step 3: 結果表示
    print_and_save_results(results)


if __name__ == "__main__":
    main()
