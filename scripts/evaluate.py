"""RAG パイプラインの性能評価スクリプト。"""
import argparse
import json
import sys
import time

sys.path.insert(0, "src")

from tech_blog_rag.embedder import get_client
from tech_blog_rag.generator import generate
from tech_blog_rag.retriever import SearchResult, search


def load_questions(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_retrieval(
    results: list[SearchResult], expected_slugs: list[str]
) -> bool:
    """期待する記事が検索結果の top-k に含まれるかを判定。"""
    retrieved_slugs = {r.article_url.split("/")[-1] for r in results}
    return any(slug in retrieved_slugs for slug in expected_slugs)


def evaluate_keywords(answer_text: str, expected_keywords: list[str]) -> list[str]:
    """回答に含まれるキーワードのリストを返す。"""
    return [kw for kw in expected_keywords if kw.lower() in answer_text.lower()]


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 性能評価")
    parser.add_argument(
        "--questions", default="eval/questions.json", help="評価データセットのパス"
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="検索精度のみ評価（LLM 回答生成をスキップ）",
    )
    parser.add_argument("--top-k", type=int, default=5, help="検索結果の件数")
    args = parser.parse_args()

    questions = load_questions(args.questions)
    client = get_client()

    retrieval_hits = 0
    total_keywords = 0
    matched_keywords = 0
    results_detail: list[dict] = []

    print(f"評価開始: {len(questions)} 件の質問\n")
    print("=" * 60)

    for i, q in enumerate(questions, 1):
        question = q["question"]
        expected_slugs = q["expected_slugs"]
        expected_kws = q["expected_keywords"]

        print(f"\n[{i}/{len(questions)}] {question}")

        # Retrieval 評価
        search_results = search(client, question, top_k=args.top_k)
        hit = evaluate_retrieval(search_results, expected_slugs)
        if hit:
            retrieval_hits += 1

        retrieved_titles = [r.article_title for r in search_results[:3]]
        print(f"  検索: {'HIT' if hit else 'MISS'} | Top-3: {retrieved_titles}")

        if not args.retrieval_only:
            time.sleep(2)

        # 回答生成 + キーワード評価
        detail = {"question": question, "retrieval_hit": hit}
        if not args.retrieval_only:
            # Rate limit 対策: リトライ付きで回答生成
            answer = None
            for attempt in range(3):
                try:
                    answer = generate(client, question, search_results)
                    break
                except Exception as e:
                    if "429" in str(e) and attempt < 2:
                        wait = 15 * (attempt + 1)
                        print(f"  Rate limit - {wait}秒待機...")
                        time.sleep(wait)
                    else:
                        raise

            found_kws = evaluate_keywords(answer.text, expected_kws)
            total_keywords += len(expected_kws)
            matched_keywords += len(found_kws)
            missing_kws = set(expected_kws) - set(found_kws)

            print(f"  キーワード: {len(found_kws)}/{len(expected_kws)}", end="")
            if missing_kws:
                print(f" (欠落: {missing_kws})", end="")
            print()

            detail["keyword_found"] = found_kws
            detail["keyword_missing"] = list(missing_kws)
            # 無料枠 5 req/min 対策: 各質問後に待機
            time.sleep(15)

        results_detail.append(detail)

    # サマリー
    print("\n" + "=" * 60)
    print("=== 評価結果 ===")
    print(
        f"Retrieval Hit Rate: {retrieval_hits}/{len(questions)} "
        f"({retrieval_hits / len(questions) * 100:.0f}%)"
    )
    if not args.retrieval_only and total_keywords > 0:
        print(
            f"Keyword Coverage:   {matched_keywords}/{total_keywords} "
            f"({matched_keywords / total_keywords * 100:.0f}%)"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
