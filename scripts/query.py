"""質問応答 CLI。"""
import argparse
import sys

sys.path.insert(0, "src")

from tech_blog_rag.pipeline import query


def print_answer(question: str) -> None:
    answer = query(question)
    print(f"\n{answer.text}\n")


def interactive_mode() -> None:
    print("tech-blog-rag 対話モード（exit/quit/q で終了）")
    while True:
        try:
            question = input("\n質問> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n終了します。")
            break

        if question.lower() in ("exit", "quit", "q", ""):
            print("終了します。")
            break

        print_answer(question)


def main() -> None:
    parser = argparse.ArgumentParser(description="技術記事に質問する")
    parser.add_argument("-q", "--question", help="質問文")
    args = parser.parse_args()

    if args.question:
        print_answer(args.question)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
