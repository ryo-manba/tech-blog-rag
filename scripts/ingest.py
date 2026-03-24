"""データ取り込み CLI。"""
import argparse
import logging
import sys

sys.path.insert(0, "src")

from tech_blog_rag.config import CONTENT_DIR
from tech_blog_rag.pipeline import get_status, ingest


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="技術記事を取り込む")
    parser.add_argument("--content-dir", default=CONTENT_DIR, help="zenn-content のパス")
    parser.add_argument("--status", action="store_true", help="ChromaDB のステータスを表示")
    args = parser.parse_args()

    if args.status:
        status = get_status()
        print(f"コレクション: {status['collection_name']}")
        print(f"チャンク数: {status['chunk_count']}")
        return

    result = ingest(content_dir=args.content_dir)
    print(f"取り込み完了:")
    print(f"  記事数: {result['articles']}")
    print(f"  チャンク数: {result['chunks']}")
    print(f"  格納数: {result['stored']}")


if __name__ == "__main__":
    main()
