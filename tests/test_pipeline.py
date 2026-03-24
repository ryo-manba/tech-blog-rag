import pytest

from tech_blog_rag.pipeline import ingest, query


@pytest.mark.api
def test_full_ingest_and_query(sample_md_dir, tmp_chroma_dir):
    # sample_md_dir は articles/ ディレクトリなので、その親を content_dir として使う
    import os

    content_dir = os.path.dirname(sample_md_dir)
    result = ingest(content_dir=content_dir, db_path=tmp_chroma_dir)
    assert result["articles"] == 2
    assert result["chunks"] > 0
    assert result["stored"] > 0

    answer = query("React Aria とは？", db_path=tmp_chroma_dir)
    assert answer.text
    assert len(answer.sources) > 0
