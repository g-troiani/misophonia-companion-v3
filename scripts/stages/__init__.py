"""Stage helpers live here so `pipeline_integrated` can still be the single-file
reference implementation while every stage can be invoked on its own."""

"""
Namespace package so the stage modules can be imported with
    from stages import <module>
"""
__all__ = [
    "common",
    "extract_clean",
    "llm_enrich",
    "chunk_text",
    "db_upsert",
    "embed_vectors",
] 