"""Unit tests for extraction, SQLite vectors, retrieval, and runtime policy."""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

import pytest

os.environ["TRACEFOLD_RUNTIME"] = "fake"
os.environ["TRACEFOLD_RETRIEVAL_THRESHOLD"] = "0.12"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from ingestion import DocumentParser, TextBlock
from rag import REFUSAL, RAGPipeline
from evaluation import load_evaluation_dataset
from runtime import FakeRuntime, RuntimeUnavailable, create_runtime
from storage import TracefoldStore, decode_embedding, encode_embedding


def test_markdown_sections_are_retained(tmp_path):
    path = tmp_path / "guide.md"
    path.write_text(
        "# Setup\n\nDownload the two local models before disconnecting.\n\n"
        "## Privacy\n\nUploaded documents never leave the device.",
        encoding="utf-8",
    )
    chunks = DocumentParser().parse(path, chunk_size=12, overlap=2)
    assert [chunk.section for chunk in chunks] == ["Setup", "Privacy"]
    assert all(chunk.token_count <= 12 for chunk in chunks)


def test_markdown_chunks_never_merge_adjacent_sections(tmp_path):
    path = tmp_path / "sections.md"
    path.write_text(
        "# Guide\n\nIntroductory context.\n\n## Arrival\n\nLate arrival instructions.\n\n"
        "## Safety\n\nEmergency instructions.",
        encoding="utf-8",
    )
    chunks = DocumentParser().parse(path, chunk_size=240, overlap=45)
    assert [chunk.section for chunk in chunks] == ["Guide", "Arrival", "Safety"]
    assert "Safety" not in chunks[1].text


def test_paragraph_aware_chunking_and_overlap():
    blocks = [
        TextBlock(text=" ".join(f"alpha{i}" for i in range(80)), page=1),
        TextBlock(text=" ".join(f"beta{i}" for i in range(80)), page=2),
    ]
    chunks = DocumentParser().chunk_blocks(blocks, chunk_size=100, overlap=20)
    assert len(chunks) >= 2
    assert chunks[0].page_start == 1
    assert chunks[-1].page_end == 2
    assert all(chunk.token_count <= 100 for chunk in chunks)


def test_scanned_pdf_reports_ocr_hint(tmp_path):
    import fitz

    path = tmp_path / "scan.pdf"
    document = fitz.open()
    document.new_page()
    document.save(path)
    document.close()
    with pytest.raises(ValueError, match="OCR"):
        DocumentParser().parse(path)


def test_embedding_binary_round_trip_is_float32():
    values = [0.125, -0.5, 1.0]
    blob = encode_embedding(values)
    assert len(blob) == 12
    assert decode_embedding(blob, 3) == values


def test_store_persists_vector_as_blob(tmp_path):
    store = TracefoldStore(tmp_path / "tracefold.db")
    store.add_document(
        document_id="doc_1",
        filename="guide.txt",
        stored_name="doc_1.txt",
        media_type="text/plain",
        sha256="a" * 64,
        size_bytes=10,
        chunks=[
            {
                "id": "chunk_1",
                "ordinal": 0,
                "text": "local evidence",
                "page_start": None,
                "page_end": None,
                "section": None,
                "token_count": 2,
                "embedding": [0.5, 0.25],
            }
        ],
    )
    with sqlite3.connect(store.db_path) as connection:
        kind, length = connection.execute(
            "SELECT typeof(embedding), length(embedding) FROM chunks"
        ).fetchone()
    assert (kind, length) == ("blob", 8)
    assert store.all_chunks()[0]["embedding"] == [0.5, 0.25]


def test_cosine_ranking_and_grounded_refusal(tmp_path):
    store = TracefoldStore(tmp_path / "tracefold.db")
    runtime = FakeRuntime()
    rag = RAGPipeline(store, runtime)
    chunks = DocumentParser().chunk_blocks(
        [TextBlock("Foundry Local performs private inference directly on the device.")]
    )
    store.add_document(
        document_id="doc_1",
        filename="privacy.txt",
        stored_name="doc_1.txt",
        media_type="text/plain",
        sha256="b" * 64,
        size_bytes=20,
        chunks=rag.embed_chunks(chunks),
    )
    answer = rag.answer("Where does private inference run?")
    assert answer["refused"] is False
    assert answer["citations"][0]["filename"] == "privacy.txt"
    refused = rag.answer("What is the capital of Botswana?")
    assert refused["refused"] is True


def test_runtime_mode_must_be_explicitly_valid(monkeypatch):
    monkeypatch.setenv("TRACEFOLD_RUNTIME", "unexpected")
    with pytest.raises(RuntimeUnavailable, match="must be"):
        create_runtime()


def test_unknown_or_missing_model_citations_are_rejected():
    citations = [{"index": 1, "filename": "guide.md"}]
    assert RAGPipeline.validate_generated_answer("A claim without evidence.", citations) == (
        REFUSAL,
        [],
    )
    assert RAGPipeline.validate_generated_answer("A claim [9].", citations) == (REFUSAL, [])
    accepted, used = RAGPipeline.validate_generated_answer("A grounded claim [1].", citations)
    assert accepted == "A grounded claim [1]."
    assert used == citations


def test_checked_in_evaluation_dataset_resolves_demo_sources():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    payload = load_evaluation_dataset(
        Path(project_root) / "docs" / "eval-dataset.json",
        Path(project_root) / "demo_docs",
    )
    assert len(payload["cases"]) == 12
    assert payload["dataset_id"] == "northstar-summer-lab-2026-v1"
    assert payload["resolved_sources"]["01-program-handbook.md"].endswith(
        "demo_docs/01-program-handbook.md"
    )
