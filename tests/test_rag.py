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

from evaluation import Evaluator, load_evaluation_dataset
from ingestion import DocumentParser, TextBlock
from rag import REFUSAL, REFUSAL_TR, RAGPipeline, RetrievalResult
from runtime import FakeRuntime, RuntimeUnavailable, create_runtime
from storage import TracefoldStore, decode_embedding, encode_embedding
from translation import (
    BaseTranslationEngine,
    ControlledTranslationService,
    FakeTranslationEngine,
)

from scripts.seed_demo import require_complete_evaluation


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
    citations = [{"index": 1, "filename": "guide.md", "text": "Grounded evidence: 17:30."}]
    assert RAGPipeline.validate_generated_answer("A claim without evidence.", citations) == (
        REFUSAL,
        [],
    )
    assert RAGPipeline.validate_generated_answer("A claim [9].", citations) == (REFUSAL, [])
    accepted, used = RAGPipeline.validate_generated_answer("A grounded claim [1].", citations)
    assert accepted == "A grounded claim [1]."
    assert used == citations


def test_every_factual_sentence_requires_its_own_valid_citation():
    citations = [
        {"index": 1, "filename": "guide.md", "text": "Check out by 10:00."},
        {"index": 2, "filename": "travel.md", "text": "Store luggage until 12:15."},
    ]

    answer = "Check out by 10:00 [1]. Store luggage until 12:15 [2]."
    assert RAGPipeline.validate_generated_answer(answer, citations) == (answer, citations)
    assert RAGPipeline.citation_sentence_coverage(answer, citations) is True

    uncited_sentence = "Check out by 10:00. Store luggage until 12:15 [2]."
    assert RAGPipeline.validate_generated_answer(uncited_sentence, citations) == (REFUSAL, [])
    assert RAGPipeline.citation_sentence_coverage(uncited_sentence, citations) is False


def test_contact_and_numeric_facts_must_exist_in_the_cited_passage():
    citations = [
        {
            "index": 1,
            "filename": "contacts.md",
            "text": "Call +90 312 555 0149 before 19:30 and email program@northstar.example.",
        }
    ]
    grounded = (
        "Call the +90 312 555 0149 number before 19:30 and send email to "
        "program@northstar.example [1]."
    )
    assert RAGPipeline.validate_generated_answer(grounded, citations) == (
        grounded,
        citations,
    )
    assert RAGPipeline.protected_facts_grounded(grounded, citations) is True

    invented = "Call the +90 312 555 9999 number before 21:00 [1]."
    assert RAGPipeline.validate_generated_answer(invented, citations) == (REFUSAL, [])


def test_turkish_prompt_preserves_evidence_and_localizes_refusal():
    evidence = "Foundry Local, 17 Temmuz 2026 saat 14:30'da hazır olacak."
    result = RetrievalResult(
        chunk={
            "filename": "duyuru.txt",
            "page_start": 1,
            "section": "Takvim",
            "text": evidence,
        },
        score=0.9,
    )

    messages = RAGPipeline._messages("Ne zaman hazır olacak?", [result], locale="tr")
    system_prompt = messages[0]["content"]

    assert "Write the final answer in Turkish." in system_prompt
    assert REFUSAL_TR in system_prompt
    assert evidence in system_prompt
    assert "[1] Source: duyuru.txt" in system_prompt
    assert RAGPipeline.validate_generated_answer(
        "Kaynağa dayanmayan yanıt.", [{"index": 1}], locale="tr"
    ) == (REFUSAL_TR, [])


def test_turkish_retrieval_keeps_original_and_adds_local_search_translation(tmp_path):
    class QueryTranslationEngine(BaseTranslationEngine):
        def translate_batch(self, texts, *, source_locale, target_locale):
            assert source_locale == "tr"
            assert target_locale == "en"
            assert texts == ["Saat 20:00'den sonra varış prosedürü nedir?"]
            return ["What is the late-arrival procedure after 20:00?"]

        def status(self, *, probe=False):
            return {"runtime": "query-test", "ready": True}

    translation = ControlledTranslationService(QueryTranslationEngine())
    pipeline = RAGPipeline(
        TracefoldStore(tmp_path / "tracefold.db"),
        FakeRuntime(),
        translation,
    )

    assert pipeline._retrieval_queries(
        "Saat 20:00'den sonra varış prosedürü nedir?", locale="tr"
    ) == [
        "Saat 20:00'den sonra varış prosedürü nedir?",
        "What is the late-arrival procedure after 20:00?",
    ]


def test_turkish_generation_produces_direct_turkish_answer(tmp_path):
    """Turkish locale now generates directly in Turkish (no round-trip translation)."""

    class TurkishRuntime(FakeRuntime):
        def generate(self, messages):
            assert "Write the final answer in Turkish." in messages[0]["content"]
            return "+90 312 555 0149 numaralı telefonu arayın [1]."

    translation = ControlledTranslationService(FakeTranslationEngine())
    pipeline = RAGPipeline(
        TracefoldStore(tmp_path / "tracefold.db"),
        TurkishRuntime(),
        translation,
    )
    result = RetrievalResult(
        chunk={
            "filename": "travel.md",
            "text": "Call +90 312 555 0149 before 19:30.",
            "page_start": None,
            "section": "Late arrivals",
        },
        score=0.9,
    )
    citations = [
        {
            "index": 1,
            "filename": "travel.md",
            "text": "Call +90 312 555 0149 before 19:30.",
        }
    ]

    answer, used = pipeline._generate_validated("Ne yapmalıyım?", [result], citations, locale="tr")

    assert "+90 312 555 0149" in answer
    assert used == citations


def test_turkish_quality_gate_catches_known_malformed_patterns():
    assert RAGPipeline.turkish_quality_issues("+90 312 555 0149'yi arayın [1].") == [
        "suffix_on_number"
    ]
    assert "known_malformed_word" in RAGPipeline.turkish_quality_issues(
        "Evetkin zamanınızı bildirin [1]."
    )
    assert (
        RAGPipeline.turkish_quality_issues("+90 312 555 0149 numaralı telefonu arayın [1].") == []
    )
    assert (
        RAGPipeline.turkish_quality_issues("Program 2026'da başlayacak; 19:30'dan önce gelin [1].")
        == []
    )
    assert "english_role_leak" in RAGPipeline.turkish_quality_issues(
        "Residence Duty Phone ile iletişime geçin [1]."
    )
    assert "unexpected_script" in RAGPipeline.turkish_quality_issues(
        "Yanıtın içine başka bir yazı sistemi sızdı 回答 [1]."
    )
    assert "known_malformed_phrase" in RAGPipeline.turkish_quality_issues(
        "North Lawn yanında mavi ışık bulutu bulunur [1]."
    )
    assert "english_language_leak" in RAGPipeline.turkish_quality_issues(
        "Call the office and return the laptop before 17:30 [1]."
    )


def test_explicit_revision_is_dirty_unless_clean_state_is_explicit(monkeypatch):
    monkeypatch.setenv("TRACEFOLD_CODE_REVISION", "abc123")
    monkeypatch.delenv("TRACEFOLD_CODE_DIRTY", raising=False)
    assert Evaluator._code_provenance() == {
        "revision": "abc123",
        "dirty": True,
        "source": "environment",
    }

    monkeypatch.setenv("TRACEFOLD_CODE_DIRTY", "false")
    assert Evaluator._code_provenance()["dirty"] is False


def test_release_evaluation_command_fails_closed():
    with pytest.raises(RuntimeError, match="23/24 cases passed"):
        require_complete_evaluation({"metrics": {"passed_cases": 23, "total_cases": 24}})
    require_complete_evaluation({"metrics": {"passed_cases": 24, "total_cases": 24}})


def test_evaluator_requires_full_source_recall_and_citation_precision():
    class StubRuntime:
        def embed(self, texts):
            return [[1.0] for _text in texts]

        def status(self, *, probe=False):
            return {"runtime": "stub", "ready": True}

    class StubRag:
        threshold = 0.6
        evidence_floor = 0.3
        max_chunks_per_document = 2
        runtime = StubRuntime()

        class StubTranslation:
            round_trip_threshold = 0.6

            class Glossary:
                id = "stub-glossary"

            glossary = Glossary()

            def status(self, *, probe=False):
                return {"runtime": "stub", "ready": True}

        translation = StubTranslation()

        def __init__(self, response):
            self.response = response

        def answer(self, question, *, top_k, locale):
            return self.response

        @staticmethod
        def citation_sentence_coverage(answer, citations):
            return True

        @staticmethod
        def protected_facts_grounded(answer, citations):
            return True

        @staticmethod
        def turkish_quality_issues(answer):
            return []

    class StubStore:
        def __init__(self):
            self.saved = None

        def list_documents(self):
            return [{"filename": "one.md"}, {"filename": "two.md"}]

        def counts(self):
            return {"documents": 2, "chunks": 2}

        def save_evaluation(self, metrics, cases):
            self.saved = {"metrics": metrics, "cases": cases}
            return "eval_stub"

        def latest_evaluation(self):
            return {"created_at": "2026-07-18T00:00:00Z"}

    base_response = {
        "answer": "Grounded answer [1].",
        "matches": [{"filename": "one.md"}],
        "citations": [{"index": 1, "filename": "one.md", "text": "Grounded answer."}],
        "refused": False,
        "retrieval_ms": 1.0,
    }
    case = {
        "id": "strict-source-gate",
        "question": "Question?",
        "locale": "en",
        "expected_sources": ["one.md", "two.md"],
    }
    partial = Evaluator(StubRag(base_response), StubStore()).run([case])
    assert partial["cases"][0]["source_recall"] == 0.5
    assert partial["cases"][0]["passed"] is False

    imprecise_response = {
        **base_response,
        "matches": [{"filename": "one.md"}, {"filename": "two.md"}],
        "citations": [
            {"index": 1, "filename": "one.md", "text": "Grounded answer."},
            {"index": 2, "filename": "unexpected.md", "text": "Adjacent answer."},
        ],
    }
    imprecise = Evaluator(StubRag(imprecise_response), StubStore()).run([case])
    assert imprecise["cases"][0]["citation_precision"] == 0.5
    assert imprecise["cases"][0]["passed"] is False

    incomplete_fact_case = {
        "id": "strict-required-fact-gate",
        "question": "Which requirements apply?",
        "locale": "en",
        "expected_sources": ["one.md"],
        "required_facts": ["Grounded answer", "Wi-Fi USB installation rights"],
    }
    incomplete = Evaluator(StubRag(base_response), StubStore()).run([incomplete_fact_case])
    assert incomplete["cases"][0]["required_fact_complete"] is False
    assert incomplete["cases"][0]["passed"] is False


def test_checked_in_evaluation_dataset_resolves_demo_sources():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    payload = load_evaluation_dataset(
        Path(project_root) / "docs" / "eval-dataset.json",
        Path(project_root) / "demo_docs",
    )
    assert len(payload["cases"]) == 24
    assert payload["dataset_id"] == "northstar-summer-lab-2026-bilingual-parity-v3"
    assert all(case.get("locale") in {"en", "tr"} for case in payload["cases"])
    assert sum(case.get("locale", "en") == "tr" for case in payload["cases"]) == 12
    assert all(
        case.get("paired_case_id") for case in payload["cases"] if case.get("locale") == "tr"
    )
    paired_targets = [
        case["paired_case_id"] for case in payload["cases"] if case.get("paired_case_id")
    ]
    assert len(paired_targets) == len(set(paired_targets)) == 12
    assert payload["resolved_sources"]["01-program-handbook.md"].endswith(
        "demo_docs/01-program-handbook.md"
    )
