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
    TranslationRejected,
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

    unrelated = "The moon is made of cheese [1]."
    assert RAGPipeline.validate_generated_answer(unrelated, citations) == (REFUSAL, [])


def test_model_answer_normalization_keeps_plain_text_and_readable_citations():
    assert (
        RAGPipeline._normalize_model_answer("Call **+90 312 555 0155** now.[1][2]")
        == "Call +90 312 555 0155 now. [1] [2]"
    )
    assert (
        RAGPipeline._normalize_model_answer(
            "<think>Draft reasoning that must stay private.</think> Check out by 10:00 [1]."
        )
        == "Check out by 10:00 [1]."
    )
    assert RAGPipeline._normalize_model_answer("<think>unfinished private reasoning") == ""


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

    # A shared [1][2] at paragraph end is valid when both sources corroborate both claims.
    shared_citation = "Check out by 10:00. Store luggage until 12:15 [1][2]."
    assert RAGPipeline.validate_generated_answer(shared_citation, citations) == (
        shared_citation,
        citations,
    )

    # But a shared citation to a single source is rejected when a second source exists.
    wrong_shared = "Check out by 10:00. Store luggage until 12:15 [1]."
    assert RAGPipeline.validate_generated_answer(wrong_shared, citations) == (REFUSAL, [])

    hallucinated_shared_citation = "The moon is made of cheese. Check out by 10:00 [1]."
    assert RAGPipeline.validate_generated_answer(hallucinated_shared_citation, citations) == (
        REFUSAL,
        [],
    )


def test_generated_answer_rejects_more_than_four_claims():
    citations = [{"index": 1, "filename": "guide.md", "text": "One. Two. Three. Four. Five."}]

    answer = "One. [1]\nTwo. [1]\nThree. [1]\nFour. [1]\nFive. [1]"

    assert RAGPipeline.validate_generated_answer(answer, citations) == (REFUSAL, [])


def test_narrow_question_scope_limits_adjacent_claims():
    assert RAGPipeline._scope_claim_limit("When is the drill and where should we assemble?") == 1
    assert (
        RAGPipeline._scope_claim_limit("How do checkout and luggage storage work for the shuttle?")
        == 2
    )
    assert RAGPipeline._scope_claim_limit("What must I return and by when?") is None
    assert (
        RAGPipeline._scope_claim_limit(
            "What should I return with the loan laptop and how does its deadline relate to the "
            "team handoff?"
        )
        == 1
    )


def test_single_claim_scope_repair_demands_one_cited_sentence(tmp_path):
    class NarrowRuntime(FakeRuntime):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def generate(self, messages):
            self.calls += 1
            if self.calls == 1:
                return "The drill is Tuesday at 11:30 [1]. People assemble at North Lawn [1]."
            assert "EXACTLY ONE short sentence" in messages[-1]["content"]
            assert "do not use a period" in messages[-1]["content"]
            assert "EVERY evidence number" in messages[-1]["content"]
            assert "[1][2]" in messages[-1]["content"]
            return "The drill is Tuesday at 11:30, and people assemble at North Lawn [1][2]."

    schedule_evidence = "The drill is Tuesday at 11:30."
    location_evidence = "People assemble at North Lawn."
    schedule_result = RetrievalResult(
        chunk={
            "filename": "safety.md",
            "text": schedule_evidence,
            "page_start": None,
            "section": "Drill",
        },
        score=0.9,
    )
    location_result = RetrievalResult(
        chunk={
            "filename": "safety.md",
            "text": location_evidence,
            "page_start": None,
            "section": "Assembly",
        },
        score=0.8,
    )
    citations = [
        {"index": 1, "filename": "safety.md", "text": schedule_evidence},
        {"index": 2, "filename": "safety.md", "text": location_evidence},
    ]
    runtime = NarrowRuntime()
    pipeline = RAGPipeline(TracefoldStore(tmp_path / "tracefold.db"), runtime)

    answer, used = pipeline._generate_validated(
        "When is the drill and where should people assemble?",
        [schedule_result, location_result],
        citations,
    )

    assert runtime.calls == 2
    assert answer == "The drill is Tuesday at 11:30, and people assemble at North Lawn [1] [2]."
    assert used == citations


def test_single_scope_can_preserve_one_model_supplied_trailing_citation() -> None:
    answer = "The drill is Tuesday at 11:30. People assemble at North Lawn [1][2]."

    assert RAGPipeline._merge_trailing_citation_sentences(answer) == (
        "The drill is Tuesday at 11:30; People assemble at North Lawn [1] [2]."
    )


def test_narrow_scope_requires_one_supporting_citation_per_question_facet() -> None:
    citations = [
        {"index": 1, "text": "The evacuation drill is Tuesday at 11:30."},
        {"index": 2, "text": "The Orion Building assembly point is North Lawn."},
        {"index": 3, "text": "Unrelated medical support."},
    ]

    assert RAGPipeline._required_scope_citation_indices(
        "When is the evacuation drill and where should Orion Building occupants assemble?",
        citations,
    ) == {1, 2}
    assert (
        RAGPipeline._merge_trailing_citation_sentences(
            "The drill is Tuesday. People assemble at North Lawn [2]."
        )
        == "The drill is Tuesday. People assemble at North Lawn [2]."
    )
    assert (
        RAGPipeline._merge_trailing_citation_sentences(
            "The drill is Tuesday [1]. People assemble at North Lawn [2]."
        )
        == "The drill is Tuesday [1]. People assemble at North Lawn [2]."
    )


def test_device_specification_completeness_requires_every_evidence_category():
    question = "What specifications should my laptop have?"
    citations = [
        {
            "index": 1,
            "filename": "equipment.md",
            "text": (
                "The laptop needs 16 GB RAM, 60 GB free storage, Wi-Fi 5 or newer, "
                "a working browser, a USB-C or USB-A port, and permission to install "
                "software and local development tools."
            ),
        }
    ]

    incomplete = "The laptop needs 16 GB RAM, 60 GB free storage, and Wi-Fi 5 [1]."
    complete = (
        "The laptop needs 16 GB RAM, 60 GB free storage, Wi-Fi 5 or newer, a working "
        "browser, a USB-C or USB-A port, and permission to install software and local "
        "development tools [1]."
    )
    category_only = (
        "The laptop needs RAM, storage, Wi-Fi, a browser, USB ports, and permission to install "
        "software [1]."
    )

    assert not RAGPipeline._specification_requirements_complete(
        question,
        incomplete,
        citations,
    )
    assert RAGPipeline._specification_requirements_complete(question, complete, citations)
    assert not RAGPipeline._specification_requirements_complete(
        question,
        category_only,
        citations,
    )
    assert not RAGPipeline._specification_requirements_complete(
        question,
        complete + " Complete all operating-system updates before arrival [1].",
        citations,
    )
    assert RAGPipeline._specification_requirements_complete(
        "When must I return the loan laptop?",
        incomplete,
        citations,
    )


@pytest.mark.parametrize(
    "question",
    [
        "What laptop specifications should I prepare for the lab?",
        "What features should the laptop I am preparing for the lab have?",
    ],
)
def test_device_spec_excerpt_filters_adjacent_advice_even_under_budget(question):
    evidence = (
        "Bring a laptop that can run a current 64-bit operating system and that meets these "
        "recommended minimums:\n\n"
        "- **16 GB RAM**\n"
        "- **60 GB free storage** after system updates\n"
        "- Wi-Fi 5 or newer\n"
        "- A working browser and USB-C or USB-A port\n"
        "- Permission to install software and local development tools\n\n"
        "Complete operating-system updates before arrival. Participants using an employer- "
        "or school-managed device should confirm they have local installation rights. A tablet "
        "alone is not sufficient for the project work."
    )

    excerpt = RAGPipeline._focused_excerpt(question, evidence, 800)

    assert "16 GB RAM" in excerpt
    assert "60 GB free storage** after system updates" in excerpt
    assert "Wi-Fi 5 or newer" in excerpt
    assert "working browser and USB-C or USB-A port" in excerpt
    assert "Permission to install software and local development tools" in excerpt
    assert "Complete operating-system updates" not in excerpt
    assert "employer- or school-managed" not in excerpt
    assert "tablet" not in excerpt.casefold()


def test_non_spec_question_keeps_preparation_evidence():
    evidence = "Complete operating-system updates before arrival."

    assert (
        RAGPipeline._focused_excerpt(
            "When should I complete operating-system updates?",
            evidence,
            800,
        )
        == evidence
    )


def test_spec_excerpt_preserves_a_requirement_that_has_a_before_arrival_prefix():
    evidence = "Before arrival, your laptop must have 16 GB RAM."

    excerpt = RAGPipeline._focused_excerpt(
        "What laptop specifications should I prepare for the lab?",
        evidence,
        800,
    )

    assert excerpt == "your laptop must have 16 GB RAM."


def test_device_specification_requires_storage_update_qualifier_and_split_chunks():
    question = "What specifications should my laptop have?"
    citations = [
        {
            "index": 1,
            "text": (
                "The laptop needs 16 GB RAM, 60 GB free storage after system updates, and "
                "Wi-Fi 5 or newer."
            ),
        },
        {
            "index": 2,
            "text": (
                "The laptop needs a working browser, USB-C or USB-A ports, and permission to "
                "install software and local development tools."
            ),
        },
    ]
    complete = (
        "The laptop needs 16 GB RAM and 60 GB free storage after system updates [1].\n"
        "It needs Wi-Fi 5 or newer, a working browser, and USB-C or USB-A ports [1][2].\n"
        "You need permission to install software and local development tools [2]."
    )
    missing_update_relation = complete.replace(" after system updates", "")
    missing_second_chunk = (
        "The laptop needs 16 GB RAM and 60 GB free storage after system updates [1].\n"
        "It needs Wi-Fi 5 or newer [1]."
    )

    assert RAGPipeline._specification_requirements_complete(question, complete, citations)
    # "after system updates" qualifier is not required when the numeric value and "free storage" are present
    assert RAGPipeline._specification_requirements_complete(
        question,
        missing_update_relation,
        citations,
    )
    assert not RAGPipeline._specification_requirements_complete(
        question,
        missing_second_chunk,
        citations,
    )


def test_incomplete_device_specification_answer_gets_one_grounded_repair(tmp_path):
    class SpecificationRuntime(FakeRuntime):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def generate(self, messages):
            self.calls += 1
            if self.calls == 1:
                return "The laptop needs 16 GB RAM, 60 GB free storage, and Wi-Fi 5 [1]."
            assert "include every listed memory" in messages[-1]["content"]
            return (
                "The laptop needs 16 GB RAM and 60 GB free storage [1].\n"
                "It needs Wi-Fi 5 or newer and a working browser [1].\n"
                "It needs a USB-C or USB-A port [1].\n"
                "You need permission to install software and local development tools [1]."
            )

    evidence = (
        "The laptop needs 16 GB RAM and 60 GB free storage. It needs Wi-Fi 5 or newer, "
        "a working browser, and a USB-C or USB-A port. You need permission to install "
        "software and local development tools."
    )
    result = RetrievalResult(
        chunk={
            "filename": "equipment.md",
            "text": evidence,
            "page_start": None,
            "section": "Participant laptop",
        },
        score=0.9,
    )
    citations = [{"index": 1, "filename": "equipment.md", "text": evidence}]
    runtime = SpecificationRuntime()
    pipeline = RAGPipeline(TracefoldStore(tmp_path / "tracefold.db"), runtime)

    answer, used = pipeline._generate_validated(
        "What specifications should my laptop have?",
        [result],
        citations,
    )

    assert runtime.calls == 2
    assert "working browser" in answer
    assert "USB-C or USB-A port" in answer
    assert "permission to install software" in answer
    assert used == citations


def test_device_specification_repair_fails_closed_when_still_incomplete(tmp_path):
    class StillIncompleteRuntime(FakeRuntime):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def generate(self, _messages):
            self.calls += 1
            return "The laptop needs 16 GB RAM, 60 GB free storage, and Wi-Fi 5 [1]."

    evidence = (
        "The laptop needs 16 GB RAM, 60 GB free storage, Wi-Fi 5, a working browser, "
        "a USB-C port, and permission to install software."
    )
    result = RetrievalResult(
        chunk={
            "filename": "equipment.md",
            "text": evidence,
            "page_start": None,
            "section": "Participant laptop",
        },
        score=0.9,
    )
    citations = [{"index": 1, "filename": "equipment.md", "text": evidence}]
    runtime = StillIncompleteRuntime()
    pipeline = RAGPipeline(TracefoldStore(tmp_path / "tracefold.db"), runtime)

    assert pipeline._generate_validated(
        "What specifications should my laptop have?",
        [result],
        citations,
    ) == (REFUSAL, [])
    assert runtime.calls == 2


def test_exact_emergency_conduct_contacts_and_order_are_required():
    question = (
        "If threatening conduct creates immediate danger, what must I do first, and how do I "
        "report the conduct afterward? Include the exact contacts."
    )
    citations = [
        {
            "index": 1,
            "filename": "conduct.md",
            "text": (
                "Email the Conduct Lead at conduct@northstar.example. In an immediate threat, "
                "follow the Safety Guide and use the campus emergency number first."
            ),
        },
        {
            "index": 2,
            "filename": "safety.md",
            "text": (
                "In an immediate threat, the emergency procedure takes priority over confidential "
                "conduct reporting: call **5555** from a campus phone or **+90 312 555 0155** "
                "from a mobile before making a conduct report."
            ),
        },
    ]
    complete = (
        "Apply the emergency procedure before the confidential conduct report [2].\n"
        "Call campus emergency extension 5555 from a campus phone or +90 312 555 0155 from a "
        "mobile [2].\n"
        "After immediate safety is addressed, email the conduct concern to "
        "conduct@northstar.example for the Conduct Lead [1][2]."
    )
    vague = (
        "Use the campus emergency number first [2].\n"
        "After immediate safety is addressed, email conduct@northstar.example for the Conduct "
        "Lead [1][2]."
    )
    reversed_order = (
        "Apply the emergency procedure before the confidential conduct report [2].\n"
        "After immediate safety is addressed, email the conduct concern to "
        "conduct@northstar.example for the Conduct Lead [1][2].\n"
        "Call campus emergency extension 5555 from a campus phone or +90 312 555 0155 from a "
        "mobile [2]."
    )
    phone_before_priority = (
        "Call campus emergency extension 5555 from a campus phone or +90 312 555 0155 from a "
        "mobile [2].\n"
        "Apply the emergency procedure before the confidential conduct report [2].\n"
        "After immediate safety is addressed, email the conduct concern to "
        "conduct@northstar.example for the Conduct Lead [1][2]."
    )

    assert RAGPipeline._emergency_conduct_requirements_complete(
        question,
        complete,
        citations,
    )
    assert not RAGPipeline._emergency_conduct_requirements_complete(
        question,
        vague,
        citations,
    )
    assert not RAGPipeline._emergency_conduct_requirements_complete(
        question,
        reversed_order,
        citations,
    )
    assert not RAGPipeline._emergency_conduct_requirements_complete(
        question,
        phone_before_priority,
        citations,
    )


def test_emergency_contact_extraction_ignores_unrelated_contacts_in_the_same_chunks():
    question = (
        "For immediate threatening behavior, what phone numbers and email should I use first and "
        "then to report it?"
    )
    citations = [
        {
            "index": 1,
            "text": (
                "Email the Conduct Lead at conduct@northstar.example. "
                "Email Participant Support at support@northstar.example for general help."
            ),
        },
        {
            "index": 2,
            "text": (
                "In an immediate threat, the emergency procedure takes priority over confidential "
                "conduct reporting: call 5555 from a campus phone or +90 312 555 0155 from a "
                "mobile before making a conduct report. Call the non-emergency Security Desk at "
                "+90 312 555 0199 for access questions."
            ),
        },
    ]

    facts = RAGPipeline._emergency_conduct_contact_facts(question, citations)

    assert facts == {
        "extensions": [("5555", 2)],
        "phones": [("+90 312 555 0155", 2)],
        "emails": [("conduct@northstar.example", 1)],
    }


def test_emergency_contact_extraction_is_role_scoped_within_one_sentence():
    question = (
        "For immediate threatening behavior, what phone numbers and email should I use first and "
        "then to report it?"
    )
    citations = [
        {
            "index": 1,
            "text": (
                "Email the Conduct Lead at conduct@northstar.example or email Participant Support "
                "at support@northstar.example for general help."
            ),
        },
        {
            "index": 2,
            "text": (
                "In an immediate threat, the emergency procedure takes priority over confidential "
                "reporting: call 5555 from a campus phone or +90 312 555 0155 from a mobile before "
                "making a conduct report, but call +90 312 555 0199 for non-emergency access help."
            ),
        },
    ]

    facts = RAGPipeline._emergency_conduct_contact_facts(question, citations)

    assert facts == {
        "extensions": [("5555", 2)],
        "phones": [("+90 312 555 0155", 2)],
        "emails": [("conduct@northstar.example", 1)],
    }


def test_incomplete_emergency_contacts_get_one_grounded_repair(tmp_path):
    class ContactRuntime(FakeRuntime):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def generate(self, messages):
            self.calls += 1
            if self.calls == 1:
                return (
                    "Use the campus emergency number first [2].\n"
                    "After immediate safety is addressed, email conduct@northstar.example for "
                    "the Conduct Lead [1][2]."
                )
            repair = messages[-1]["content"]
            assert "Use exactly three separate cited sentences" in repair
            assert "5555" in repair
            assert "+90 312 555 0155" in repair
            assert "conduct@northstar.example" in repair
            return (
                "Apply the emergency procedure before the confidential conduct report [2].\n"
                "Call campus emergency extension 5555 from a campus phone or "
                "+90 312 555 0155 from a mobile [2].\n"
                "After immediate safety is addressed, email the conduct concern to "
                "conduct@northstar.example for the Conduct Lead [1][2]."
            )

    question = (
        "If threatening conduct creates immediate danger, what must I do first, and how do I "
        "report the conduct afterward? Include the exact contacts."
    )
    conduct = (
        "Email the Conduct Lead at conduct@northstar.example. In an immediate threat, follow the "
        "Safety Guide and use the campus emergency number first."
    )
    safety = (
        "In an immediate threat, the emergency procedure takes priority over confidential conduct "
        "reporting: call **5555** from a campus phone or **+90 312 555 0155** from a mobile before "
        "making a conduct report."
    )
    grounded = [
        RetrievalResult(
            chunk={
                "filename": "conduct.md",
                "text": conduct,
                "page_start": None,
                "section": "Reporting",
            },
            score=0.9,
        ),
        RetrievalResult(
            chunk={
                "filename": "safety.md",
                "text": safety,
                "page_start": None,
                "section": "Emergency",
            },
            score=0.8,
        ),
    ]
    citations = [
        {"index": 1, "filename": "conduct.md", "text": conduct},
        {"index": 2, "filename": "safety.md", "text": safety},
    ]
    runtime = ContactRuntime()
    pipeline = RAGPipeline(TracefoldStore(tmp_path / "tracefold.db"), runtime)

    answer, used = pipeline._generate_validated(question, grounded, citations)

    assert runtime.calls == 2
    assert "extension 5555" in answer
    assert "conduct@northstar.example" in answer
    assert used == citations


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


def test_numeric_hardware_roles_cannot_be_swapped_under_a_valid_citation():
    citations = [
        {
            "index": 1,
            "filename": "equipment.md",
            "text": "The laptop requires 16 GB RAM and 60 GB free storage.",
        }
    ]

    swapped = "The laptop requires 60 GB RAM and 16 GB storage [1]."

    assert RAGPipeline.validate_generated_answer(swapped, citations) == (REFUSAL, [])


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
    assert RAGPipeline.validate_generated_answer(
        f"{REFUSAL_TR} [2]",
        [{"index": 2, "text": "irrelevant"}],
        locale="tr",
    ) == (REFUSAL_TR, [])
    assert RAGPipeline.validate_generated_answer(
        f"{REFUSAL} [1]",
        [{"index": 1, "text": "irrelevant"}],
    ) == (REFUSAL, [])


def test_turkish_retrieval_keeps_original_and_adds_local_search_translation(tmp_path):
    class QueryTranslationEngine(BaseTranslationEngine):
        def translate_batch(self, texts, *, source_locale, target_locale):
            assert source_locale == "tr"
            assert target_locale == "en"
            assert texts == ["Saat ZXQ0QXZ'den sonra varış prosedürü nedir?"]
            return ["What is the late-arrival procedure after ZXQ0QXZ?"]

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


def test_turkish_generation_translates_validated_claims_and_reconstructs_citations(tmp_path):
    class EnglishRuntime(FakeRuntime):
        def generate(self, messages):
            assert "Write the final answer in English." in messages[0]["content"]
            return "Call +90 312 555 0149 before 19:30 [1]."

    class ClaimTranslationEngine(BaseTranslationEngine):
        def translate_batch(self, texts, *, source_locale, target_locale):
            if (source_locale, target_locale) == ("en", "tr"):
                assert texts == ["Call ZXQ0QXZ before 19:30"]
                return ["19:30'dan önce ZXQ0QXZ'yi arayın"]
            assert (source_locale, target_locale) == ("tr", "en")
            return ["Call +90 312 555 0149 before 19:30"]

        def status(self, *, probe=False):
            return {"runtime": "claim-test", "ready": True}

    translation = ControlledTranslationService(ClaimTranslationEngine())
    pipeline = RAGPipeline(
        TracefoldStore(tmp_path / "tracefold.db"),
        EnglishRuntime(),
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

    assert answer == "19:30'dan önce +90 312 555 0149 numaralı telefonu arayın. [1]"
    assert used == citations


def test_turkish_generation_gets_one_grounded_repair_after_translation_rejection(tmp_path):
    class RepairRuntime(FakeRuntime):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def generate(self, messages):
            self.calls += 1
            if self.calls == 1:
                return "Call +90 312 555 0149 before 19:30 [1]. Read the guide [1]."
            assert "only the minimum facts" in messages[-1]["content"]
            return "Call +90 312 555 0149 before 19:30 [1]."

    class RejectOnceTranslation:
        def __init__(self):
            self.calls = 0

        @staticmethod
        def atomize_claim(claim):
            return [claim]

        def claims_to_turkish(self, claims):
            self.calls += 1
            if self.calls == 1:
                raise TranslationRejected("compound optional claim")
            assert claims == ["Call +90 312 555 0149 before 19:30"]
            return ["19:30'dan önce +90 312 555 0149 numaralı telefonu arayın"]

    runtime = RepairRuntime()
    translation = RejectOnceTranslation()
    pipeline = RAGPipeline(
        TracefoldStore(tmp_path / "tracefold.db"),
        runtime,
        translation,
    )
    result = RetrievalResult(
        chunk={
            "filename": "travel.md",
            "text": "Call +90 312 555 0149 before 19:30. Read the guide.",
            "page_start": None,
            "section": "Late arrivals",
        },
        score=0.9,
    )
    citations = [
        {
            "index": 1,
            "filename": "travel.md",
            "text": "Call +90 312 555 0149 before 19:30. Read the guide.",
        }
    ]

    answer, used = pipeline._generate_validated(
        "Ne yapmalıyım?",
        [result],
        citations,
        locale="tr",
        focus_question="What should I do?",
    )

    assert runtime.calls == 2
    assert translation.calls == 2
    assert answer == "19:30'dan önce +90 312 555 0149 numaralı telefonu arayın. [1]"
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
        def citation_sentence_coverage(answer, citations, *, locale="en"):
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


def test_required_fact_recall_accepts_clear_ordering_paraphrases():
    fact = "The immediate emergency procedure takes priority over confidential conduct reporting."
    answer = (
        "Follow the emergency procedure first. After addressing the immediate threat, report the "
        "conduct concern confidentially."
    )

    assert Evaluator._required_fact_recall(answer, fact) >= 0.65
    assert (
        Evaluator._required_fact_recall(
            "Confidential conduct reporting takes priority over the immediate emergency procedure.",
            fact,
        )
        == 0.0
    )

    required_order = [
        {
            "before_all": ["5555", "+90 312 555 0155"],
            "after_all": ["conduct@northstar.example"],
            "before_actions": ["call", "ara"],
            "after_actions": ["email", "report", "send", "e-posta", "bildir", "gönder"],
        }
    ]
    ordered = "Call 5555 or +90 312 555 0155 first. Then email conduct@northstar.example."
    reversed_order = "Email conduct@northstar.example first. Then call 5555 or +90 312 555 0155."
    assert Evaluator._required_order_complete(ordered, required_order) is True
    assert Evaluator._required_order_complete(reversed_order, required_order) is False

    misleading = (
        "The immediate emergency procedure lists 5555 and +90 312 555 0155. "
        "Confidential conduct reporting takes priority over the immediate emergency procedure. "
        "First email conduct@northstar.example; then call 5555 or +90 312 555 0155."
    )
    assert Evaluator._required_fact_recall(misleading, fact) == 0.0
    assert Evaluator._required_order_complete(misleading, required_order) is False
