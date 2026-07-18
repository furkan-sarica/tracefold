"""End-to-end API contracts for the Tracefold backend."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["TRACEFOLD_RUNTIME"] = "fake"
os.environ["TRACEFOLD_RETRIEVAL_THRESHOLD"] = "0.12"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from fastapi.testclient import TestClient
from main import create_app, safe_filename
from rag import REFUSAL, REFUSAL_TR
from runtime import FakeRuntime


def make_client(tmp_path) -> TestClient:
    return TestClient(create_app(data_dir=tmp_path / "data", runtime=FakeRuntime()))


def upload_guide(client: TestClient):
    content = (
        "Foundry Local runs language and embedding models directly on the user's device.\n\n"
        "Tracefold retrieves relevant document chunks and cites every grounded answer."
    )
    return client.post(
        "/api/documents",
        files={"file": ("local-guide.txt", content, "text/plain")},
    )


def test_health_and_runtime_contract(tmp_path):
    client = make_client(tmp_path)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["service"] == "Tracefold"
    assert health.json()["documents"] == 0

    runtime = client.get("/api/runtime")
    assert runtime.status_code == 200
    assert runtime.json()["runtime"] == "fake"
    assert runtime.json()["embedding_model"]["variant_id"] == "fake-embedding-v1"
    assert runtime.json()["available"] is True


def test_document_lifecycle_job_and_grounded_chat(tmp_path):
    client = make_client(tmp_path)
    uploaded = upload_guide(client)
    assert uploaded.status_code == 201, uploaded.text
    payload = uploaded.json()
    assert payload["status"] == "indexed"
    assert payload["document"]["chunk_count"] == 1

    job = client.get(f"/api/jobs/{payload['job_id']}")
    assert job.status_code == 200
    assert job.json()["status"] == "completed"
    assert job.json()["progress"] == 1

    documents = client.get("/api/documents").json()
    assert documents["document_count"] == 1
    assert documents["chunks"] == 1
    assert documents["documents"][0]["filename"] == "local-guide.txt"

    chat = client.post(
        "/api/chat",
        json={"question": "Where does Foundry Local run language models?", "top_k": 5},
    )
    assert chat.status_code == 200, chat.text
    answer = chat.json()
    assert answer["refused"] is False
    assert "device" in answer["answer"].lower()
    assert answer["citations"][0]["filename"] == "local-guide.txt"
    assert answer["citations"][0]["chunk_id"].startswith("chunk_")
    assert answer["citations"][0]["text"]

    document_id = payload["document"]["id"]
    deleted = client.delete(f"/api/documents/{document_id}")
    assert deleted.status_code == 200
    assert client.get("/api/documents").json()["document_count"] == 0


def test_unicode_turkish_filenames_are_preserved_without_path_components(tmp_path):
    assert safe_filename("../Öğrenci Bilgilendirme Şartları.pdf") == (
        "Öğrenci_Bilgilendirme_Şartları.pdf"
    )
    assert safe_filename(r"..\Öğrenci Bilgilendirme Şartları.pdf") == (
        "Öğrenci_Bilgilendirme_Şartları.pdf"
    )
    long_name = ("Öğrenci_Bilgilendirme_" * 20) + "Şartları.markdown"
    shortened = safe_filename(long_name)
    assert len(shortened) <= 180
    assert shortened.endswith(".markdown")

    client = make_client(tmp_path)
    response = client.post(
        "/api/documents",
        files={
            "file": (
                "Öğrenci Bilgilendirme Şartları.txt",
                "Yerel modeller cihaz üzerinde çalışır.",
                "text/plain",
            )
        },
    )
    assert response.status_code == 201, response.text
    assert response.json()["document"]["filename"] == "Öğrenci_Bilgilendirme_Şartları.txt"

    long_upload = client.post(
        "/api/documents",
        files={"file": (long_name, "# Başlık\n\nFarklı Türkçe içerik.", "text/markdown")},
    )
    assert long_upload.status_code == 201, long_upload.text
    assert long_upload.json()["document"]["filename"].endswith(".markdown")


def test_query_alias_and_sse_event_order(tmp_path):
    client = make_client(tmp_path)
    assert upload_guide(client).status_code == 201
    response = client.post(
        "/api/chat/stream",
        json={
            "query": "Tracefold retrieves relevant document chunks and cites grounded answers. How?"
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    text = response.text
    positions = [
        text.index("event: retrieval"),
        text.index("event: token"),
        text.index("event: citations"),
        text.index("event: done"),
    ]
    assert positions == sorted(positions)
    assert "local-guide.txt" in text


def test_refusal_has_no_citations(tmp_path):
    client = make_client(tmp_path)
    assert upload_guide(client).status_code == 201
    response = client.post(
        "/api/chat",
        json={"question": "What is the airspeed velocity of an unladen swallow?"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["refused"] is True
    assert payload["answer"] == REFUSAL
    assert payload["citations"] == []


def test_chat_locale_defaults_to_english_and_validates_supported_values(tmp_path):
    client = make_client(tmp_path)

    default_response = client.post("/api/chat", json={"question": "Bilinmeyen soru"})
    assert default_response.status_code == 200
    assert default_response.json()["answer"] == REFUSAL

    turkish_response = client.post(
        "/api/chat", json={"question": "Bilinmeyen soru", "locale": "tr"}
    )
    assert turkish_response.status_code == 200
    assert turkish_response.json()["answer"] == REFUSAL_TR
    assert turkish_response.json()["refused"] is True

    invalid_response = client.post(
        "/api/chat", json={"question": "Bilinmeyen soru", "locale": "de"}
    )
    assert invalid_response.status_code == 422
    assert invalid_response.json()["detail"][0]["loc"][-1] == "locale"


def test_turkish_stream_refusal_uses_authoritative_metadata(tmp_path):
    client = make_client(tmp_path)
    response = client.post("/api/chat/stream", json={"question": "Bilinmeyen soru", "locale": "tr"})

    assert response.status_code == 200
    assert REFUSAL_TR in response.text
    assert 'event: citations\ndata: {"citations": []}' in response.text
    assert '"refused": true' in response.text


def test_duplicate_validation_and_clear(tmp_path):
    client = make_client(tmp_path)
    assert upload_guide(client).status_code == 201
    duplicate = upload_guide(client)
    assert duplicate.status_code == 409

    unsupported = client.post(
        "/api/documents",
        files={"file": ("notes.csv", "a,b", "text/csv")},
    )
    assert unsupported.status_code == 415
    assert client.delete("/api/documents").json()["deleted"] == 1


def test_evaluation_run_and_latest(tmp_path):
    client = make_client(tmp_path)
    assert upload_guide(client).status_code == 201
    response = client.post(
        "/api/evaluations/run",
        json={
            "top_k": 5,
            "cases": [
                {
                    "id": "answerable-1",
                    "question": "Where does Foundry Local run models?",
                    "answerable": True,
                    "expected_sources": ["local-guide.txt"],
                    "expected_terms": ["device"],
                },
                {
                    "id": "negative-1",
                    "question": "Who won the 1998 World Cup?",
                    "answerable": False,
                },
            ],
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["metrics"]["hit_at_k"] == 1
    assert response.json()["metrics"]["refusal_accuracy"] == 1
    assert response.json()["metrics"]["locales"] == ["en"]
    assert response.json()["metrics"]["locale_breakdown"]["en"]["total_cases"] == 2

    latest = client.get("/api/evaluations/latest")
    assert latest.status_code == 200
    assert latest.json()["id"] == response.json()["id"]


def test_checked_in_fully_paired_bilingual_evaluation_contract(tmp_path):
    client = make_client(tmp_path)
    project_root = Path(__file__).parent.parent
    for path in sorted((project_root / "demo_docs").glob("*.md")):
        response = client.post(
            "/api/documents",
            files={"file": (path.name, path.read_bytes(), "text/markdown")},
        )
        assert response.status_code == 201, response.text

    evaluation = client.post("/api/evaluations/run", json={"top_k": 5})
    assert evaluation.status_code == 200, evaluation.text
    payload = evaluation.json()
    assert payload["dataset_id"] == "northstar-summer-lab-2026-bilingual-parity-v3"
    assert payload["metrics"]["total_cases"] == 24
    assert payload["metrics"]["answerable_cases"] == 16
    assert payload["metrics"]["negative_cases"] == 8
    assert len(payload["cases"]) == 24
    assert payload["metrics"]["locales"] == ["en", "tr"]
    assert payload["metrics"]["locale_breakdown"]["en"]["total_cases"] == 12
    assert payload["metrics"]["locale_breakdown"]["tr"]["total_cases"] == 12
    assert payload["metrics"]["bilingual_pairs_evaluated"] == 12
    assert payload["metrics"]["bilingual_pair_accuracy"] is not None
    assert all(case["locale"] in {"en", "tr"} for case in payload["cases"])
    provenance = payload["metrics"]["run_context"]["code_provenance"]
    assert provenance["revision"]
    assert isinstance(provenance["dirty"], bool)
    for metric in ("hit_at_k", "source_recall_at_k", "citation_precision", "refusal_accuracy"):
        value = payload["metrics"][metric]
        assert value is None or 0 <= value <= 1
