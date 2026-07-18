from __future__ import annotations

import json
from copy import deepcopy

import pytest

from backend.storage import TracefoldStore
from scripts.review_evaluation import annotate_release_review


def release_report() -> dict:
    return {
        "id": "eval_release",
        "status": "complete",
        "metrics": {
            "total_cases": 24,
            "passed_cases": 24,
            "run_context": {
                "code_provenance": {
                    "revision": "abc123",
                    "dirty": False,
                }
            },
        },
        "cases": [
            {
                "id": f"NSL-TR-{index:02d}",
                "locale": "tr",
                "question": f"Soru {index}?",
                "answer": f"Doğal ve kanıtlı yanıt {index}. [1]",
                "passed": True,
            }
            for index in range(1, 13)
        ],
    }


def test_manual_turkish_review_records_reproducible_answer_digest() -> None:
    report = release_report()
    review = annotate_release_review(
        report,
        reviewer="Tracefold release QA",
        reviewed_at="2026-07-18T12:00:00+00:00",
        notes="All cases reviewed against their cited evidence.",
    )

    assert review["status"] == "approved"
    assert review["cases_reviewed"] == 12
    assert len(review["answers_sha256"]) == 64
    assert report["metrics"]["run_context"]["manual_language_review"] == review

    same = annotate_release_review(
        release_report(),
        reviewer="Tracefold release QA",
        reviewed_at="2026-07-18T12:00:00+00:00",
    )
    assert same["answers_sha256"] == review["answers_sha256"]


def test_stored_review_digest_is_recomputed_against_case_answers(tmp_path) -> None:
    report = release_report()
    annotate_release_review(
        report,
        reviewer="Tracefold release QA",
        reviewed_at="2026-07-18T12:00:00+00:00",
    )
    store = TracefoldStore(tmp_path / "tracefold.db")
    run_id = store.save_evaluation(report["metrics"], report["cases"])

    assert (
        store.latest_evaluation()["metrics"]["run_context"]["manual_language_review"]["verified"]
        is True
    )

    tampered = report["cases"]
    tampered[0]["answer"] = "Sonradan değiştirilmiş yanıt."
    with store._connect() as database:
        database.execute(
            "UPDATE evaluation_runs SET cases_json=? WHERE id=?",
            (json.dumps(tampered), run_id),
        )
    assert (
        store.latest_evaluation()["metrics"]["run_context"]["manual_language_review"]["verified"]
        is False
    )


@pytest.mark.parametrize("mutation", ["dirty", "failed_case", "missing_answer"])
def test_manual_turkish_review_fails_closed(mutation: str) -> None:
    report = deepcopy(release_report())
    if mutation == "dirty":
        report["metrics"]["run_context"]["code_provenance"]["dirty"] = True
    elif mutation == "failed_case":
        report["cases"][0]["passed"] = False
    else:
        report["cases"][0]["answer"] = ""

    with pytest.raises(ValueError):
        annotate_release_review(
            report,
            reviewer="Tracefold release QA",
            reviewed_at="2026-07-18T12:00:00+00:00",
        )
