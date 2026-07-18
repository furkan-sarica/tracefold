"""Record an explicit case-by-case Turkish language review on a release evaluation.

Run this only after reading every Turkish question and answer in the generated
report. The annotation is written to both the JSON evidence file and the matching
local SQLite evaluation so the Quality screen and release artifacts agree.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from storage import TracefoldStore, review_answers_sha256  # noqa: E402


def turkish_review_cases(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        case
        for case in report.get("cases") or []
        if str(case.get("locale") or "").casefold().startswith("tr")
    ]


def annotate_release_review(
    report: dict[str, Any],
    *,
    reviewer: str,
    reviewed_at: str,
    notes: str = "",
) -> dict[str, Any]:
    """Validate release evidence and attach a reproducible manual-review record."""

    reviewer = reviewer.strip()
    if len(reviewer) < 3:
        raise ValueError("A specific reviewer label is required.")
    try:
        parsed_reviewed_at = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("reviewed_at must be an ISO-8601 timestamp.") from exc
    if parsed_reviewed_at.tzinfo is None or parsed_reviewed_at.utcoffset() is None:
        raise ValueError("reviewed_at must include a timezone offset.")

    metrics = report.get("metrics") or {}
    total = int(metrics.get("total_cases") or 0)
    passed = int(metrics.get("passed_cases") or 0)
    if report.get("status") != "complete" or total != 24 or passed != total:
        raise ValueError("Manual approval requires a complete 24/24 release evaluation.")

    provenance = (metrics.get("run_context") or {}).get("code_provenance") or {}
    if not provenance.get("revision") or provenance.get("dirty") is not False:
        raise ValueError("Manual approval requires a clean, identified code revision.")

    cases = turkish_review_cases(report)
    if len(cases) != 12:
        raise ValueError("Manual approval requires exactly 12 Turkish release cases.")
    if any(case.get("passed") is not True for case in cases):
        raise ValueError("Every Turkish case must pass before manual approval.")
    if any(not str(case.get("question") or "").strip() for case in cases):
        raise ValueError("Every Turkish case must include its reviewed question.")
    if any(not str(case.get("answer") or "").strip() for case in cases):
        raise ValueError("Every Turkish case must include its reviewed answer.")

    digest = review_answers_sha256(cases)
    review = {
        "status": "approved",
        "verified": True,
        "locale": "tr",
        "reviewer": reviewer,
        "reviewer_type": "case-by-case release audit",
        "reviewed_at": reviewed_at,
        "cases_reviewed": len(cases),
        "answers_sha256": digest,
        "criteria": [
            "natural Turkish grammar and word order",
            "complete meaning and procedure order",
            "exact names, dates, times, numbers, phone numbers, and email addresses",
            "natural terminology and suffixes",
            "matching evidence citations or deterministic grounded refusal",
        ],
    }
    if notes.strip():
        review["notes"] = notes.strip()
    metrics.setdefault("run_context", {})["manual_language_review"] = review
    return review


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-json",
        type=Path,
        default=PROJECT_ROOT / "docs" / "evaluation-report.json",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.getenv("TRACEFOLD_DATA_DIR", PROJECT_ROOT / "data")),
    )
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--notes", default="")
    parser.add_argument(
        "--approve",
        action="store_true",
        help="Confirm that every printed Turkish answer was reviewed against the criteria.",
    )
    args = parser.parse_args()

    report_path = args.report_json.expanduser().resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    cases = turkish_review_cases(report)
    for case in cases:
        print(f"\n{case.get('id')}\nSoru: {case.get('question')}\nYanıt: {case.get('answer')}")
    if not args.approve:
        raise SystemExit(
            "Review every Turkish answer above, then rerun with --approve to record the decision."
        )

    reviewed_at = datetime.now(UTC).isoformat()
    review = annotate_release_review(
        report,
        reviewer=args.reviewer,
        reviewed_at=reviewed_at,
        notes=args.notes,
    )
    data_dir = args.data_dir.expanduser().resolve()
    database_path = data_dir / "tracefold.db"
    if not database_path.is_file():
        raise SystemExit(f"Evaluation database does not exist: {database_path}")
    store = TracefoldStore(database_path)
    run_id = str(report.get("id") or "")
    if not store.update_evaluation_metrics(run_id, report["metrics"]):
        raise SystemExit(f"Evaluation run is not present in the database: {run_id}")

    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(report_path)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "run_id": run_id,
                "review": review,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
