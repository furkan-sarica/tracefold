"""Validate Tracefold's final report, presentation pair, and release checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import unicodedata
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from storage import review_answers_sha256  # noqa: E402

EXPECTED_DATASET = "northstar-summer-lab-2026-bilingual-parity-v3"
EXPECTED_EMBEDDING_MODEL = "qwen3-embedding-0.6b"
EXPECTED_CHAT_MODEL = "qwen2.5-14b"
EXPECTED_TRANSLATION_MODELS = {
    "tr_to_en": (
        "Helsinki-NLP/opus-mt-tc-big-tr-en",
        "2261c8fc7b1af59caee87f8ff0ecf3fbccfe8391",
    ),
    "en_to_tr": (
        "Helsinki-NLP/opus-mt-tc-big-en-tr",
        "e539fc16a8a1a0ea5950eb339b595bfcce990e90",
    ),
}
EXPECTED_GLOSSARY = "northstar-summer-lab-en-tr-v1"
EXPECTED_CONFIGURATION = {
    "chunk_size": 240,
    "chunk_overlap": 45,
    "top_k": 5,
    "retrieval_threshold": 0.60,
    "evidence_floor": 0.30,
    "max_chunks_per_document": 2,
    "max_context_chunks": 4,
    "context_char_budget": 3200,
    "translation_runtime": "marian",
    "translation_beams": 4,
    "translation_max_tokens": 256,
    "translation_round_trip_threshold": 0.60,
    "translation_glossary_id": EXPECTED_GLOSSARY,
}
EXPECTED_QUALITY_POLICY = {
    "required_fact_recall_threshold": 0.65,
    "per_required_fact_recall_threshold": 0.50,
    "protected_fact_completeness_required": True,
    "full_expected_source_recall_required": True,
    "citation_precision_required": 1.0,
}
EXPECTED_REVIEW_CRITERIA = {
    "natural Turkish grammar and word order",
    "complete meaning and procedure order",
    "exact names, dates, times, numbers, phone numbers, and email addresses",
    "natural terminology and suffixes",
    "matching evidence citations or deterministic grounded refusal",
}
LOCALE_MARKERS = {
    "en": (
        "THE TRUST GAP",
        "LOCAL RAG ARCHITECTURE",
        "PRODUCT DEMO",
        "RELEASE EVIDENCE",
        "WHY TRACEFOLD",
    ),
    "tr": (
        "GÜVEN AÇIĞI",
        "YEREL RAG MİMARİSİ",
        "ÜRÜN DEMOSU",
        "SÜRÜM KANITI",
        "NEDEN TRACEFOLD",
    ),
}
RELEASE_ASSETS = {
    "Tracefold_Final_Presentation.pptx": Path("presentation/Tracefold_Final_Presentation.pptx"),
    "Tracefold_Final_Presentation.pdf": Path("presentation/Tracefold_Final_Presentation.pdf"),
    "Tracefold_Final_Presentation_TR.pptx": Path(
        "presentation/Tracefold_Final_Presentation_TR.pptx"
    ),
    "Tracefold_Final_Presentation_TR.pdf": Path("presentation/Tracefold_Final_Presentation_TR.pdf"),
    "evaluation-report.json": Path("docs/evaluation-report.json"),
}
BEHAVIOR_PATHS = (
    "backend/",
    "frontend/",
    "scripts/",
    "tests/",
    "demo_docs/",
    ".github/",
    ".env.example",
    "Makefile",
    "docs/eval-dataset.json",
    "pyproject.toml",
    "requirements",
)
SHA256 = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


class ArtifactError(ValueError):
    """A release artifact is missing, stale, corrupt, or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArtifactError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _same_value(actual: Any, expected: Any) -> bool:
    """Compare JSON configuration values without accepting booleans as integers."""

    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, float):
        return (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-9)
        )
    return actual == expected


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _require_expected(mapping: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    for key, value in expected.items():
        _require(
            _same_value(mapping.get(key), value),
            f"{label} must record {key}={value!r}.",
        )


def _validate_foundry_model(model: dict[str, Any], *, alias: str, label: str) -> None:
    recorded_alias = str(model.get("alias") or "").casefold()
    variant_id = str(model.get("variant_id") or "").casefold()
    _require(recorded_alias == alias, f"Release report must identify {label} as {alias}.")
    _require(
        variant_id.startswith(alias),
        f"Release report must record the selected {label} variant for {alias}.",
    )
    _require(
        model.get("cached") is True and model.get("loaded") is True,
        f"Release report must prove the {label} model was cached and loaded.",
    )
    _require(
        str(model.get("device") or "").upper() in {"CPU", "GPU"},
        f"Release report must record the {label} execution device.",
    )
    _require(
        bool(str(model.get("execution_provider") or "").strip()),
        f"Release report must record the {label} execution provider.",
    )


def _validate_manual_review(review: dict[str, Any], cases: list[dict[str, Any]]) -> None:
    _require(
        review.get("status") == "approved" and review.get("verified") is True,
        "Turkish manual language review must be approved and verified.",
    )
    _require(
        review.get("locale") == "tr" and review.get("cases_reviewed") == 12,
        "Manual review must cover exactly 12 Turkish cases.",
    )
    _require(
        len(str(review.get("reviewer") or "").strip()) >= 3
        and review.get("reviewer_type") == "case-by-case release audit",
        "Manual review must identify a case-by-case release reviewer.",
    )
    try:
        reviewed_at = datetime.fromisoformat(
            str(review.get("reviewed_at") or "").replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ArtifactError("Manual review timestamp must be valid ISO-8601.") from exc
    _require(
        reviewed_at.tzinfo is not None and reviewed_at.utcoffset() is not None,
        "Manual review timestamp must include a timezone offset.",
    )
    criteria = review.get("criteria") or []
    _require(
        isinstance(criteria, list)
        and all(isinstance(item, str) for item in criteria)
        and set(criteria) == EXPECTED_REVIEW_CRITERIA,
        "Manual review must record every Turkish release-review criterion.",
    )
    turkish_cases = [case for case in cases if case.get("locale") == "tr"]
    _require(
        len(turkish_cases) == 12
        and all(str(case.get("question") or "").strip() for case in turkish_cases)
        and all(str(case.get("answer") or "").strip() for case in turkish_cases),
        "Every reviewed Turkish case must include its exact question and answer.",
    )
    expected_digest = review_answers_sha256(cases)
    _require(
        review.get("answers_sha256") == expected_digest,
        "Manual review digest does not match the exported Turkish answers.",
    )


def _validate_runtime(context: dict[str, Any]) -> None:
    runtime = _mapping(context.get("runtime"))
    _require(
        runtime.get("runtime") == "foundry_local"
        and runtime.get("ready") is True
        and runtime.get("error") is None,
        "Release evaluation must prove a ready Foundry Local runtime without errors.",
    )
    _validate_foundry_model(
        _mapping(runtime.get("embedding_model")),
        alias=EXPECTED_EMBEDDING_MODEL,
        label="embedding",
    )
    _validate_foundry_model(
        _mapping(runtime.get("chat_model")), alias=EXPECTED_CHAT_MODEL, label="chat"
    )

    translation = _mapping(context.get("translation"))
    _require(
        translation.get("runtime") == "marian_opus"
        and translation.get("ready") is True
        and translation.get("offline_only") is True
        and translation.get("error") is None,
        "Release evaluation must prove Marian OPUS is ready in offline-only mode.",
    )
    models = translation.get("models") or {}
    _require(
        isinstance(models, dict) and set(models) == set(EXPECTED_TRANSLATION_MODELS),
        "Release report must contain exactly the pinned Turkish translation model pair.",
    )
    for direction, (model_id, revision) in EXPECTED_TRANSLATION_MODELS.items():
        model = _mapping(models.get(direction))
        _require(
            model.get("id") == model_id and model.get("revision") == revision,
            f"Release report has an unexpected pinned {direction} model identity or revision.",
        )
        _require(
            model.get("cached") is True and model.get("loaded") is True,
            f"Release report must prove the pinned {direction} model was cached and loaded.",
        )
    glossary = _mapping(translation.get("glossary"))
    _require(
        glossary.get("id") == EXPECTED_GLOSSARY
        and isinstance(glossary.get("entries"), int)
        and not isinstance(glossary.get("entries"), bool)
        and glossary["entries"] > 0,
        "Release report must identify a populated pinned terminology glossary.",
    )
    _require(
        _same_value(translation.get("round_trip_threshold"), 0.60),
        "Release report must record the 0.60 translation round-trip threshold.",
    )


def _validate_configuration(metrics: dict[str, Any], context: dict[str, Any]) -> None:
    configuration = _mapping(context.get("configuration"))
    _require_expected(configuration, EXPECTED_CONFIGURATION, "Release configuration")
    _require(
        configuration.get("locales") == ["en", "tr"],
        "Release configuration must evaluate English and Turkish.",
    )
    policy = _mapping(configuration.get("quality_policy"))
    _require_expected(policy, EXPECTED_QUALITY_POLICY, "Release quality policy")
    _require(
        policy.get("manual_language_review_required_for_release") == ["tr"],
        "Release quality policy must require Turkish manual language review.",
    )
    _require(
        _same_value(metrics.get("top_k"), EXPECTED_CONFIGURATION["top_k"])
        and _same_value(
            metrics.get("retrieval_threshold"),
            EXPECTED_CONFIGURATION["retrieval_threshold"],
        ),
        "Measured retrieval settings must match the release configuration.",
    )
    _require(
        metrics.get("locales") == ["en", "tr"]
        and metrics.get("bilingual_pairs_evaluated") == 12
        and _same_value(metrics.get("bilingual_pair_accuracy"), 1.0),
        "Release report must prove all 12 bilingual pairs passed.",
    )


def validate_report(path: Path) -> dict[str, Any]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"Invalid evaluation report {path}: {exc}") from exc

    _require(isinstance(report, dict), "Evaluation report root must be a JSON object.")
    metrics = _mapping(report.get("metrics"))
    context = _mapping(metrics.get("run_context"))
    _require(report.get("status") == "complete", "Evaluation status must be complete.")
    _require(report.get("dataset_id") == EXPECTED_DATASET, "Unexpected evaluation dataset.")
    _require(metrics.get("total_cases") == 24, "Evaluation must contain 24 cases.")
    _require(metrics.get("passed_cases") == 24, "All 24 evaluation cases must pass.")

    _require(context.get("dataset_id") == EXPECTED_DATASET, "Run context dataset is inconsistent.")
    breakdown = _mapping(metrics.get("locale_breakdown"))
    for locale in ("en", "tr"):
        locale_metrics = _mapping(breakdown.get(locale))
        _require(locale_metrics.get("total_cases") == 12, f"{locale} must contain 12 cases.")
        _require(locale_metrics.get("passed_cases") == 12, f"All {locale} cases must pass.")

    cases = report.get("cases") or []
    _require(
        isinstance(cases, list)
        and len(cases) == 24
        and all(isinstance(case, dict) and case.get("passed") is True for case in cases),
        "Every exported case must be present and passed.",
    )
    provenance = _mapping(context.get("code_provenance"))
    revision = str(provenance.get("revision") or "")
    _require(bool(GIT_SHA.fullmatch(revision)), "Evaluation revision must be a full Git SHA.")
    _require(provenance.get("dirty") is False, "Evaluation must come from a clean Git tree.")

    _validate_manual_review(_mapping(context.get("manual_language_review")), cases)
    _validate_runtime(context)
    _validate_configuration(metrics, context)
    return report


def _pptx_slide_text(path: Path) -> list[str]:
    try:
        with zipfile.ZipFile(path) as archive:
            _require(archive.testzip() is None, f"Corrupt PPTX member in {path}.")
            names = set(archive.namelist())
            _require(
                "[Content_Types].xml" in names and "ppt/presentation.xml" in names,
                f"{path} is not a complete PPTX package.",
            )
            slides = sorted(
                name for name in names if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)
            )
            slide_text: list[str] = []
            for slide in slides:
                root = ElementTree.fromstring(archive.read(slide))
                slide_text.append(
                    " ".join(node.text or "" for node in root.iter() if node.tag.endswith("}t"))
                )
            return slide_text
    except (OSError, zipfile.BadZipFile, ElementTree.ParseError) as exc:
        raise ArtifactError(f"Invalid PPTX {path}: {exc}") from exc


def _pptx_text(path: Path) -> tuple[int, str]:
    """Return aggregate PPTX text for local inspection and backward compatibility."""

    slides = _pptx_slide_text(path)
    return len(slides), " ".join(slides)


def _asset_locale(path: Path, locale: str | None) -> str:
    inferred = "tr" if path.stem.casefold().endswith("_tr") else "en"
    selected = locale or inferred
    _require(selected in LOCALE_MARKERS, f"Unsupported presentation locale: {selected}")
    return selected


def _normalized_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _require_locale_markers(text: str, *, locale: str, asset_name: str) -> None:
    normalized = _normalized_text(text)
    missing = [
        marker for marker in LOCALE_MARKERS[locale] if _normalized_text(marker) not in normalized
    ]
    _require(
        not missing,
        f"{asset_name} is missing {locale.upper()} locale markers: {', '.join(missing)}.",
    )
    opposite = "tr" if locale == "en" else "en"
    opposite_hits = [
        marker for marker in LOCALE_MARKERS[opposite] if _normalized_text(marker) in normalized
    ]
    _require(
        len(opposite_hits) < 2,
        f"{asset_name} appears to contain the swapped {opposite.upper()} deck.",
    )


def validate_pptx(path: Path, *, locale: str | None = None) -> None:
    _require(path.is_file() and path.stat().st_size > 10_000, f"Missing or empty PPTX: {path}")
    selected_locale = _asset_locale(path, locale)
    slides = _pptx_slide_text(path)
    text = " ".join(slides)
    _require(len(slides) == 6, f"{path.name} must contain exactly six slides.")
    substantial_slides = sum(len(re.findall(r"\w+", slide, re.UNICODE)) >= 5 for slide in slides)
    word_count = len(re.findall(r"\w+", text, re.UNICODE))
    _require(
        substantial_slides >= 5 and word_count >= 100,
        f"{path.name} does not contain a substantive six-slide presentation.",
    )
    _require(
        re.search(r"24\s*/\s*24", text) is not None,
        f"{path.name} does not contain the final 24/24 result.",
    )
    _require(
        re.search(r"qwen\s*2\.5.{0,40}?14\s*b", text, re.IGNORECASE | re.DOTALL) is not None,
        f"{path.name} does not identify the 14B release model.",
    )
    _require_locale_markers(text, locale=selected_locale, asset_name=path.name)


def validate_pdf(path: Path, *, locale: str | None = None) -> None:
    _require(path.is_file() and path.stat().st_size > 512, f"Missing or empty PDF: {path}")
    selected_locale = _asset_locale(path, locale)
    try:
        import fitz

        with fitz.open(path) as document:
            _require(
                not document.needs_pass, f"Encrypted PDF is not a public release asset: {path}"
            )
            _require(document.page_count == 6, f"{path.name} must contain exactly six pages.")
            page_text: list[str] = []
            for page in document:
                ratio = page.rect.width / page.rect.height
                _require(1.70 <= ratio <= 1.85, f"{path.name} contains a non-16:9 page.")
                page_text.append(page.get_text("text"))
            text = " ".join(page_text)
            substantive_pages = sum(
                len(re.findall(r"\w+", item, re.UNICODE)) >= 5 for item in page_text
            )
            word_count = len(re.findall(r"\w+", text, re.UNICODE))
            _require(
                substantive_pages >= 5 and word_count >= 100,
                f"{path.name} must contain meaningful extractable text; raster-only exports fail.",
            )
            _require(
                re.search(r"24\s*/\s*24", text) is not None,
                f"{path.name} does not contain the final 24/24 result as extractable text.",
            )
            _require(
                re.search(
                    r"qwen\s*2\.5.{0,40}?14\s*b",
                    text,
                    re.IGNORECASE | re.DOTALL,
                )
                is not None,
                f"{path.name} does not identify the 14B release model as extractable text.",
            )
            _require_locale_markers(text, locale=selected_locale, asset_name=path.name)
    except ArtifactError:
        raise
    except Exception as exc:
        raise ArtifactError(f"Invalid PDF {path}: {exc}") from exc


def asset_paths(root: Path) -> dict[str, Path]:
    return {name: root / relative for name, relative in RELEASE_ASSETS.items()}


def write_checksums(paths: dict[str, Path], manifest: Path) -> None:
    _require(not manifest.is_symlink(), "Refusing to replace a symlinked checksum manifest.")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(f"{_sha256(path)}  {name}\n" for name, path in sorted(paths.items()))
    temporary = manifest.with_suffix(manifest.suffix + ".tmp")
    temporary.write_text(payload, encoding="ascii")
    os.replace(temporary, manifest)


def validate_checksums(paths: dict[str, Path], manifest: Path) -> None:
    _require(manifest.is_file(), f"Missing checksum manifest: {manifest}")
    recorded: dict[str, str] = {}
    for line_number, line in enumerate(manifest.read_text(encoding="ascii").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  ([^/\\]+)", line)
        _require(match is not None, f"Malformed checksum line {line_number}.")
        assert match is not None
        digest, name = match.groups()
        _require(name not in recorded, f"Duplicate checksum entry: {name}")
        recorded[name] = digest
    _require(set(recorded) == set(paths), "Checksum manifest has missing or unexpected assets.")
    for name, path in paths.items():
        _require(SHA256.fullmatch(recorded[name]) is not None, f"Invalid SHA-256 for {name}.")
        _require(_sha256(path) == recorded[name], f"Checksum mismatch for {name}.")


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def validate_git_provenance(root: Path, revision: str, *, require_clean: bool) -> None:
    _require(
        _git(root, "cat-file", "-e", f"{revision}^{{commit}}").returncode == 0,
        "Evaluated Git revision is not available locally.",
    )
    _require(
        _git(root, "merge-base", "--is-ancestor", revision, "HEAD").returncode == 0,
        "Evaluated revision is not an ancestor of the release commit.",
    )
    changed = _git(root, "diff", "--name-only", f"{revision}..HEAD")
    _require(changed.returncode == 0, "Could not compare evaluated and release revisions.")
    behavior_changes = [
        path for path in changed.stdout.splitlines() if path.startswith(BEHAVIOR_PATHS)
    ]
    _require(
        not behavior_changes,
        "Runtime behavior changed after evaluation: " + ", ".join(behavior_changes),
    )
    if require_clean:
        status = _git(root, "status", "--porcelain")
        _require(
            status.returncode == 0 and not status.stdout.strip(),
            "Final release artifact verification requires a clean Git tree.",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--checksums", type=Path, default=Path("release/SHA256SUMS"))
    parser.add_argument("--write-checksums", action="store_true")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    paths = asset_paths(root)
    report = validate_report(paths["evaluation-report.json"])
    for name, path in paths.items():
        if name.endswith(".pptx"):
            validate_pptx(path, locale="tr" if "_TR." in name else "en")
        elif name.endswith(".pdf"):
            validate_pdf(path, locale="tr" if "_TR." in name else "en")
    provenance = report["metrics"]["run_context"]["code_provenance"]
    validate_git_provenance(root, provenance["revision"], require_clean=not args.write_checksums)

    manifest = args.checksums
    if not manifest.is_absolute():
        manifest = root / manifest
    if args.write_checksums:
        write_checksums(paths, manifest)
        print(f"Wrote verified checksums: {manifest}")
    else:
        validate_checksums(paths, manifest)
        print(f"Release artifacts verified: {manifest}")


if __name__ == "__main__":
    try:
        main()
    except ArtifactError as exc:
        raise SystemExit(f"Release artifact check failed: {exc}") from None
