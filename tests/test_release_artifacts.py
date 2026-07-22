from __future__ import annotations

import html
import json
import zipfile
from pathlib import Path
from typing import Any

import fitz
import pytest
from storage import review_answers_sha256

from scripts.check_release_artifacts import (
    EXPECTED_REVIEW_CRITERIA,
    LOCALE_MARKERS,
    ArtifactError,
    validate_checksums,
    validate_pdf,
    validate_pptx,
    validate_report,
    write_checksums,
)


def valid_report() -> dict[str, Any]:
    cases = [
        {
            "id": f"NSL-{locale.upper()}-{index:02d}",
            "locale": locale,
            "question": (f"Question {index}" if locale == "en" else f"{index}. Türkçe soru"),
            "answer": f"Answer {index}" if locale == "en" else f"{index}. Türkçe yanıt",
            "passed": True,
        }
        for locale in ("en", "tr")
        for index in range(12)
    ]
    report: dict[str, Any] = {
        "status": "complete",
        "dataset_id": "northstar-summer-lab-2026-bilingual-parity-v3",
        "cases": cases,
        "metrics": {
            "total_cases": 24,
            "passed_cases": 24,
            "locales": ["en", "tr"],
            "bilingual_pairs_evaluated": 12,
            "bilingual_pair_accuracy": 1.0,
            "top_k": 5,
            "retrieval_threshold": 0.60,
            "locale_breakdown": {
                "en": {"total_cases": 12, "passed_cases": 12},
                "tr": {"total_cases": 12, "passed_cases": 12},
            },
            "run_context": {
                "dataset_id": "northstar-summer-lab-2026-bilingual-parity-v3",
                "code_provenance": {"revision": "a" * 40, "dirty": False},
                "runtime": {
                    "runtime": "foundry_local",
                    "ready": True,
                    "error": None,
                    "embedding_model": {
                        "alias": "qwen3-embedding-0.6b",
                        "variant_id": "qwen3-embedding-0.6b-generic-gpu:1",
                        "device": "GPU",
                        "execution_provider": "WebGpuExecutionProvider",
                        "cached": True,
                        "loaded": True,
                    },
                    "chat_model": {
                        "alias": "qwen2.5-14b",
                        "variant_id": "qwen2.5-14b-instruct-generic-gpu:4",
                        "device": "GPU",
                        "execution_provider": "WebGpuExecutionProvider",
                        "cached": True,
                        "loaded": True,
                    },
                },
                "translation": {
                    "runtime": "marian_opus",
                    "ready": True,
                    "offline_only": True,
                    "error": None,
                    "models": {
                        "tr_to_en": {
                            "id": "Helsinki-NLP/opus-mt-tc-big-tr-en",
                            "revision": "2261c8fc7b1af59caee87f8ff0ecf3fbccfe8391",
                            "cached": True,
                            "loaded": True,
                        },
                        "en_to_tr": {
                            "id": "Helsinki-NLP/opus-mt-tc-big-en-tr",
                            "revision": "e539fc16a8a1a0ea5950eb339b595bfcce990e90",
                            "cached": True,
                            "loaded": True,
                        },
                    },
                    "glossary": {
                        "id": "northstar-summer-lab-en-tr-v1",
                        "entries": 55,
                    },
                    "round_trip_threshold": 0.60,
                },
                "configuration": {
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
                    "translation_glossary_id": "northstar-summer-lab-en-tr-v1",
                    "locales": ["en", "tr"],
                    "quality_policy": {
                        "required_fact_recall_threshold": 0.65,
                        "per_required_fact_recall_threshold": 0.50,
                        "protected_fact_completeness_required": True,
                        "full_expected_source_recall_required": True,
                        "citation_precision_required": 1.0,
                        "manual_language_review_required_for_release": ["tr"],
                    },
                },
                "manual_language_review": {
                    "status": "approved",
                    "verified": True,
                    "locale": "tr",
                    "reviewer": "Tracefold release QA",
                    "reviewer_type": "case-by-case release audit",
                    "reviewed_at": "2026-07-22T12:00:00+00:00",
                    "cases_reviewed": 12,
                    "criteria": sorted(EXPECTED_REVIEW_CRITERIA),
                    "answers_sha256": review_answers_sha256(cases),
                },
            },
        },
    }
    return report


def write_report(tmp_path: Path, report: dict[str, Any]) -> Path:
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    return path


def set_nested(mapping: dict[str, Any], path: str, value: Any) -> None:
    target = mapping
    keys = path.split(".")
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value


def deck_text(locale: str) -> list[str]:
    markers = LOCALE_MARKERS[locale]
    filler = (
        "Local evidence citations retrieval documents privacy reproducible evaluation "
        "grounded answer model runtime source passage bilingual quality release"
        if locale == "en"
        else "Yerel kanıt atıf arama belgeler gizlilik yinelenebilir değerlendirme "
        "kaynaklı yanıt model çalışma kaynak metin parçası iki dilli kalite sürüm"
    )
    slides = [
        f"TRACEFOLD {markers[index] if index < len(markers) else 'TRACEFOLD'} {filler} {filler}"
        for index in range(6)
    ]
    slides[4] += " 24/24 Qwen2.5 14B"
    return slides


def write_pptx(path: Path, slides: list[str]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("ppt/presentation.xml", '<p:presentation xmlns:p="p"/>')
        for index, text in enumerate(slides, 1):
            payload = f'<p:sld xmlns:p="p" xmlns:a="a"><a:t>{html.escape(text)}</a:t></p:sld>'
            archive.writestr(f"ppt/slides/slide{index}.xml", payload)
        archive.writestr("padding.bin", b"0" * 12_000)


def write_pdf(path: Path, slides: list[str]) -> None:
    font_buffer = fitz.Font("cjk").buffer
    with fitz.open() as document:
        for text in slides:
            page = document.new_page(width=1280, height=720)
            page.insert_font(fontname="tracefold", fontbuffer=font_buffer)
            page.insert_textbox(
                fitz.Rect(50, 50, 1230, 670),
                text,
                fontname="tracefold",
                fontsize=15,
            )
        document.save(path)


def test_release_report_requires_clean_approved_24_case_run(tmp_path: Path):
    report = valid_report()
    path = write_report(tmp_path, report)
    assert validate_report(path)["metrics"]["passed_cases"] == 24

    report["metrics"]["run_context"]["code_provenance"]["dirty"] = True
    write_report(tmp_path, report)
    with pytest.raises(ArtifactError, match="clean Git tree"):
        validate_report(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("metrics.run_context.runtime.runtime", "fake", "Foundry Local"),
        ("metrics.run_context.runtime.ready", False, "Foundry Local"),
        (
            "metrics.run_context.runtime.embedding_model.alias",
            "different-embedding",
            "qwen3-embedding-0.6b",
        ),
        ("metrics.run_context.runtime.embedding_model.loaded", False, "cached and loaded"),
        ("metrics.run_context.runtime.chat_model.alias", "qwen2.5-7b", "qwen2.5-14b"),
        ("metrics.run_context.runtime.chat_model.cached", False, "cached and loaded"),
        ("metrics.run_context.translation.offline_only", False, "offline-only"),
        (
            "metrics.run_context.translation.models.en_to_tr.id",
            "untrusted/model",
            "unexpected pinned",
        ),
        (
            "metrics.run_context.translation.models.tr_to_en.revision",
            "d" * 40,
            "unexpected pinned",
        ),
        ("metrics.run_context.translation.models.tr_to_en.cached", False, "cached and loaded"),
        ("metrics.run_context.translation.round_trip_threshold", 0.4, "0.60"),
        ("metrics.run_context.configuration.retrieval_threshold", 0.55, "retrieval_threshold"),
        ("metrics.run_context.configuration.max_context_chunks", 3, "max_context_chunks"),
        (
            "metrics.run_context.configuration.quality_policy.citation_precision_required",
            0.9,
            "citation_precision_required",
        ),
        ("metrics.bilingual_pair_accuracy", 0.99, "all 12 bilingual pairs"),
        (
            "metrics.run_context.manual_language_review.reviewer_type",
            "automated",
            "case-by-case",
        ),
        ("metrics.run_context.manual_language_review.criteria", [], "every Turkish"),
    ],
)
def test_release_report_rejects_unproved_runtime_and_policy(
    tmp_path: Path,
    field: str,
    value: Any,
    message: str,
):
    report = valid_report()
    set_nested(report, field, value)
    path = write_report(tmp_path, report)
    with pytest.raises(ArtifactError, match=message):
        validate_report(path)


def test_release_report_rejects_stale_manual_review_digest(tmp_path: Path):
    report = valid_report()
    report["cases"][12]["answer"] = "İncelemeden sonra değişen Türkçe yanıt"
    path = write_report(tmp_path, report)
    with pytest.raises(ArtifactError, match="digest"):
        validate_report(path)


def test_bilingual_pptx_files_are_substantive_and_not_swapped(tmp_path: Path):
    english = tmp_path / "Tracefold_Final_Presentation.pptx"
    turkish = tmp_path / "Tracefold_Final_Presentation_TR.pptx"
    write_pptx(english, deck_text("en"))
    write_pptx(turkish, deck_text("tr"))

    validate_pptx(english, locale="en")
    validate_pptx(turkish, locale="tr")
    with pytest.raises(ArtifactError, match="TR locale markers"):
        validate_pptx(english, locale="tr")
    with pytest.raises(ArtifactError, match="EN locale markers"):
        validate_pptx(turkish, locale="en")


def test_padded_but_empty_pptx_is_rejected(tmp_path: Path):
    pptx = tmp_path / "Tracefold_Final_Presentation.pptx"
    write_pptx(pptx, ["empty"] * 6)
    with pytest.raises(ArtifactError, match="substantive"):
        validate_pptx(pptx, locale="en")


def test_bilingual_pdfs_are_textual_and_not_swapped(tmp_path: Path):
    english = tmp_path / "Tracefold_Final_Presentation.pdf"
    turkish = tmp_path / "Tracefold_Final_Presentation_TR.pdf"
    write_pdf(english, deck_text("en"))
    write_pdf(turkish, deck_text("tr"))

    validate_pdf(english, locale="en")
    validate_pdf(turkish, locale="tr")
    with pytest.raises(ArtifactError, match="TR locale markers"):
        validate_pdf(english, locale="tr")
    with pytest.raises(ArtifactError, match="EN locale markers"):
        validate_pdf(turkish, locale="en")


def test_raster_only_pdf_is_rejected(tmp_path: Path):
    pdf = tmp_path / "Tracefold_Final_Presentation.pdf"
    with fitz.open() as document:
        for _ in range(6):
            document.new_page(width=1280, height=720)
        document.save(pdf)
    with pytest.raises(ArtifactError, match="extractable text"):
        validate_pdf(pdf, locale="en")


def test_pdf_requires_six_16_by_9_pages(tmp_path: Path):
    pdf = tmp_path / "Tracefold_Final_Presentation.pdf"
    slides = deck_text("en")
    font_buffer = fitz.Font("cjk").buffer
    with fitz.open() as document:
        for index, text in enumerate(slides):
            page = document.new_page(width=1280, height=720 if index else 800)
            page.insert_font(fontname="tracefold", fontbuffer=font_buffer)
            page.insert_text(
                (50, 100),
                text,
                fontname="tracefold",
                fontsize=10,
            )
        document.save(pdf)
    with pytest.raises(ArtifactError, match="non-16:9"):
        validate_pdf(pdf, locale="en")


def test_checksum_manifest_rejects_tampering(tmp_path: Path):
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    paths = {"first.bin": first, "second.bin": second}
    manifest = tmp_path / "SHA256SUMS"
    write_checksums(paths, manifest)
    validate_checksums(paths, manifest)

    first.write_bytes(b"tampered")
    with pytest.raises(ArtifactError, match="Checksum mismatch"):
        validate_checksums(paths, manifest)
