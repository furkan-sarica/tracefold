"""Reproducible retrieval, citation, answer, and refusal evaluation."""

from __future__ import annotations

import json
import math
import os
import platform
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from rag import AnswerLocale, RAGPipeline
from storage import TracefoldStore


def load_evaluation_dataset(dataset_path: Path, demo_docs_dir: Path) -> dict[str, Any]:
    """Load the checked-in dataset and validate every named source against demo_docs/."""
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("cases"), list) or not payload["cases"]:
        raise ValueError("Evaluation dataset must contain a non-empty 'cases' list.")
    resolved_sources: dict[str, str] = {}
    seen_ids: set[str] = set()
    for case in payload["cases"]:
        case_id = str(case.get("id") or "").strip()
        if not case_id or case_id in seen_ids:
            raise ValueError("Every evaluation case must have a unique non-empty 'id'.")
        seen_ids.add(case_id)
        if case.get("locale") not in {"en", "tr"}:
            raise ValueError(f"Unsupported evaluation locale in {case_id}: {case.get('locale')}")
        for filename in case.get("expected_sources", []):
            if Path(filename).name != filename:
                raise ValueError(f"Unsafe evaluation source name: {filename}")
            source_path = demo_docs_dir / filename
            if not source_path.is_file():
                raise ValueError(f"Evaluation source does not exist in demo_docs/: {filename}")
            resolved_sources[filename] = str(source_path.resolve())
    paired_targets = [
        str(case["paired_case_id"]) for case in payload["cases"] if case.get("paired_case_id")
    ]
    if len(paired_targets) != len(set(paired_targets)):
        raise ValueError("Each bilingual evaluation pair target must be unique.")
    missing_pairs = sorted(set(paired_targets) - seen_ids)
    if missing_pairs:
        raise ValueError("Evaluation pair target does not exist: " + ", ".join(missing_pairs))
    payload["resolved_sources"] = resolved_sources
    return payload


class Evaluator:
    def __init__(self, rag: RAGPipeline, store: TracefoldStore):
        self.rag = rag
        self.store = store

    @staticmethod
    def _tokens(value: str) -> set[str]:
        stopwords = {
            "a",
            "an",
            "and",
            "are",
            "at",
            "be",
            "by",
            "for",
            "from",
            "in",
            "is",
            "it",
            "of",
            "on",
            "or",
            "over",
            "takes",
            "the",
            "to",
            "with",
            "ve",
            "veya",
            "bir",
            "bu",
            "da",
            "de",
            "ile",
            "için",
            "mı",
            "mi",
            "mu",
            "mü",
            "ne",
            "nasıl",
            "nedir",
        }
        aliases = {
            "prioritize": "priority",
            "prioritized": "priority",
            "prioritizes": "priority",
        }

        def normalize(token: str) -> str:
            token = aliases.get(token, token)
            if token.isascii() and token.isalpha():
                if len(token) > 6 and token.endswith("ingly"):
                    token = token[:-5]
                elif len(token) > 5 and token.endswith("ing"):
                    token = token[:-3]
                elif len(token) > 5 and token.endswith("edly"):
                    token = token[:-4]
                elif len(token) > 4 and token.endswith("ed"):
                    token = token[:-2]
                elif len(token) > 5 and token.endswith("ly"):
                    token = token[:-2]
            return aliases.get(token, token)

        return {
            normalized
            for token in re.findall(r"[^\W_]+", value.casefold(), flags=re.UNICODE)
            if token not in stopwords
            if (normalized := normalize(token)) not in stopwords
        }

    @classmethod
    def _snippet_recall(cls, answer: str, snippet: str | None) -> float | None:
        if not snippet:
            return None
        expected = cls._tokens(snippet)
        if not expected:
            return None
        return round(len(expected & cls._tokens(answer)) / len(expected), 4)

    @classmethod
    def _required_fact_recall(cls, answer: str, fact: str) -> float | None:
        """Token recall with hard guards for numeric and explicit priority relations."""

        expected_numbers = set(re.findall(r"\d+", fact))
        answer_numbers = set(re.findall(r"\d+", answer))
        if expected_numbers and not expected_numbers.issubset(answer_numbers):
            return 0.0
        priority = re.search(
            r"(?P<first>.+?)\s+(?:takes?\s+priority\s+over|must\s+happen\s+before|"
            r"should\s+happen\s+before)\s+(?P<second>.+?)(?:[.!?]|$)",
            fact,
            flags=re.IGNORECASE,
        )
        if priority:
            folded_answer = answer.casefold()
            expected_first = cls._tokens(priority.group("first"))
            expected_second = cls._tokens(priority.group("second"))

            explicit_relations = re.finditer(
                r"(?P<first>[^.!?;]+?)\s+(?:takes?\s+priority\s+over|must\s+happen\s+before|"
                r"should\s+happen\s+before)\s+(?P<second>[^.!?;]+)",
                answer,
                flags=re.IGNORECASE,
            )
            for relation in explicit_relations:
                observed_first = cls._tokens(relation.group("first"))
                observed_second = cls._tokens(relation.group("second"))
                correct = len(expected_first & observed_first) + len(
                    expected_second & observed_second
                )
                inverse = len(expected_first & observed_second) + len(
                    expected_second & observed_first
                )
                if inverse > correct:
                    return 0.0

            def earliest(entity: str) -> int | None:
                positions = [
                    folded_answer.find(token)
                    for token in cls._tokens(entity)
                    if folded_answer.find(token) >= 0
                ]
                return min(positions) if positions else None

            first_position = earliest(priority.group("first"))
            second_position = earliest(priority.group("second"))
            if (
                first_position is None
                or second_position is None
                or first_position >= second_position
            ):
                return 0.0
            return cls._snippet_recall(
                answer,
                f"{priority.group('first')} {priority.group('second')}",
            )
        return cls._snippet_recall(answer, fact)

    @staticmethod
    def _required_order_complete(answer: str, relations: list[dict[str, Any]]) -> bool | None:
        """Verify explicit dataset order anchors without inferring from bag-of-words overlap."""

        if not relations:
            return None
        folded = answer.casefold()

        def anchored_position(term: str, actions: list[str]) -> int:
            positions: list[int] = []
            for match in re.finditer(re.escape(term), folded):
                clause_start = max(
                    folded.rfind(separator, 0, match.start()) for separator in ".!?;\n"
                )
                clause_end_candidates = [
                    position
                    for separator in ".!?;\n"
                    if (position := folded.find(separator, match.end())) >= 0
                ]
                clause_end = min(clause_end_candidates) if clause_end_candidates else len(folded)
                clause = folded[clause_start + 1 : clause_end]
                if not actions or any(action in clause for action in actions):
                    positions.append(match.start())
            return min(positions) if positions else -1

        for relation in relations:
            before_terms = [str(value).casefold() for value in relation.get("before_all") or []]
            after_terms = [str(value).casefold() for value in relation.get("after_all") or []]
            before_actions = [
                str(value).casefold() for value in relation.get("before_actions") or []
            ]
            after_actions = [str(value).casefold() for value in relation.get("after_actions") or []]
            if not before_terms or not after_terms:
                return False
            before_positions = [anchored_position(value, before_actions) for value in before_terms]
            after_positions = [anchored_position(value, after_actions) for value in after_terms]
            if any(position < 0 for position in (*before_positions, *after_positions)):
                return False
            if max(before_positions) >= min(after_positions):
                return False
        return True

    @staticmethod
    def _protected_fact_complete(answer: str, facts: list[str]) -> bool:
        expected_emails = {
            value.casefold()
            for fact in facts
            for value in re.findall(
                r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
                fact,
                flags=re.IGNORECASE,
            )
        }
        answer_casefold = answer.casefold()
        if not all(value in answer_casefold for value in expected_emails):
            return False
        expected_numbers = {
            value for fact in facts for value in re.findall(r"\d+(?:[.,:]\d+)?", fact)
        }
        answer_numbers = set(re.findall(r"\d+(?:[.,:]\d+)?", answer))
        return expected_numbers.issubset(answer_numbers)

    @staticmethod
    def _code_provenance() -> dict[str, Any]:
        """Record the exact candidate code state without making evaluation depend on Git."""

        explicit_revision = os.getenv("TRACEFOLD_CODE_REVISION", "").strip()
        explicit_dirty = os.getenv("TRACEFOLD_CODE_DIRTY", "").strip().casefold()
        if explicit_revision:
            return {
                "revision": explicit_revision,
                "dirty": explicit_dirty not in {"0", "false", "no"},
                "source": "environment",
            }

        project_root = Path(__file__).resolve().parent.parent
        try:
            revision = subprocess.run(
                ["git", "-C", str(project_root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
            status = subprocess.run(
                [
                    "git",
                    "-C",
                    str(project_root),
                    "status",
                    "--porcelain",
                    "--untracked-files=normal",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout
            return {"revision": revision, "dirty": bool(status.strip()), "source": "git"}
        except (OSError, subprocess.SubprocessError):
            return {"revision": "unavailable", "dirty": True, "source": "unavailable"}

    def run(
        self,
        cases: list[dict[str, Any]],
        *,
        top_k: int = 5,
        dataset_id: str | None = None,
    ) -> dict[str, Any]:
        if not cases:
            raise ValueError("At least one evaluation case is required.")

        indexed_sources = {document["filename"] for document in self.store.list_documents()}
        required_sources = {
            source for case in cases for source in (case.get("expected_sources") or [])
        }
        missing_sources = sorted(required_sources - indexed_sources)
        if missing_sources:
            raise ValueError(
                "Index the demo corpus before evaluation. Missing: " + ", ".join(missing_sources)
            )

        warmup_started = time.perf_counter()
        self.rag.runtime.embed(["Tracefold evaluation warm-up"])
        model_warmup_ms = round((time.perf_counter() - warmup_started) * 1000, 2)

        results = []
        for case in cases:
            locale: AnswerLocale = case.get("locale", "en")
            if locale not in {"en", "tr"}:
                raise ValueError(
                    f"Unsupported evaluation locale in {case.get('id') or 'unnamed case'}: {locale}"
                )
            response = self.rag.answer(case["question"], top_k=top_k, locale=locale)
            expected_sources = set(case.get("expected_sources") or [])
            retrieved_sources = {item["filename"] for item in response["matches"]}
            cited_sources = {item["filename"] for item in response["citations"]}
            must_refuse = bool(case.get("must_refuse") or not case.get("answerable", True))
            answerable = not must_refuse
            source_hits = expected_sources & retrieved_sources
            source_recall = len(source_hits) / len(expected_sources) if expected_sources else None
            citation_source_precision = (
                len(cited_sources & expected_sources) / len(cited_sources)
                if cited_sources
                else (0.0 if answerable else None)
            )
            citation_valid = bool(cited_sources) and not response["refused"] if answerable else None
            citation_sentence_coverage = (
                self.rag.citation_sentence_coverage(response["answer"], response["citations"])
                if answerable and not response["refused"]
                else None
            )
            protected_fact_grounded = (
                self.rag.protected_facts_grounded(response["answer"], response["citations"])
                if answerable and not response["refused"]
                else None
            )
            language_quality = (
                not self.rag.turkish_quality_issues(response["answer"]) if locale == "tr" else True
            )
            terms = [term.lower() for term in case.get("expected_terms") or []]
            term_match = (
                all(term in response["answer"].lower() for term in terms) if terms else None
            )
            fact_scores = [
                self._required_fact_recall(response["answer"], fact)
                for fact in (case.get("required_facts") or [])
            ]
            fact_scores = [score for score in fact_scores if score is not None]
            required_fact_recall = (
                round(sum(fact_scores) / len(fact_scores), 4) if fact_scores else None
            )
            required_fact_complete = (
                all(score >= 0.50 for score in fact_scores) if fact_scores else None
            )
            required_order_complete = self._required_order_complete(
                response["answer"],
                case.get("required_order") or [],
            )
            protected_fact_complete = self._protected_fact_complete(
                response["answer"], case.get("required_facts") or []
            )
            refusal_correct = response["refused"] == must_refuse
            passed = (
                refusal_correct and language_quality
                if not answerable
                else bool(source_hits)
                and source_recall == 1.0
                and citation_valid
                and citation_source_precision == 1.0
                and citation_sentence_coverage
                and protected_fact_grounded
                and protected_fact_complete
                and language_quality
                and refusal_correct
                and term_match is not False
                and required_fact_complete is not False
                and required_order_complete is not False
                and (required_fact_recall is None or required_fact_recall >= 0.65)
            )
            results.append(
                {
                    "id": case.get("id"),
                    "type": case.get("type", "answerable" if answerable else "unanswerable"),
                    "category": case.get("category"),
                    "question": case["question"],
                    "locale": locale,
                    "paired_case_id": case.get("paired_case_id"),
                    "answerable": answerable,
                    "answer": response["answer"],
                    "expected_sources": sorted(expected_sources),
                    "retrieved_sources": sorted(retrieved_sources),
                    "cited_sources": sorted(cited_sources),
                    "hit": bool(source_hits) if answerable else None,
                    "source_recall": round(source_recall, 4) if source_recall is not None else None,
                    "citation_precision": (
                        round(citation_source_precision, 4)
                        if citation_source_precision is not None
                        else None
                    ),
                    "citation_correct": citation_valid,
                    "citation_sentence_coverage": citation_sentence_coverage,
                    "protected_fact_grounded": protected_fact_grounded,
                    "protected_fact_complete": protected_fact_complete,
                    "language_quality": language_quality,
                    "snippet_token_recall": self._snippet_recall(
                        response["answer"], case.get("expected_answer_snippet")
                    ),
                    "required_fact_recall": required_fact_recall,
                    "required_fact_complete": required_fact_complete,
                    "required_order_complete": required_order_complete,
                    "term_match": term_match,
                    "refusal_correct": refusal_correct,
                    "passed": passed,
                    "retrieval_ms": response["retrieval_ms"],
                }
            )

        def average(items: list[dict[str, Any]], key: str) -> float | None:
            values = [item[key] for item in items if item[key] is not None]
            return round(sum(float(value) for value in values) / len(values), 4) if values else None

        def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
            answerable_items = [item for item in items if item["answerable"]]
            negative_items = [item for item in items if not item["answerable"]]
            retrieval = sorted(item["retrieval_ms"] for item in items)
            p95_index = max(0, math.ceil(0.95 * len(retrieval)) - 1)
            return {
                "total_cases": len(items),
                "answerable_cases": len(answerable_items),
                "negative_cases": len(negative_items),
                "passed_cases": sum(bool(item["passed"]) for item in items),
                "hit_at_k": average(answerable_items, "hit"),
                "source_recall_at_k": average(answerable_items, "source_recall"),
                "citation_precision": average(answerable_items, "citation_precision"),
                "citation_accuracy": average(answerable_items, "citation_correct"),
                "citation_sentence_coverage": average(
                    answerable_items, "citation_sentence_coverage"
                ),
                "protected_fact_grounding_accuracy": average(
                    answerable_items, "protected_fact_grounded"
                ),
                "protected_fact_completeness": average(answerable_items, "protected_fact_complete"),
                "snippet_token_recall": average(answerable_items, "snippet_token_recall"),
                "required_fact_recall": average(answerable_items, "required_fact_recall"),
                "required_order_accuracy": average(
                    answerable_items,
                    "required_order_complete",
                ),
                "required_fact_completeness": average(answerable_items, "required_fact_complete"),
                "custom_term_accuracy": average(answerable_items, "term_match"),
                "language_quality_accuracy": average(items, "language_quality"),
                "refusal_accuracy": average(negative_items, "refusal_correct"),
                "average_retrieval_ms": round(
                    sum(item["retrieval_ms"] for item in items) / len(items), 2
                ),
                "retrieval_p95_ms": round(retrieval[p95_index], 2),
            }

        counts = self.store.counts()
        locales = sorted({item["locale"] for item in results}, key=("en", "tr").index)
        locale_breakdown = {
            locale: summarize([item for item in results if item["locale"] == locale])
            for locale in locales
        }
        paired_results: dict[tuple[str, str], bool] = {}
        by_id = {item["id"]: item for item in results if item.get("id")}
        for item in results:
            paired_case_id = item.get("paired_case_id")
            if paired_case_id and paired_case_id in by_id:
                pair_key = tuple(sorted((str(item["id"]), str(paired_case_id))))
                paired_results[pair_key] = bool(item["passed"] and by_id[paired_case_id]["passed"])

        metrics = {
            **summarize(results),
            "locales": locales,
            "locale_breakdown": locale_breakdown,
            "bilingual_pairs_evaluated": len(paired_results),
            "bilingual_pair_accuracy": (
                round(sum(paired_results.values()) / len(paired_results), 4)
                if paired_results
                else None
            ),
            "model_warmup_ms": model_warmup_ms,
            "top_k": top_k,
            "retrieval_threshold": self.rag.threshold,
            "run_context": {
                "dataset_id": dataset_id,
                "runtime": self.rag.runtime.status(probe=False),
                "translation": self.rag.translation.status(probe=False),
                "platform": {
                    "system": platform.system(),
                    "release": platform.release(),
                    "machine": platform.machine(),
                },
                "code_provenance": self._code_provenance(),
                "configuration": {
                    "chunk_size": int(os.getenv("TRACEFOLD_CHUNK_SIZE", "240")),
                    "chunk_overlap": int(os.getenv("TRACEFOLD_CHUNK_OVERLAP", "45")),
                    "top_k": top_k,
                    "retrieval_threshold": self.rag.threshold,
                    "evidence_floor": self.rag.evidence_floor,
                    "max_chunks_per_document": self.rag.max_chunks_per_document,
                    "max_context_chunks": int(os.getenv("TRACEFOLD_MAX_CONTEXT_CHUNKS", "4")),
                    "context_char_budget": int(os.getenv("TRACEFOLD_CONTEXT_CHAR_BUDGET", "3200")),
                    "translation_runtime": os.getenv("TRACEFOLD_TRANSLATION_RUNTIME", "marian"),
                    "translation_beams": int(os.getenv("TRACEFOLD_TRANSLATION_BEAMS", "4")),
                    "translation_max_tokens": int(
                        os.getenv("TRACEFOLD_TRANSLATION_MAX_TOKENS", "256")
                    ),
                    "translation_round_trip_threshold": (self.rag.translation.round_trip_threshold),
                    "translation_glossary_id": self.rag.translation.glossary.id,
                    "locales": locales,
                    "quality_policy": {
                        "required_fact_recall_threshold": 0.65,
                        "per_required_fact_recall_threshold": 0.50,
                        "protected_fact_completeness_required": True,
                        "full_expected_source_recall_required": True,
                        "citation_precision_required": 1.0,
                        "manual_language_review_required_for_release": ["tr"],
                    },
                },
                "documents": counts["documents"],
                "chunks": counts["chunks"],
            },
        }
        run_id = self.store.save_evaluation(metrics, results)
        saved = self.store.latest_evaluation()
        return {
            "id": run_id,
            "dataset_id": dataset_id,
            "status": "complete",
            "created_at": saved["created_at"] if saved else None,
            "retrieval_p95_ms": metrics["retrieval_p95_ms"],
            "metrics": metrics,
            "cases": results,
        }
