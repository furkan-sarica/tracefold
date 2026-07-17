"""Reproducible retrieval, citation, answer, and refusal evaluation."""

from __future__ import annotations

import json
import math
import os
import platform
import re
import time
from pathlib import Path
from typing import Any

from rag import RAGPipeline
from storage import TracefoldStore


def load_evaluation_dataset(dataset_path: Path, demo_docs_dir: Path) -> dict[str, Any]:
    """Load the checked-in dataset and validate every named source against demo_docs/."""
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("cases"), list) or not payload["cases"]:
        raise ValueError("Evaluation dataset must contain a non-empty 'cases' list.")
    resolved_sources: dict[str, str] = {}
    for case in payload["cases"]:
        for filename in case.get("expected_sources", []):
            if Path(filename).name != filename:
                raise ValueError(f"Unsafe evaluation source name: {filename}")
            source_path = demo_docs_dir / filename
            if not source_path.is_file():
                raise ValueError(f"Evaluation source does not exist in demo_docs/: {filename}")
            resolved_sources[filename] = str(source_path.resolve())
    payload["resolved_sources"] = resolved_sources
    return payload


class Evaluator:
    def __init__(self, rag: RAGPipeline, store: TracefoldStore):
        self.rag = rag
        self.store = store

    @staticmethod
    def _tokens(value: str) -> set[str]:
        stopwords = {
            "a", "an", "and", "are", "at", "be", "by", "for", "from", "in", "is",
            "it", "of", "on", "or", "the", "to", "with",
        }
        return {token for token in re.findall(r"[a-z0-9]+", value.lower()) if token not in stopwords}

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
        """Token recall with a hard guard for omitted numeric facts."""

        expected_numbers = set(re.findall(r"\d+", fact))
        answer_numbers = set(re.findall(r"\d+", answer))
        if expected_numbers and not expected_numbers.issubset(answer_numbers):
            return 0.0
        return cls._snippet_recall(answer, fact)

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
            response = self.rag.answer(case["question"], top_k=top_k)
            expected_sources = set(case.get("expected_sources") or [])
            retrieved_sources = {item["filename"] for item in response["matches"]}
            cited_sources = {item["filename"] for item in response["citations"]}
            must_refuse = bool(case.get("must_refuse") or not case.get("answerable", True))
            answerable = not must_refuse
            source_hits = expected_sources & retrieved_sources
            source_recall = (
                len(source_hits) / len(expected_sources) if expected_sources else None
            )
            citation_source_precision = (
                len(cited_sources & expected_sources) / len(cited_sources)
                if cited_sources
                else (0.0 if answerable else None)
            )
            citation_valid = bool(cited_sources) and not response["refused"] if answerable else None
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
            refusal_correct = response["refused"] == must_refuse
            passed = (
                refusal_correct
                if not answerable
                else bool(source_hits)
                and citation_valid
                and refusal_correct
                and (required_fact_recall is None or required_fact_recall >= 0.50)
            )
            results.append(
                {
                    "id": case.get("id"),
                    "type": case.get("type", "answerable" if answerable else "unanswerable"),
                    "category": case.get("category"),
                    "question": case["question"],
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
                    "snippet_token_recall": self._snippet_recall(
                        response["answer"], case.get("expected_answer_snippet")
                    ),
                    "required_fact_recall": required_fact_recall,
                    "term_match": term_match,
                    "refusal_correct": refusal_correct,
                    "passed": passed,
                    "retrieval_ms": response["retrieval_ms"],
                }
            )

        answerable_results = [item for item in results if item["answerable"]]
        negative_results = [item for item in results if not item["answerable"]]

        def average(items: list[dict[str, Any]], key: str) -> float | None:
            values = [item[key] for item in items if item[key] is not None]
            return round(sum(float(value) for value in values) / len(values), 4) if values else None

        retrieval_values = sorted(item["retrieval_ms"] for item in results)
        p95_index = max(0, math.ceil(0.95 * len(retrieval_values)) - 1)
        counts = self.store.counts()
        metrics = {
            "total_cases": len(results),
            "answerable_cases": len(answerable_results),
            "negative_cases": len(negative_results),
            "passed_cases": sum(bool(item["passed"]) for item in results),
            "hit_at_k": average(answerable_results, "hit"),
            "source_recall_at_k": average(answerable_results, "source_recall"),
            "citation_precision": average(answerable_results, "citation_precision"),
            "citation_accuracy": average(answerable_results, "citation_correct"),
            "snippet_token_recall": average(answerable_results, "snippet_token_recall"),
            "required_fact_recall": average(answerable_results, "required_fact_recall"),
            "custom_term_accuracy": average(answerable_results, "term_match"),
            "refusal_accuracy": average(negative_results, "refusal_correct"),
            "average_retrieval_ms": round(
                sum(item["retrieval_ms"] for item in results) / len(results), 2
            ),
            "retrieval_p95_ms": round(retrieval_values[p95_index], 2),
            "model_warmup_ms": model_warmup_ms,
            "top_k": top_k,
            "retrieval_threshold": self.rag.threshold,
            "run_context": {
                "dataset_id": dataset_id,
                "runtime": self.rag.runtime.status(probe=False),
                "platform": {
                    "system": platform.system(),
                    "release": platform.release(),
                    "machine": platform.machine(),
                },
                "configuration": {
                    "chunk_size": int(os.getenv("TRACEFOLD_CHUNK_SIZE", "240")),
                    "chunk_overlap": int(os.getenv("TRACEFOLD_CHUNK_OVERLAP", "45")),
                    "top_k": top_k,
                    "retrieval_threshold": self.rag.threshold,
                    "evidence_floor": self.rag.evidence_floor,
                    "max_chunks_per_document": self.rag.max_chunks_per_document,
                    "context_char_budget": int(
                        os.getenv("TRACEFOLD_CONTEXT_CHAR_BUDGET", "1800")
                    ),
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
