"""SQLite persistence for Tracefold documents, chunks, jobs, and evaluations."""

from __future__ import annotations

import hashlib
import hmac
import json
import sqlite3
import struct
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def encode_embedding(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def decode_embedding(blob: bytes, dimensions: int) -> list[float]:
    if len(blob) != dimensions * 4:
        raise ValueError("Stored embedding size does not match its dimensions.")
    return list(struct.unpack(f"<{dimensions}f", blob))


def review_answers_sha256(cases: list[dict[str, Any]]) -> str:
    """Digest the exact Turkish case set covered by a manual release review."""

    reviewed_answers = [
        {
            "id": str(case.get("id") or ""),
            "question": str(case.get("question") or ""),
            "answer": str(case.get("answer") or ""),
            "passed": case.get("passed") is True,
        }
        for case in cases
        if str(case.get("locale") or "").casefold().startswith("tr")
    ]
    return hashlib.sha256(
        json.dumps(
            reviewed_answers,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


class TracefoldStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    stored_name TEXT NOT NULL,
                    media_type TEXT,
                    sha256 TEXT NOT NULL UNIQUE,
                    size_bytes INTEGER NOT NULL,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    ordinal INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    page_start INTEGER,
                    page_end INTEGER,
                    section TEXT,
                    token_count INTEGER NOT NULL,
                    embedding BLOB NOT NULL,
                    embedding_dimensions INTEGER NOT NULL,
                    UNIQUE(document_id, ordinal)
                );
                CREATE TABLE IF NOT EXISTS ingestion_jobs (
                    id TEXT PRIMARY KEY,
                    document_id TEXT,
                    status TEXT NOT NULL,
                    progress REAL NOT NULL,
                    message TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS evaluation_runs (
                    id TEXT PRIMARY KEY,
                    metrics_json TEXT NOT NULL,
                    cases_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id, ordinal);
                CREATE INDEX IF NOT EXISTS idx_documents_created ON documents(created_at DESC);
                """
            )

    def create_job(self, *, document_id: str | None = None, message: str = "Queued") -> str:
        job_id = f"job_{uuid.uuid4().hex}"
        now = utc_now()
        with self._connect() as db:
            db.execute(
                "INSERT INTO ingestion_jobs VALUES (?, ?, 'queued', 0, ?, NULL, ?, ?)",
                (job_id, document_id, message, now, now),
            )
        return job_id

    def update_job(
        self,
        job_id: str,
        *,
        status: str,
        progress: float,
        message: str | None = None,
        error: str | None = None,
        document_id: str | None = None,
    ) -> None:
        with self._connect() as db:
            db.execute(
                """UPDATE ingestion_jobs SET status=?, progress=?, message=?, error=?,
                   document_id=COALESCE(?, document_id), updated_at=? WHERE id=?""",
                (status, max(0, min(1, progress)), message, error, document_id, utc_now(), job_id),
            )

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM ingestion_jobs WHERE id=?", (job_id,)).fetchone()
        return dict(row) if row else None

    def find_document_by_hash(self, sha256: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM documents WHERE sha256=?", (sha256,)).fetchone()
        return dict(row) if row else None

    def add_document(
        self,
        *,
        document_id: str,
        filename: str,
        stored_name: str,
        media_type: str | None,
        sha256: str,
        size_bytes: int,
        chunks: list[dict[str, Any]],
    ) -> None:
        now = utc_now()
        with self._lock, self._connect() as db:
            db.execute(
                """INSERT INTO documents
                   (id, filename, stored_name, media_type, sha256, size_bytes, chunk_count,
                    status, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 'ready', ?, ?)""",
                (
                    document_id,
                    filename,
                    stored_name,
                    media_type,
                    sha256,
                    size_bytes,
                    len(chunks),
                    now,
                    now,
                ),
            )
            db.executemany(
                """INSERT INTO chunks
                   (id, document_id, ordinal, text, page_start, page_end, section,
                    token_count, embedding, embedding_dimensions)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        item["id"],
                        document_id,
                        item["ordinal"],
                        item["text"],
                        item.get("page_start"),
                        item.get("page_end"),
                        item.get("section"),
                        item["token_count"],
                        encode_embedding(item["embedding"]),
                        len(item["embedding"]),
                    )
                    for item in chunks
                ],
            )

    def list_documents(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                """SELECT id, filename, media_type, size_bytes, chunk_count, status,
                   created_at, updated_at FROM documents ORDER BY created_at DESC"""
            ).fetchall()
        return [dict(row) for row in rows]

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
        return dict(row) if row else None

    def delete_document(self, document_id: str) -> dict[str, Any] | None:
        document = self.get_document(document_id)
        if not document:
            return None
        with self._connect() as db:
            db.execute("DELETE FROM documents WHERE id=?", (document_id,))
        return document

    def clear_documents(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            documents = [dict(row) for row in db.execute("SELECT * FROM documents").fetchall()]
            db.execute("DELETE FROM documents")
        return documents

    def all_chunks(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                """SELECT c.*, d.filename FROM chunks c
                   JOIN documents d ON d.id=c.document_id ORDER BY d.created_at, c.ordinal"""
            ).fetchall()
        results = []
        for row in rows:
            item = dict(row)
            item["embedding"] = decode_embedding(
                item.pop("embedding"), item["embedding_dimensions"]
            )
            results.append(item)
        return results

    def counts(self) -> dict[str, int]:
        with self._connect() as db:
            documents = db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            chunks = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return {"documents": documents, "chunks": chunks}

    def save_evaluation(self, metrics: dict[str, Any], cases: list[dict[str, Any]]) -> str:
        run_id = f"eval_{uuid.uuid4().hex}"
        created_at = utc_now()
        with self._connect() as db:
            db.execute(
                "INSERT INTO evaluation_runs VALUES (?, ?, ?, ?)",
                (run_id, json.dumps(metrics), json.dumps(cases), created_at),
            )
        return run_id

    def update_evaluation_metrics(self, run_id: str, metrics: dict[str, Any]) -> bool:
        """Replace one run's metrics after an explicit release-review annotation."""

        with self._connect() as db:
            cursor = db.execute(
                "UPDATE evaluation_runs SET metrics_json=? WHERE id=?",
                (json.dumps(metrics), run_id),
            )
        return cursor.rowcount == 1

    def latest_evaluation(self) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM evaluation_runs ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        if not row:
            return None
        metrics = json.loads(row["metrics_json"])
        cases = json.loads(row["cases_json"])
        review = (metrics.get("run_context") or {}).get("manual_language_review")
        if isinstance(review, dict):
            expected_digest = str(review.get("answers_sha256") or "")
            review["verified"] = bool(expected_digest) and hmac.compare_digest(
                expected_digest,
                review_answers_sha256(cases),
            )
        return {
            "id": row["id"],
            "status": "complete",
            "metrics": metrics,
            "cases": cases,
            "created_at": row["created_at"],
            **metrics,
        }
