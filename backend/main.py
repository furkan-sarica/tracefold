"""Tracefold: private, source-grounded RAG powered by Foundry Local."""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
import uuid
from pathlib import Path
from typing import Annotated, Any, Literal

from evaluation import Evaluator, load_evaluation_dataset
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from ingestion import SUPPORTED_EXTENSIONS, DocumentParser
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from rag import RAGPipeline
from runtime import BaseRuntime, RuntimeUnavailable, create_runtime
from storage import TracefoldStore

MAX_UPLOAD_BYTES = int(os.getenv("TRACEFOLD_MAX_UPLOAD_MB", "25")) * 1024 * 1024


class ChatRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    question: str = Field(
        min_length=1,
        max_length=4000,
        validation_alias=AliasChoices("question", "query"),
    )
    top_k: int = Field(default=5, ge=1, le=12)
    locale: Literal["en", "tr"] = "en"


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str | None = None
    type: str | None = None
    category: str | None = None
    question: str = Field(min_length=1, max_length=4000)
    locale: Literal["en", "tr"] = "en"
    paired_case_id: str | None = None
    answerable: bool = True
    expected_sources: list[str] = Field(default_factory=list)
    expected_terms: list[str] = Field(default_factory=list)
    required_facts: list[str] = Field(default_factory=list)
    expected_answer_snippet: str | None = None
    must_refuse: bool = False


class EvaluationRequest(BaseModel):
    cases: list[EvaluationCase] | None = Field(default=None, max_length=100)
    top_k: int = Field(default=5, ge=1, le=12)


def safe_filename(filename: str) -> str:
    basename = re.split(r"[/\\]", filename)[-1]
    normalized = unicodedata.normalize("NFC", basename)
    cleaned = "".join(
        "_" if unicodedata.category(character).startswith("C") else character
        for character in normalized
    )
    cleaned = re.sub(r"[<>:\"/\\|?*]+", "_", cleaned)
    cleaned = re.sub(r"\s+", "_", cleaned).strip(" ._")
    if cleaned in {"", ".", ".."}:
        cleaned = f"document_{uuid.uuid4().hex[:8]}.txt"
    if len(cleaned) > 180:
        suffix = Path(cleaned).suffix
        if suffix and len(suffix) < 32:
            cleaned = cleaned[: 180 - len(suffix)].rstrip(" ._") + suffix
        else:
            cleaned = cleaned[:180].rstrip(" ._")
    return cleaned or f"document_{uuid.uuid4().hex[:8]}.txt"


def sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(
    *,
    data_dir: str | Path | None = None,
    runtime: BaseRuntime | None = None,
) -> FastAPI:
    root_data = Path(
        data_dir or os.getenv("TRACEFOLD_DATA_DIR") or Path(__file__).parent.parent / "data"
    )
    documents_dir = root_data / "documents"
    documents_dir.mkdir(parents=True, exist_ok=True)
    store = TracefoldStore(root_data / "tracefold.db")
    selected_runtime = runtime or create_runtime()
    parser = DocumentParser()
    rag = RAGPipeline(store, selected_runtime)
    evaluator = Evaluator(rag, store)

    def combined_runtime_status(*, probe: bool = False) -> dict[str, Any]:
        status = selected_runtime.status(probe=probe)
        translation_status = rag.translation.status(probe=probe)
        return {
            **status,
            "ready": bool(status.get("ready")) and bool(translation_status.get("ready")),
            "translation": translation_status,
        }

    application = FastAPI(
        title="Tracefold",
        version="3.0.0",
        description="Answers you can trace — fully local RAG with Foundry Local.",
    )
    allowed_origins = [
        item.strip()
        for item in os.getenv(
            "TRACEFOLD_ALLOWED_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8000,http://127.0.0.1:8000",
        ).split(",")
        if item.strip()
    ]
    application.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
    )

    application.state.store = store
    application.state.runtime = selected_runtime
    application.state.translation = rag.translation
    application.state.rag = rag
    application.state.documents_dir = documents_dir

    @application.exception_handler(RuntimeUnavailable)
    async def runtime_unavailable_handler(_request, exc: RuntimeUnavailable):
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=503,
            content={
                "detail": str(exc),
                "code": "runtime_unavailable",
                "runtime": combined_runtime_status(),
            },
        )

    frontend_dist = Path(
        os.getenv("TRACEFOLD_FRONTEND_DIST") or Path(__file__).parent.parent / "frontend" / "dist"
    )

    @application.get("/")
    async def root():
        if (frontend_dist / "index.html").is_file():
            return FileResponse(frontend_dist / "index.html")
        return JSONResponse(
            {"service": "Tracefold", "tagline": "Answers you can trace.", "docs": "/docs"}
        )

    @application.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "healthy",
            "service": "Tracefold",
            "runtime": combined_runtime_status(probe=False),
            **store.counts(),
        }

    @application.get("/api/runtime")
    async def runtime_status() -> dict[str, Any]:
        status = combined_runtime_status(probe=True)
        translation_status = status["translation"]
        return {
            **status,
            "available": (
                status.get("error") is None
                and translation_status.get("error") is None
                and bool(translation_status.get("ready"))
            ),
        }

    @application.post("/api/documents", status_code=201)
    async def upload_document(file: Annotated[UploadFile, File(...)]) -> dict[str, Any]:
        if not file.filename:
            raise HTTPException(status_code=400, detail="A filename is required.")
        display_name = safe_filename(file.filename)
        extension = Path(display_name).suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type. Use: {', '.join(sorted(SUPPORTED_EXTENSIONS))}.",
            )

        payload = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File is larger than the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
            )
        if not payload:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")

        digest = hashlib.sha256(payload).hexdigest()
        duplicate = store.find_document_by_hash(digest)
        if duplicate:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "This document is already indexed.",
                    "document_id": duplicate["id"],
                },
            )

        document_id = f"doc_{uuid.uuid4().hex}"
        stored_name = f"{document_id}{extension}"
        stored_path = documents_dir / stored_name
        job_id = store.create_job(document_id=document_id, message="Validating document")
        stored_path.write_bytes(payload)
        try:
            store.update_job(job_id, status="processing", progress=0.2, message="Extracting text")
            parsed_chunks = parser.parse(
                stored_path,
                chunk_size=int(os.getenv("TRACEFOLD_CHUNK_SIZE", "240")),
                overlap=int(os.getenv("TRACEFOLD_CHUNK_OVERLAP", "45")),
            )
            store.update_job(
                job_id, status="processing", progress=0.55, message="Creating embeddings"
            )
            embedded_chunks = rag.embed_chunks(parsed_chunks)
            store.update_job(job_id, status="processing", progress=0.9, message="Saving index")
            store.add_document(
                document_id=document_id,
                filename=display_name,
                stored_name=stored_name,
                media_type=file.content_type,
                sha256=digest,
                size_bytes=len(payload),
                chunks=embedded_chunks,
            )
            store.update_job(
                job_id,
                status="completed",
                progress=1,
                message="Ready to search",
                document_id=document_id,
            )
        except RuntimeUnavailable:
            stored_path.unlink(missing_ok=True)
            store.update_job(
                job_id,
                status="failed",
                progress=1,
                message="Foundry Local is unavailable",
                error="The configured local models could not be loaded.",
            )
            raise
        except ValueError as exc:
            stored_path.unlink(missing_ok=True)
            store.update_job(
                job_id, status="failed", progress=1, message="Ingestion failed", error=str(exc)
            )
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            stored_path.unlink(missing_ok=True)
            store.update_job(
                job_id, status="failed", progress=1, message="Ingestion failed", error=str(exc)
            )
            raise HTTPException(
                status_code=500, detail=f"Document ingestion failed: {exc}"
            ) from exc

        return {
            "status": "indexed",
            "job_id": job_id,
            "document": store.get_document(document_id),
            "runtime": combined_runtime_status(),
        }

    @application.get("/api/documents")
    async def list_documents() -> dict[str, Any]:
        counts = store.counts()
        return {
            "documents": store.list_documents(),
            "document_count": counts["documents"],
            "chunk_count": counts["chunks"],
            "chunks": counts["chunks"],
        }

    @application.delete("/api/documents")
    async def clear_documents() -> dict[str, Any]:
        removed = store.clear_documents()
        for document in removed:
            (documents_dir / document["stored_name"]).unlink(missing_ok=True)
        return {"status": "cleared", "deleted": len(removed)}

    @application.delete("/api/documents/{document_id}")
    async def delete_document(document_id: str) -> dict[str, Any]:
        document = store.delete_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found.")
        (documents_dir / document["stored_name"]).unlink(missing_ok=True)
        return {"status": "deleted", "document_id": document_id}

    @application.get("/api/jobs/{job_id}")
    async def get_job(job_id: str) -> dict[str, Any]:
        job = store.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Ingestion job not found.")
        return job

    @application.post("/api/chat")
    async def chat(request: ChatRequest) -> dict[str, Any]:
        return {
            **rag.answer(request.question.strip(), top_k=request.top_k, locale=request.locale),
            "runtime": combined_runtime_status(),
        }

    @application.post("/api/chat/stream")
    async def chat_stream(request: ChatRequest) -> StreamingResponse:
        metadata, tokens = rag.answer_stream(
            request.question.strip(), top_k=request.top_k, locale=request.locale
        )

        def events():
            answer = ""
            try:
                yield sse(
                    "retrieval",
                    {
                        "retrieval_ms": metadata["retrieval_ms"],
                        "matches": metadata["matches"],
                        "threshold": metadata["threshold"],
                    },
                )
                for token in tokens:
                    answer += token
                    yield sse("token", {"text": token})
                citations = [] if metadata["refused"] else metadata["citations"]
                yield sse("citations", {"citations": citations})
                yield sse(
                    "done",
                    {"answer": answer.strip(), "refused": metadata["refused"]},
                )
            except Exception as exc:
                yield sse("error", {"code": "generation_failed", "message": str(exc)})

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @application.post("/api/evaluations/run")
    async def run_evaluation(request: EvaluationRequest | None = None) -> dict[str, Any]:
        request = request or EvaluationRequest()
        dataset_id: str | None = None
        if request.cases:
            cases = [case.model_dump() for case in request.cases]
        else:
            project_root = Path(__file__).parent.parent
            dataset = load_evaluation_dataset(
                project_root / "docs" / "eval-dataset.json",
                project_root / "demo_docs",
            )
            dataset_id = dataset.get("dataset_id")
            cases = dataset["cases"]
        try:
            return evaluator.run(cases, top_k=request.top_k, dataset_id=dataset_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @application.get("/api/evaluations/latest")
    async def latest_evaluation() -> dict[str, Any]:
        run = store.latest_evaluation()
        if not run:
            raise HTTPException(status_code=404, detail="No evaluation has been run yet.")
        return run

    @application.get("/{asset_path:path}", include_in_schema=False)
    async def frontend_asset(asset_path: str):
        if asset_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API route not found.")
        candidate = (frontend_dist / asset_path).resolve()
        try:
            candidate.relative_to(frontend_dist.resolve())
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="Asset not found.") from exc
        if candidate.is_file():
            return FileResponse(candidate)
        index = frontend_dist / "index.html"
        if index.is_file():
            return FileResponse(index)
        raise HTTPException(status_code=404, detail="Frontend build not found.")

    return application


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
