"""Index the checked-in synthetic corpus and optionally run the release evaluation."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from evaluation import Evaluator, load_evaluation_dataset  # noqa: E402
from ingestion import DocumentParser  # noqa: E402
from rag import RAGPipeline  # noqa: E402
from runtime import create_runtime  # noqa: E402
from storage import TracefoldStore  # noqa: E402


def reset_data(data_dir: Path) -> None:
    for path in (
        data_dir / "tracefold.db",
        data_dir / "tracefold.db-wal",
        data_dir / "tracefold.db-shm",
    ):
        path.unlink(missing_ok=True)
    documents_dir = data_dir / "documents"
    if documents_dir.exists():
        shutil.rmtree(documents_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.getenv("TRACEFOLD_DATA_DIR", PROJECT_ROOT / "data")),
        help="Tracefold data directory (default: repository data/).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing local index before seeding the synthetic corpus.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run the checked-in 12-case evaluation after indexing.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    if args.reset:
        reset_data(data_dir)
    documents_dir = data_dir / "documents"
    documents_dir.mkdir(parents=True, exist_ok=True)

    store = TracefoldStore(data_dir / "tracefold.db")
    runtime = create_runtime()
    rag = RAGPipeline(store, runtime)
    document_parser = DocumentParser()

    indexed = 0
    skipped = 0
    for source in sorted((PROJECT_ROOT / "demo_docs").glob("*.md")):
        payload = source.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if store.find_document_by_hash(digest):
            print(f"skip   {source.name} (already indexed)")
            skipped += 1
            continue
        document_id = f"demo_{source.stem}"
        stored_name = f"{document_id}.md"
        chunks = document_parser.parse(
            source,
            chunk_size=int(os.getenv("TRACEFOLD_CHUNK_SIZE", "240")),
            overlap=int(os.getenv("TRACEFOLD_CHUNK_OVERLAP", "45")),
        )
        embedded = rag.embed_chunks(chunks)
        store.add_document(
            document_id=document_id,
            filename=source.name,
            stored_name=stored_name,
            media_type="text/markdown",
            sha256=digest,
            size_bytes=len(payload),
            chunks=embedded,
        )
        shutil.copy2(source, documents_dir / stored_name)
        print(f"index  {source.name} ({len(chunks)} chunks)")
        indexed += 1

    counts = store.counts()
    print(
        f"ready  {counts['documents']} documents / {counts['chunks']} chunks "
        f"({indexed} indexed, {skipped} skipped)"
    )
    print(runtime.status(probe=False))

    if args.evaluate:
        dataset = load_evaluation_dataset(
            PROJECT_ROOT / "docs" / "eval-dataset.json",
            PROJECT_ROOT / "demo_docs",
        )
        result = Evaluator(rag, store).run(
            dataset["cases"],
            top_k=int(dataset.get("default_top_k", 5)),
            dataset_id=dataset.get("dataset_id"),
        )
        print("evaluation", result["id"], result["metrics"])


if __name__ == "__main__":
    main()
