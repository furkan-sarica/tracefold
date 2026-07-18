"""Download and validate the Foundry Local models used by Tracefold.

Run this once while online. The application can then index documents and answer
questions with the network disabled.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from collections.abc import Iterable
from pathlib import Path

from foundry_local_sdk import Configuration, FoundryLocalManager

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from translation import MarianTranslationEngine  # noqa: E402

DEFAULT_EMBEDDING_MODEL = "qwen3-embedding-0.6b"
DEFAULT_CHAT_MODEL = "qwen2.5-7b"
WEBGPU_EP = "WebGpuExecutionProvider"


def progress(label: str):
    """Return a compact progress callback for SDK downloads."""

    last_bucket = -1

    def report(percent: float) -> None:
        nonlocal last_bucket
        bucket = int(percent // 5)
        if bucket != last_bucket or percent >= 100:
            last_bucket = bucket
            print(f"{label:<28} {percent:5.1f}%", flush=True)

    return report


def select_variant(model, *, prefer_gpu: bool, registered_eps: Iterable[str]):
    """Pick a cached variant on the requested device, then download that device."""

    variants = list(model.variants)
    registered = set(registered_eps)

    def device(variant) -> str:
        runtime = getattr(variant.info, "runtime", None)
        value = getattr(getattr(runtime, "device_type", None), "value", "")
        return str(value).upper()

    gpu_candidates = [variant for variant in variants if device(variant) == "GPU"]
    cpu_candidates = [variant for variant in variants if device(variant) == "CPU"]

    if prefer_gpu and WEBGPU_EP in registered and gpu_candidates:
        candidates = gpu_candidates
    else:
        candidates = cpu_candidates

    if not candidates:
        requested = "GPU" if prefer_gpu and WEBGPU_EP in registered else "CPU"
        raise RuntimeError(f"No {requested} model variant is available for {model.id}.")

    selected = next((variant for variant in candidates if variant.is_cached), candidates[0])
    model.select_variant(selected)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", action="store_true", help="Prefer CPU model variants.")
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Cache models without loading them and running inference.",
    )
    args = parser.parse_args()

    embedding_alias = os.getenv("TRACEFOLD_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    chat_alias = os.getenv("TRACEFOLD_CHAT_MODEL", DEFAULT_CHAT_MODEL)

    FoundryLocalManager.initialize(Configuration(app_name="tracefold"))
    manager = FoundryLocalManager.instance

    eps = manager.discover_eps()
    if not args.cpu and not any(ep.name == WEBGPU_EP and ep.is_registered for ep in eps):
        print("Preparing the WebGPU execution provider...")

        ep_buckets: dict[str, int] = {}

        def ep_progress(name: str, percent: float) -> None:
            bucket = int(percent // 5)
            if ep_buckets.get(name) != bucket or percent >= 100:
                ep_buckets[name] = bucket
                print(f"{name:<28} {percent:5.1f}%", flush=True)

        manager.download_and_register_eps(
            [WEBGPU_EP],
            ep_progress,
        )
        eps = manager.discover_eps()

    registered_eps = [ep.name for ep in eps if ep.is_registered]
    print("Execution providers:", ", ".join(registered_eps) or "CPU")

    models = []
    for alias in (embedding_alias, chat_alias):
        model = manager.catalog.get_model(alias)
        if model is None:
            raise SystemExit(f"Model alias is not present in the Foundry Local catalog: {alias}")
        model = select_variant(model, prefer_gpu=not args.cpu, registered_eps=registered_eps)
        print(f"Selected {alias}: {model.id}")
        if not model.is_cached:
            model.download(progress(alias))
        print(f"Cached at {model.get_path()}")
        models.append(model)

    print("Preparing pinned offline English-Turkish translation models...")
    translation = MarianTranslationEngine(local_files_only=False)
    translation.prepare(smoke=not args.skip_smoke)
    print("Translation models:", translation.status(probe=False))
    del translation
    gc.collect()

    if args.skip_smoke:
        return

    embedding_model, chat_model = models
    for model in models:
        if not model.is_loaded:
            print(f"Loading {model.id}...")
            model.load()

    embedding = embedding_model.get_embedding_client().generate_embedding(
        "Tracefold keeps private documents on the device."
    )
    dimensions = len(embedding.data[0].embedding)
    response = chat_model.get_chat_client().complete_chat(
        [
            {
                "role": "system",
                "content": (
                    "Answer only from this context: Tracefold keeps private documents "
                    "on the device."
                ),
            },
            {"role": "user", "content": "Where are the documents kept?"},
        ]
    )
    answer = response.choices[0].message.content or ""
    print(f"Embedding smoke test: {dimensions} dimensions")
    print(f"Chat smoke test: {answer.strip()}")


if __name__ == "__main__":
    main()
