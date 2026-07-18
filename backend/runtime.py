"""Explicit Foundry Local and deterministic test runtimes."""

from __future__ import annotations

import hashlib
import math
import os
import re
import threading
from collections.abc import Iterable
from typing import Any

DEFAULT_EMBEDDING_MODEL = "qwen3-embedding-0.6b"
DEFAULT_CHAT_MODEL = "qwen2.5-14b"


class RuntimeUnavailable(RuntimeError):
    """Raised when the configured local inference runtime cannot be used."""


class BaseRuntime:
    def status(self, *, probe: bool = False) -> dict[str, Any]:
        raise NotImplementedError

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def generate_stream(self, messages: list[dict[str, str]]) -> Iterable[str]:
        raise NotImplementedError

    def generate(self, messages: list[dict[str, str]]) -> str:
        return "".join(self.generate_stream(messages)).strip()


class FoundryLocalRuntime(BaseRuntime):
    """Foundry Local SDK runtime. Failures are visible; there is no production fallback."""

    def __init__(self) -> None:
        self.embedding_alias = os.getenv("TRACEFOLD_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self.chat_alias = os.getenv("TRACEFOLD_CHAT_MODEL", DEFAULT_CHAT_MODEL)
        self.device_preference = os.getenv("TRACEFOLD_DEVICE", "auto").strip().lower()
        if self.device_preference not in {"auto", "cpu", "gpu"}:
            raise RuntimeUnavailable("TRACEFOLD_DEVICE must be 'auto', 'gpu', or 'cpu'.")
        self._manager: Any = None
        self._embedding_model: Any = None
        self._chat_model: Any = None
        self._embedding_client: Any = None
        self._chat_client: Any = None
        self._ready = False
        self._last_error: str | None = None
        self._lock = threading.RLock()

    def _manager_instance(self) -> Any:
        if self._manager is not None:
            return self._manager
        try:
            from foundry_local_sdk import Configuration, FoundryLocalManager

            if FoundryLocalManager.instance is None:
                FoundryLocalManager.initialize(Configuration(app_name="tracefold"))
            self._manager = FoundryLocalManager.instance
            return self._manager
        except Exception as exc:
            self._last_error = str(exc)
            raise RuntimeUnavailable(
                "Foundry Local could not be initialized. Install/configure Foundry Local and "
                f"restart Tracefold. Details: {exc}"
            ) from exc

    @staticmethod
    def _device(variant: Any) -> str:
        runtime = getattr(variant.info, "runtime", None)
        device = getattr(runtime, "device_type", None)
        return str(getattr(device, "value", device) or "UNKNOWN").upper()

    @staticmethod
    def _execution_provider(variant: Any) -> str | None:
        runtime = getattr(variant.info, "runtime", None)
        return getattr(runtime, "execution_provider", None)

    def _select_variant(self, alias: str) -> Any:
        manager = self._manager_instance()
        registered = {
            item.name for item in manager.discover_eps() if getattr(item, "is_registered", False)
        }
        webgpu = "WebGpuExecutionProvider"
        if self.device_preference != "cpu" and webgpu not in registered:
            try:
                result = manager.download_and_register_eps([webgpu])
                if not result.success and webgpu not in result.registered_eps:
                    raise RuntimeUnavailable(
                        f"WebGPU registration failed: {', '.join(result.failed_eps) or result.status}."
                    )
                registered = {
                    item.name
                    for item in manager.discover_eps()
                    if getattr(item, "is_registered", False)
                }
            except Exception as exc:
                if self.device_preference == "gpu":
                    raise RuntimeUnavailable(
                        "TRACEFOLD_DEVICE=gpu was requested, but WebGPU could not be registered. "
                        f"Details: {exc}"
                    ) from exc

        model = manager.catalog.get_model(alias)
        if model is None:
            raise RuntimeUnavailable(
                f"Model alias '{alias}' was not found in the Foundry Local catalog. "
                "Set TRACEFOLD_CHAT_MODEL or TRACEFOLD_EMBEDDING_MODEL to an installed alias."
            )

        variants = list(model.variants)
        if self.device_preference == "cpu":
            compatible = [variant for variant in variants if self._device(variant) == "CPU"]
        elif self.device_preference == "gpu":
            compatible = [
                variant
                for variant in variants
                if self._device(variant) == "GPU"
                and self._execution_provider(variant) in registered
            ]
        else:
            compatible = [
                variant
                for variant in variants
                if self._device(variant) == "CPU" or self._execution_provider(variant) in registered
            ]
        if not compatible:
            raise RuntimeUnavailable(f"No compatible local variant is available for '{alias}'.")

        cached = [variant for variant in compatible if variant.is_cached]
        if cached:
            # Cached GPU/NPU is preferred, then cached CPU; avoids unnecessary downloads.
            return sorted(
                cached, key=lambda v: {"GPU": 0, "NPU": 1, "CPU": 2}.get(self._device(v), 3)
            )[0]

        gpu = [variant for variant in compatible if self._device(variant) == "GPU"]
        if gpu:
            return gpu[0]
        cpu = [variant for variant in compatible if self._device(variant) == "CPU"]
        if cpu and (self.device_preference == "cpu" or cpu[0].is_cached):
            return cpu[0]
        raise RuntimeUnavailable(
            f"No cached compatible variant is available for '{alias}'. WebGPU could not be used; "
            "run model preparation again or explicitly set TRACEFOLD_DEVICE=cpu."
        )

    def _inspect(self) -> None:
        with self._lock:
            if self._embedding_model is None:
                self._embedding_model = self._select_variant(self.embedding_alias)
            if self._chat_model is None:
                self._chat_model = self._select_variant(self.chat_alias)

    def ensure_ready(self) -> None:
        with self._lock:
            if self._ready:
                return
            try:
                self._inspect()
                for model in (self._embedding_model, self._chat_model):
                    if not model.is_cached:
                        model.download()
                    if not model.is_loaded:
                        model.load()
                self._embedding_client = self._embedding_model.get_embedding_client()
                self._chat_client = self._chat_model.get_chat_client()
                self._chat_client.settings.temperature = 0.0
                self._chat_client.settings.random_seed = int(
                    os.getenv("TRACEFOLD_RANDOM_SEED", "42")
                )
                self._chat_client.settings.max_tokens = int(
                    os.getenv("TRACEFOLD_MAX_OUTPUT_TOKENS", "220")
                )
                self._ready = True
                self._last_error = None
            except RuntimeUnavailable:
                raise
            except Exception as exc:
                self._last_error = str(exc)
                raise RuntimeUnavailable(
                    f"Foundry Local models could not be loaded: {exc}"
                ) from exc

    def _model_status(self, model: Any, alias: str) -> dict[str, Any]:
        if model is None:
            return {
                "alias": alias,
                "variant_id": None,
                "device": None,
                "cached": False,
                "loaded": False,
            }
        return {
            "alias": alias,
            "variant_id": model.id,
            "device": self._device(model),
            "execution_provider": self._execution_provider(model),
            "cached": bool(model.is_cached),
            "loaded": bool(model.is_loaded),
        }

    def status(self, *, probe: bool = False) -> dict[str, Any]:
        if probe:
            try:
                self._inspect()
                self._last_error = None
            except RuntimeUnavailable as exc:
                self._last_error = str(exc)
        return {
            "runtime": "foundry_local",
            "ready": self._ready,
            "device_preference": self.device_preference,
            "embedding_model": self._model_status(self._embedding_model, self.embedding_alias),
            "chat_model": self._model_status(self._chat_model, self.chat_alias),
            "error": self._last_error,
        }

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.ensure_ready()
        if not texts:
            return []
        response = self._embedding_client.generate_embeddings(texts)
        return [list(item.embedding) for item in sorted(response.data, key=lambda item: item.index)]

    def generate_stream(self, messages: list[dict[str, str]]) -> Iterable[str]:
        self.ensure_ready()
        in_thinking = False
        for event in self._chat_client.complete_streaming_chat(messages):
            if event.choices and event.choices[0].delta.content:
                token = event.choices[0].delta.content
                # Qwen3 thinking mode: skip <think>...</think> blocks
                if "<think>" in token:
                    in_thinking = True
                    token = token.split("<think>")[0]
                if "</think>" in token:
                    in_thinking = False
                    token = token.split("</think>", 1)[1] if "</think>" in token else ""
                if in_thinking:
                    continue
                if token:
                    yield token


class FakeRuntime(BaseRuntime):
    """Deterministic, local-only runtime enabled explicitly for automated tests."""

    dimensions = 384

    def status(self, *, probe: bool = False) -> dict[str, Any]:
        return {
            "runtime": "fake",
            "ready": True,
            "embedding_model": {
                "alias": "tracefold-test-embedding",
                "variant_id": "fake-embedding-v1",
                "device": "CPU",
                "cached": True,
                "loaded": True,
            },
            "chat_model": {
                "alias": "tracefold-test-chat",
                "variant_id": "fake-chat-v1",
                "device": "CPU",
                "cached": True,
                "loaded": True,
            },
            "error": None,
        }

    @staticmethod
    def _terms(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dimensions
            terms = self._terms(text)
            features = terms + [f"{a}_{b}" for a, b in zip(terms, terms[1:], strict=False)]
            for feature in features:
                digest = hashlib.sha256(feature.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "little") % self.dimensions
                vector[index] += 1.0
            norm = math.sqrt(sum(value * value for value in vector))
            vectors.append([value / norm for value in vector] if norm else vector)
        return vectors

    def generate_stream(self, messages: list[dict[str, str]]) -> Iterable[str]:
        context = messages[0]["content"]
        question = messages[-1]["content"]
        question_terms = set(self._terms(question)) - {
            "a",
            "an",
            "the",
            "is",
            "are",
            "what",
            "where",
            "when",
            "who",
            "how",
            "does",
            "do",
        }
        matches = re.findall(r"\[(\d+)\][^\n]*\n(.+?)(?=\n\n\[|\Z)", context, flags=re.DOTALL)
        ranked = sorted(
            matches,
            key=lambda item: len(question_terms & set(self._terms(item[1]))),
            reverse=True,
        )
        if not ranked or not question_terms.intersection(self._terms(ranked[0][1])):
            answer = "I couldn't find enough evidence in the indexed documents to answer that."
        else:
            citation, text = ranked[0]
            sentences = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
            sentence = max(
                sentences,
                key=lambda value: len(question_terms & set(self._terms(value))),
                default=text,
            )
            answer = f"{sentence.strip()} [{citation}]"
        yield from re.findall(r"\S+\s*", answer)


def create_runtime() -> BaseRuntime:
    mode = os.getenv("TRACEFOLD_RUNTIME", "foundry").strip().lower()
    if mode == "fake":
        return FakeRuntime()
    if mode in {"foundry", "foundry_local"}:
        return FoundryLocalRuntime()
    raise RuntimeUnavailable(
        "TRACEFOLD_RUNTIME must be 'foundry' (production) or 'fake' (automated tests only)."
    )
