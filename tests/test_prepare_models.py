from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prepare_models import WEBGPU_EP, select_variant


class Variant:
    def __init__(self, variant_id: str, device: str, *, cached: bool) -> None:
        self.id = variant_id
        self.is_cached = cached
        self.info = SimpleNamespace(
            runtime=SimpleNamespace(device_type=SimpleNamespace(value=device))
        )


class Model:
    def __init__(self, variants: list[Variant]) -> None:
        self.id = "demo-model"
        self.variants = variants
        self.selected: Variant | None = None

    def select_variant(self, variant: Variant) -> None:
        self.selected = variant


def test_cpu_request_never_reuses_cached_gpu_variant() -> None:
    gpu = Variant("gpu-cached", "GPU", cached=True)
    cpu = Variant("cpu-download", "CPU", cached=False)
    model = Model([gpu, cpu])

    select_variant(model, prefer_gpu=False, registered_eps=[WEBGPU_EP])

    assert model.selected is cpu


def test_gpu_request_prefers_cached_gpu_over_cached_cpu() -> None:
    cpu = Variant("cpu-cached", "CPU", cached=True)
    gpu_download = Variant("gpu-download", "GPU", cached=False)
    gpu_cached = Variant("gpu-cached", "GPU", cached=True)
    model = Model([cpu, gpu_download, gpu_cached])

    select_variant(model, prefer_gpu=True, registered_eps=[WEBGPU_EP])

    assert model.selected is gpu_cached


def test_auto_falls_back_to_cpu_when_webgpu_is_unavailable() -> None:
    gpu = Variant("gpu-cached", "GPU", cached=True)
    cpu = Variant("cpu-cached", "CPU", cached=True)
    model = Model([gpu, cpu])

    select_variant(model, prefer_gpu=True, registered_eps=[])

    assert model.selected is cpu
