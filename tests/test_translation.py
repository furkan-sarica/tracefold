from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from translation import (
    EN_TO_TR_REVISION,
    TR_TO_EN_REVISION,
    BaseTranslationEngine,
    ControlledTranslationService,
    MarianTranslationEngine,
    TerminologyGlossary,
    TranslationRejected,
)


def test_glossary_protects_longest_terms_and_exact_facts() -> None:
    glossary = TerminologyGlossary()
    protected, replacements = glossary.protect(
        "Call the Residence Duty Phone at +90 312 555 0149 before 19:30 and email "
        "program@northstar.example.",
        source_locale="en",
        target_locale="tr",
    )

    assert protected == "Call the ZXQ0QXZ at ZXQ1QXZ before 19:30 and email ZXQ2QXZ."
    assert [item.value for item in replacements] == [
        "Yurt Nöbet Telefonu",
        "+90 312 555 0149",
        "program@northstar.example",
    ]
    facts = glossary.exact_facts(
        "Call the Residence Duty Phone at +90 312 555 0149 before 19:30 and email "
        "program@northstar.example."
    )
    assert facts[("phone", "903125550149")] == 1
    assert facts[("time", "19:30")] == 1
    assert facts[("email", "program@northstar.example")] == 1


def test_restore_rejects_lost_sentinels_and_normalizes_protected_suffixes() -> None:
    glossary = TerminologyGlossary()
    _protected, replacements = glossary.protect(
        "Contact the Program Office and Safety Officer by +90 312 555 0149.",
        source_locale="en",
        target_locale="tr",
    )

    restored = glossary.restore(
        "ZXQ0QXZ'ne başvurun, ZXQ1QXZ'yla görüşün ve ZXQ2QXZ'yi arayın.",
        replacements,
    )
    assert restored == (
        "Program Ofisine başvurun, Güvenlik Sorumlusuyla görüşün ve "
        "+90 312 555 0149 numaralı telefonu arayın."
    )

    with pytest.raises(TranslationRejected, match="changed or lost"):
        glossary.restore("ZXQ0QXZ ile görüşün.", replacements)


class ScriptedEngine(BaseTranslationEngine):
    def __init__(self, outputs: list[list[str]]) -> None:
        self.outputs = outputs

    def translate_batch(self, texts, *, source_locale, target_locale):
        output = self.outputs.pop(0)
        assert len(output) == len(texts)
        return output

    def status(self, *, probe=False):
        return {"runtime": "scripted", "ready": True}


def test_controlled_claim_translation_requires_round_trip_recall(monkeypatch) -> None:
    monkeypatch.setenv("TRACEFOLD_TRANSLATION_ROUNDTRIP_THRESHOLD", "0.60")
    accepted = ControlledTranslationService(
        ScriptedEngine(
            [
                ["19:30'dan önce ZXQ0QXZ'yi arayın"],
                ["Call +90 312 555 0149 before 19:30"],
            ]
        )
    )
    assert accepted.claims_to_turkish(["Call +90 312 555 0149 before 19:30"]) == [
        "19:30'dan önce +90 312 555 0149 numaralı telefonu arayın"
    ]

    rejected = ControlledTranslationService(
        ScriptedEngine([["Tamamen ilgisiz metin"], ["Unrelated statement"]])
    )
    with pytest.raises(TranslationRejected, match="Round-trip"):
        rejected.claims_to_turkish(["Summarize all findings clearly"])


def test_marian_model_revisions_are_release_pinned() -> None:
    engine = MarianTranslationEngine()
    status = engine.status()

    assert status["models"]["tr_to_en"]["revision"] == TR_TO_EN_REVISION
    assert status["models"]["en_to_tr"]["revision"] == EN_TO_TR_REVISION
    assert len(TR_TO_EN_REVISION) == len(EN_TO_TR_REVISION) == 40


def test_marian_cache_readiness_requires_tokenizer_and_weights(monkeypatch) -> None:
    import huggingface_hub

    def config_only(_model_id, filename, *, revision):
        assert len(revision) == 40
        return "/cache/config.json" if filename == "config.json" else None

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", config_only)
    assert MarianTranslationEngine._snapshot_cached("tr_to_en") is False

    def complete_snapshot(_model_id, filename, *, revision):
        assert len(revision) == 40
        if filename == "pytorch_model.bin":
            return None
        return f"/cache/{filename}"

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", complete_snapshot)
    assert MarianTranslationEngine._snapshot_cached("tr_to_en") is True


@pytest.mark.parametrize(
    ("source", "translated", "error"),
    [
        (
            "Complete checkout no later than 10:00",
            "ZXQ0QXZ'yi 10:00'dan sonra tamamlayın",
            "temporal relation",
        ),
        (
            "Call +90 312 555 0149 before 19:30",
            "19:30 numaralı telefonu ZXQ0QXZ'den önce arayın",
            "temporal relation",
        ),
        (
            "Email the update to info@example.com",
            "Güncellemeyi ZXQ0QXZ'den gönderin",
            "critical predicates",
        ),
        (
            "The drill is Tuesday, 4 August, at 11:30",
            "Tatbikat 4 Ağustos Cuma günü saat 11:30'da",
            "exact numeric/contact fact",
        ),
        (
            "Use 16 GB RAM, 60 GB storage, and Wi-Fi 5",
            "60 GB RAM, 16 GB depolama ve Wi-Fi 5 kullanın",
            "unit or role",
        ),
        (
            "The participant must not enter the Fabrication Studio",
            "Katılımcı ZXQ0QXZ'ye girebilir",
            "critical modalities",
        ),
        (
            "The participant may enter the Fabrication Studio",
            "Katılımcı ZXQ0QXZ'ye girebilir ancak girmemelidir",
            "critical modalities",
        ),
    ],
)
def test_controlled_translation_rejects_relation_fact_and_modality_changes(
    source,
    translated,
    error,
) -> None:
    service = ControlledTranslationService(ScriptedEngine([[translated]]))
    with pytest.raises(TranslationRejected, match=error):
        service.claims_to_turkish([source])


def test_extension_and_international_phone_are_bound_to_call_predicate() -> None:
    service = ControlledTranslationService(
        ScriptedEngine(
            [
                ["Dahili numarası 5555 veya ZXQ0QXZ'yi arayın"],
                ["Call extension 5555 or phone +90 312 555 0155"],
            ]
        )
    )

    assert service.claims_to_turkish(["Call extension 5555 or phone +90 312 555 0155"]) == [
        "Dahili numarası 5555 veya +90 312 555 0155 numaralı telefonu arayın"
    ]
