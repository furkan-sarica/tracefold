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

    assert (
        TerminologyGlossary._term_with_suffix("ödünç alınan nevresim torbası", "lerini")
        == "ödünç alınan nevresim torbasını"
    )


def test_turkish_glossary_protects_inflected_terms() -> None:
    glossary = TerminologyGlossary()
    protected, replacements = glossary.protect(
        "Zorunlu güvenlik oryantasyonuna katılamazsam ne olur?",
        source_locale="tr",
        target_locale="en",
    )

    assert protected == "Zorunlu ZXQ0QXZ katılamazsam ne olur?"
    assert [item.value for item in replacements] == ["safety induction"]
    assert [item.relation for item in replacements] == ["to"]


def test_turkish_glossary_carries_case_relations_into_english_terms() -> None:
    glossary = TerminologyGlossary()
    protected, replacements = glossary.protect(
        "Program Ofisine gitmeliyim ve Program Ofisiyle görüşmeliyim.",
        source_locale="tr",
        target_locale="en",
    )

    assert protected == "ZXQ0QXZ gitmeliyim ve ZXQ1QXZ görüşmeliyim."
    assert [item.value for item in replacements] == [
        "Program Office",
        "Program Office",
    ]
    assert [item.relation for item in replacements] == ["to", "with"]
    assert (
        glossary.restore("I should go ZXQ0QXZ and meet ZXQ1QXZ.", replacements)
        == "I should go to Program Office and meet with Program Office."
    )

    _protected, induction = glossary.protect(
        "Güvenlik oryantasyonuna katılamam.",
        source_locale="tr",
        target_locale="en",
    )
    assert glossary.restore("I cannot attend ZXQ0QXZ.", induction) == (
        "I cannot attend safety induction."
    )
    assert glossary.restore("I cannot attend the ZXQ0QXZ.", induction) == (
        "I cannot attend the safety induction."
    )


def test_query_protection_merges_adjacent_values_and_canonicalizes_textual_dates() -> None:
    glossary = TerminologyGlossary()
    protected, replacements = glossary.protect(
        "Northstar Summer Lab 2026 için 15 Ağustos'taki 12:30 servisi",
        source_locale="tr",
        target_locale="en",
        protect_exact=True,
    )

    assert protected == "ZXQ0QXZ için ZXQ1QXZ"
    assert [item.value for item in replacements] == [
        "Northstar Summer Lab 2026",
        "15 August 12:30 shuttle",
    ]


@pytest.mark.parametrize(
    ("turkish", "english"),
    [
        ("5 Ocak 2026", "5 January 2026"),
        ("6 Şubat 2026", "6 February 2026"),
        ("7 Mart 2026", "7 March 2026"),
        ("8 Nisan 2026", "8 April 2026"),
        ("9 Mayıs 2026", "9 May 2026"),
        ("10 Haziran 2026", "10 June 2026"),
        ("11 Temmuz 2026", "11 July 2026"),
        ("12 Ağustos 2026", "12 August 2026"),
        ("13 Eylül 2026", "13 September 2026"),
        ("14 Ekim 2026", "14 October 2026"),
        ("15 Kasım 2026", "15 November 2026"),
        ("16 Aralık 2026", "16 December 2026"),
    ],
)
def test_query_protection_translates_every_turkish_month(turkish: str, english: str) -> None:
    glossary = TerminologyGlossary()
    protected, replacements = glossary.protect(
        f"Etkinlik {turkish} tarihinde",
        source_locale="tr",
        target_locale="en",
        protect_exact=True,
    )

    assert protected == "Etkinlik ZXQ0QXZ tarihinde"
    assert [item.value for item in replacements] == [english]


def test_complex_call_claim_is_atomized_before_controlled_translation() -> None:
    assert ControlledTranslationService.atomize_claim(
        "You should call the Residence Duty Phone at +90 312 555 0149 before 19:30 "
        "to arrange for key collection from the night porter"
    ) == [
        "Call the Residence Duty Phone at +90 312 555 0149 before 19:30",
        "Arrange key collection with the night porter",
    ]

    assert ControlledTranslationService.atomize_claim(
        "Your laptop should have 16 GB RAM, 60 GB free storage after updates, Wi-Fi 5 or "
        "newer, a working browser, and USB-C or USB-A ports"
    ) == [
        "Your laptop should have 16 GB RAM",
        "Your laptop should have 60 GB free storage after updates",
        "Your laptop should have Wi-Fi 5 or newer, a working browser, and USB-C or USB-A ports",
    ]

    assert ControlledTranslationService.atomize_claim(
        "First, use the campus emergency number 5555 from a campus phone or "
        "+90 312 555 0155 from a mobile phone"
    ) == [
        "First, call extension 5555 from a campus phone or call +90 312 555 0155 "
        "from a mobile phone",
    ]
    assert ControlledTranslationService.normalize_english_claim(
        "Call 5555 from a campus phone or +90 312 555 0155 from a mobile first"
    ) == (
        "First, call extension 5555 from a campus phone or call +90 312 555 0155 "
        "from a mobile phone"
    )
    assert ControlledTranslationService.normalize_english_claim(
        "First, use the 5555 campus emergency number or the +90 312 555 0155 mobile number"
    ) == (
        "First, call extension 5555 from a campus phone or call +90 312 555 0155 "
        "from a mobile phone"
    )

    assert ControlledTranslationService.atomize_claim(
        "Your laptop should have 16 GB RAM and 60 GB free storage after updates"
    ) == [
        "Your laptop should have 16 GB RAM",
        "Your laptop should have 60 GB free storage after system updates",
    ]
    assert ControlledTranslationService.atomize_claim(
        "Evacuation drill is on Tuesday, 4 August at 11:30 and Orion Building occupants "
        "assemble at North Lawn beside the blue beacon"
    ) == [
        "The scheduled evacuation drill takes place at 11:30 on Tuesday, 4 August",
        "Orion Building occupants should assemble at North Lawn beside the blue beacon",
    ]
    assert (
        ControlledTranslationService.normalize_english_claim(
            "Ensure that you can install required software and local development tools"
        )
        == "You need permission to install required software and local development tools"
    )

    assert ControlledTranslationService.atomize_claim(
        "The loan-device return deadline is 30 minutes after the team handoff due at "
        "17:00 on the same day"
    ) == [
        "The team handoff happens at 17:00 on the same day",
        "The return deadline is 30 minutes after the team handoff",
    ]
    assert ControlledTranslationService.atomize_claim(
        "Return the loan laptop, charger, and sleeve by 17:30 on Friday, 14 August; this "
        "deadline is 30 minutes after the project team handoff due at 17:00 on the same day"
    ) == [
        "Give back the loan laptop, charger, and sleeve",
        "The return deadline is 17:30 on Friday, 14 August",
        "This deadline is 30 minutes after the project team handoff due at 17:00 on the same day",
    ]

    assert (
        ControlledTranslationService.normalize_english_claim(
            "Staff run a scheduled evacuation drill at 11:30 on Tuesday, 4 August"
        )
        == "The scheduled evacuation drill takes place at 11:30 on Tuesday, 4 August"
    )

    assert ControlledTranslationService.atomize_claim(
        "If you miss the induction, you must arrange a make-up induction and continue "
        "software-only work until the Safety Officer approves you"
    ) == [
        "If you miss the induction, arrange a make-up induction",
        "If you miss the induction, continue software-only work until the Safety Officer "
        "approves you",
    ]


class ScriptedEngine(BaseTranslationEngine):
    def __init__(self, outputs: list[list[str]]) -> None:
        self.outputs = outputs

    def translate_batch(self, texts, *, source_locale, target_locale):
        output = self.outputs.pop(0)
        assert len(output) == len(texts)
        return output

    def status(self, *, probe=False):
        return {"runtime": "scripted", "ready": True}


@pytest.mark.parametrize(
    ("source", "translated", "reversed_claim", "expected"),
    [
        (
            "Then email conduct@northstar.example",
            "Ardından ZXQ0QXZ'e e-posta gönderin",
            "Then email conduct@northstar.example",
            "Ardından conduct@northstar.example adresine e-posta gönderin",
        ),
        (
            "You should email conduct@northstar.example",
            "ZXQ0QXZ'e e-posta göndermelisiniz",
            "You should email conduct@northstar.example",
            "conduct@northstar.example adresine e-posta göndermelisiniz",
        ),
    ],
)
def test_natural_email_verbs_do_not_trigger_false_semantic_rejection(
    source: str,
    translated: str,
    reversed_claim: str,
    expected: str,
) -> None:
    service = ControlledTranslationService(ScriptedEngine([[translated], [reversed_claim]]))

    assert service.claims_to_turkish([source]) == [expected]


def test_code_of_conduct_word_does_not_look_like_software_installation() -> None:
    conduct = "Davranış Kurallarındaki gizli yolları kullanarak davranışı bildirin"
    install = "Gerekli yazılımı kurun"

    assert ControlledTranslationService.semantic_signature(conduct, locale="tr")["predicates"] == {
        "send"
    }
    assert (
        "install"
        in ControlledTranslationService.semantic_signature(
            install,
            locale="tr",
        )["predicates"]
    )


def test_conduct_email_post_edit_preserves_target_and_predicates() -> None:
    source = (
        "After immediate safety is addressed, report the behavior using the confidential "
        "routes in the Code of Conduct, such as emailing conduct@northstar.example"
    )
    translated = (
        "ZXQ0QXZ, ZXQ1QXZ e-postası gibi Davranış "
        "Kurallarındaki gizli yolları kullanarak davranışı bildirin"
    )
    service = ControlledTranslationService(ScriptedEngine([[translated], [source]]))

    assert service.claims_to_turkish([source]) == [
        "Acil güvenlik sağlandıktan sonra davranışı, Davranış Kurallarındaki gizli bildirim "
        "kanallarından biri olan conduct@northstar.example adresine e-posta göndererek bildirin"
    ]


def test_email_locative_is_rewritten_as_a_natural_target() -> None:
    source = "Email the Conduct Lead at conduct@northstar.example"
    translated = "ZXQ1QXZ adresinde ZXQ0QXZna e-posta gönderin"
    service = ControlledTranslationService(ScriptedEngine([[translated], [source]]))

    assert service.claims_to_turkish([source]) == [
        "Davranış Sorumlusuyla iletişim için conduct@northstar.example adresine e-posta gönderin"
    ]


def test_conduct_route_post_edit_uses_natural_turkish() -> None:
    service = ControlledTranslationService(ScriptedEngine([]))
    rough = (
        "Acil durumla başa çıktıktan sonra, conduct@northstar.example adresine e-posta "
        "göndermeyi içeren Davranış Kuralları'ndaki gizli yolları kullanarak davranışı bildirin"
    )

    assert service._post_edit_turkish(rough) == (
        "Acil güvenliği sağladıktan sonra davranışı, Davranış Kuralları'ndaki gizli bildirim "
        "kanallarından biri olan conduct@northstar.example adresine e-posta göndererek bildirin"
    )


def test_conduct_lead_email_post_edit_uses_natural_turkish() -> None:
    service = ControlledTranslationService(ScriptedEngine([]))
    rough = (
        "Acil güvenlik sağlandıktan sonra, Davranış Sorumlusu için "
        "conduct@northstar.example adresine davranış endişesini e-postayla gönderin"
    )

    assert service._post_edit_turkish(rough) == (
        "Acil güvenlik sağlandıktan sonra davranışla ilgili endişeyi "
        "conduct@northstar.example adresine e-posta göndererek Davranış Sorumlusuna bildirin"
    )


@pytest.mark.parametrize(
    ("source", "translated", "error"),
    [
        (
            "Return the loan laptop and store the tagged luggage",
            "ZXQ0QXZ bırakın ve ZXQ1QXZ iade edin",
            "action-object or action-modality bindings",
        ),
        (
            "First, call +90 312 555 0155, then send an email to conduct@northstar.example",
            "İlk olarak ZXQ1QXZ'e e-posta gönderin, ardından ZXQ0QXZ'yi arayın",
            "explicit order",
        ),
        (
            "You may continue software-only work but cannot enter the Fabrication Studio",
            "ZXQ0QXZ devam etmeyin; ZXQ1QXZ'e girebilirsiniz",
            "action-object or action-modality bindings",
        ),
    ],
)
def test_controlled_translation_rejects_bound_action_and_modality_swaps(
    source: str,
    translated: str,
    error: str,
) -> None:
    service = ControlledTranslationService(ScriptedEngine([[translated]]))

    with pytest.raises(TranslationRejected, match=error):
        service.claims_to_turkish([source])


@pytest.mark.parametrize(
    ("source", "translated", "reversed_claim", "expected"),
    [
        (
            "Return the loan laptop and store the tagged luggage",
            "ZXQ0QXZ'yi iade edin ve ZXQ1QXZ'yi bırakın",
            "Return the loan laptop and store the tagged luggage",
            "Ödünç dizüstü bilgisayarı iade edin ve etiketli bagajı bırakın",
        ),
        (
            "First, call +90 312 555 0155, then send an email to conduct@northstar.example",
            "Önce ZXQ0QXZ'yi arayın, ardından ZXQ1QXZ'e e-posta gönderin",
            "First call +90 312 555 0155, then send an email to conduct@northstar.example",
            "Önce +90 312 555 0155 numaralı telefonu arayın, ardından "
            "conduct@northstar.example adresine e-posta gönderin",
        ),
    ],
)
def test_controlled_translation_accepts_preserved_action_bindings_and_order(
    source: str,
    translated: str,
    reversed_claim: str,
    expected: str,
) -> None:
    service = ControlledTranslationService(ScriptedEngine([[translated], [reversed_claim]]))

    assert service.claims_to_turkish([source]) == [expected]


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


def test_round_trip_accepts_natural_beside_and_assemble_paraphrases() -> None:
    service = ControlledTranslationService(
        ScriptedEngine(
            [
                ["ZXQ0QXZ'deki ZXQ1QXZ, ZXQ4QXZ sırasında ZXQ3QXZ yanında ZXQ2QXZ toplanmalıdır"],
                [
                    "People in the Orion Building must gather in North Lawn next to the blue beacon during the drill"
                ],
            ]
        )
    )

    translated = service.claims_to_turkish(
        [
            "Orion Building occupants should assemble at North Lawn beside the blue beacon during the drill"
        ]
    )

    assert translated == [
        "Orion Building'deki kişiler, tatbikat sırasında mavi işaret lambası yanında North Lawn toplanmalıdır"
    ]

    meaning_loss = ControlledTranslationService(
        ScriptedEngine(
            [
                ["17:30'dan önce ZXQ0QXZ'ne gidin"],
                ["Go to the Program Office before 17:30"],
            ]
        )
    )
    with pytest.raises(TranslationRejected, match="unprotected recall"):
        meaning_loss.claims_to_turkish(
            ["Bring the insurance certificate to the Program Office before 17:30"]
        )


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
            "protected value",
        ),
        (
            "The event is Monday, 3 August 2026",
            "Etkinlik 3 Ağustos 2026 Cumartesi günü",
            "exact numeric/contact fact",
        ),
        (
            "The event is Wednesday, 3 September 2026",
            "Etkinlik 3 Ağustos 2026 Çarşamba günü",
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
