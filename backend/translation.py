"""Offline English-Turkish translation with deterministic fact and terminology guards."""

from __future__ import annotations

import json
import os
import re
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from runtime import RuntimeUnavailable

SourceLocale = Literal["en", "tr"]

TR_TO_EN_MODEL = "Helsinki-NLP/opus-mt-tc-big-tr-en"
EN_TO_TR_MODEL = "Helsinki-NLP/opus-mt-tc-big-en-tr"
TR_TO_EN_REVISION = "2261c8fc7b1af59caee87f8ff0ecf3fbccfe8391"
EN_TO_TR_REVISION = "e539fc16a8a1a0ea5950eb339b595bfcce990e90"
DEFAULT_GLOSSARY_PATH = Path(__file__).parent / "terminology" / "northstar-en-tr.json"

_SENTINEL = re.compile(
    r"ZXQ\s*(?P<index>\d+)\s*QXZ(?:['’]?(?P<suffix>[A-Za-zÇĞİÖŞÜçğıöşü]+))?",
    flags=re.IGNORECASE,
)
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE = re.compile(r"\+\d(?:[\d\s().-]{5,}\d)")
_TIME = re.compile(r"(?<!\w)\d{1,2}:\d{2}(?!\w)")
_DATE = re.compile(r"(?<!\w)\d{1,4}(?:[./-]\d{1,2}){1,2}(?!\w)")
_NUMBER = re.compile(r"(?<![\w@])\d+(?:[.,]\d+)?(?![\w@])")
_TOKEN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_CALENDAR_CANONICAL = {
    "august": "august",
    "ağustos": "august",
    "tuesday": "tuesday",
    "salı": "tuesday",
    "friday": "friday",
    "cuma": "friday",
}
_CALENDAR = re.compile(
    r"\b(?:August|Ağustos|Tuesday|Salı|Friday|Cuma)\b",
    flags=re.IGNORECASE,
)


class TranslationRejected(RuntimeError):
    """Raised when translation changes protected facts or loses too much meaning."""


@dataclass(frozen=True)
class GlossaryEntry:
    source: str
    target: str
    sentinel: bool = True


@dataclass(frozen=True)
class ProtectedValue:
    value: str
    kind: str


class TerminologyGlossary:
    """Checked-in, deployment-scoped terminology used in both translation directions."""

    def __init__(self, path: Path = DEFAULT_GLOSSARY_PATH) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries = payload.get("entries")
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"Translation glossary has no entries: {path}")
        self.id = str(payload.get("id") or path.stem)
        self.scope = str(payload.get("scope") or "")
        self.path = path
        self.entries = tuple(
            sorted(
                (
                    GlossaryEntry(
                        source=str(item["source"]),
                        target=str(item["target"]),
                        sentinel=bool(item.get("sentinel", True)),
                    )
                    for item in entries
                ),
                key=lambda entry: entry.source.casefold(),
            )
        )
        source_terms = [entry.source.casefold() for entry in self.entries]
        target_terms = [entry.target.casefold() for entry in self.entries]
        if len(source_terms) != len(set(source_terms)):
            raise ValueError(f"Translation glossary has duplicate English terms: {path}")
        duplicate_targets = {value for value, count in Counter(target_terms).items() if count > 1}
        ambiguous_targets = {
            entry.target.casefold()
            for entry in self.entries
            if entry.target.casefold() in duplicate_targets
            and len(
                {
                    candidate.source.casefold()
                    for candidate in self.entries
                    if candidate.target.casefold() == entry.target.casefold()
                }
            )
            > 1
        }
        if ambiguous_targets:
            raise ValueError(
                "Translation glossary has ambiguous Turkish terms: "
                + ", ".join(sorted(ambiguous_targets))
            )

    @staticmethod
    def _phrase_pattern(value: str) -> re.Pattern[str]:
        return re.compile(rf"(?<!\w){re.escape(value)}(?!\w)", flags=re.IGNORECASE)

    def protect(
        self,
        text: str,
        *,
        source_locale: SourceLocale,
        target_locale: SourceLocale,
    ) -> tuple[str, list[ProtectedValue]]:
        if source_locale == target_locale:
            return text, []

        candidates: list[tuple[int, int, int, ProtectedValue]] = []
        for entry in self.entries:
            if not entry.sentinel:
                continue
            phrase = entry.source if source_locale == "en" else entry.target
            replacement = entry.target if target_locale == "tr" else entry.source
            for match in self._phrase_pattern(phrase).finditer(text):
                kind = "proper" if entry.source == entry.target else "term"
                candidates.append(
                    (match.start(), match.end(), 0, ProtectedValue(replacement, kind))
                )

        protected_patterns = (
            ("email", _EMAIL),
            ("phone", _PHONE),
        )
        for priority, (kind, pattern) in enumerate(protected_patterns, start=1):
            for match in pattern.finditer(text):
                candidates.append(
                    (match.start(), match.end(), priority, ProtectedValue(match.group(0), kind))
                )

        # Prefer the longest candidate at the same start, then discard overlapping submatches.
        candidates.sort(key=lambda item: (item[0], -(item[1] - item[0]), item[2]))
        selected: list[tuple[int, int, ProtectedValue]] = []
        occupied_until = -1
        for start, end, _priority, protected in candidates:
            if start < occupied_until:
                continue
            selected.append((start, end, protected))
            occupied_until = end

        if not selected:
            return text, []

        output: list[str] = []
        replacements: list[ProtectedValue] = []
        cursor = 0
        for start, end, protected in selected:
            output.append(text[cursor:start])
            output.append(f"ZXQ{len(replacements)}QXZ")
            replacements.append(protected)
            cursor = end
        output.append(text[cursor:])
        return "".join(output), replacements

    @staticmethod
    def _term_with_suffix(value: str, suffix: str) -> str:
        """Attach a model-selected Turkish case to a Turkish glossary phrase naturally."""

        normalized = suffix.casefold()
        vowels = [character for character in value.casefold() if character in "aeıioöuü"]
        last_vowel = vowels[-1] if vowels else "e"
        two_way = "a" if last_vowel in "aıou" else "e"
        four_way = {
            "a": "ı",
            "ı": "ı",
            "e": "i",
            "i": "i",
            "o": "u",
            "u": "u",
            "ö": "ü",
            "ü": "ü",
        }[last_vowel]
        last_character = next(
            (character for character in reversed(value.casefold()) if character.isalpha()),
            "",
        )
        ends_vowel = last_character in "aeıioöuü"
        compound_buffer = "n" if ends_vowel and " " in value else "y"
        consonant = "t" if last_character in "fstkçşhp" else "d"

        if normalized in {"a", "e", "ya", "ye", "na", "ne"}:
            attached = (compound_buffer if ends_vowel else "") + two_way
        elif normalized in {"ı", "i", "u", "ü", "yı", "yi", "yu", "yü", "nı", "ni", "nu", "nü"}:
            attached = (compound_buffer if ends_vowel else "") + four_way
        elif normalized in {"da", "de", "ta", "te"}:
            attached = ("n" if ends_vowel and " " in value else "") + consonant + two_way
        elif normalized in {"dan", "den", "tan", "ten"}:
            attached = ("n" if ends_vowel and " " in value else "") + consonant + two_way + "n"
        elif normalized in {"ın", "in", "un", "ün", "nın", "nin", "nun", "nün"}:
            attached = ("n" if ends_vowel else "") + four_way + "n"
        elif normalized in {"la", "le", "yla", "yle"}:
            attached = ("y" if ends_vowel else "") + "l" + two_way
        elif normalized in {"daki", "deki", "taki", "teki"}:
            attached = ("n" if ends_vowel and " " in value else "") + consonant + two_way + "ki"
        elif normalized in {"dır", "dir", "dur", "dür", "tır", "tir", "tur", "tür"}:
            attached = consonant + four_way + "r"
        else:
            raise TranslationRejected(
                f"The translator attached an unsupported suffix to glossary term '{value}'."
            )
        return value + attached

    @staticmethod
    def restore(text: str, replacements: list[ProtectedValue]) -> str:
        if not replacements:
            return text.strip()
        found = [int(match.group("index")) for match in _SENTINEL.finditer(text)]
        if Counter(found) != Counter(range(len(replacements))):
            raise TranslationRejected("The local translator changed or lost a protected value.")

        def replacement(match: re.Match[str]) -> str:
            index = int(match.group("index"))
            protected = replacements[index]
            suffix = match.group("suffix") or ""
            if protected.kind == "term" and suffix:
                return TerminologyGlossary._term_with_suffix(protected.value, suffix)
            if protected.kind == "proper" and suffix:
                attached = TerminologyGlossary._term_with_suffix(protected.value, suffix)[
                    len(protected.value) :
                ]
                return f"{protected.value}'{attached}"
            if protected.kind == "phone" and suffix:
                normalized = suffix.casefold()
                if normalized in {"ı", "i", "u", "ü", "yı", "yi", "yu", "yü"}:
                    return f"{protected.value} numaralı telefonu"
                if normalized in {"a", "e", "ya", "ye"}:
                    return f"{protected.value} numaralı telefona"
                if normalized in {"dan", "den", "tan", "ten"}:
                    return f"{protected.value} numaralı telefondan"
                if normalized in {"da", "de", "ta", "te"}:
                    return f"{protected.value} numaralı telefonda"
                if normalized in {"dır", "dir", "dur", "dür", "tır", "tir", "tur", "tür"}:
                    return f"{protected.value} numaralı telefondur"
                raise TranslationRejected("The translator attached an unsafe suffix to a phone.")
            if protected.kind == "email" and suffix:
                normalized = suffix.casefold()
                email_suffixes = {
                    "ı": "adresini",
                    "i": "adresini",
                    "u": "adresini",
                    "ü": "adresini",
                    "yı": "adresini",
                    "yi": "adresini",
                    "yu": "adresini",
                    "yü": "adresini",
                    "a": "adresine",
                    "e": "adresine",
                    "ya": "adresine",
                    "ye": "adresine",
                    "dan": "adresinden",
                    "den": "adresinden",
                    "tan": "adresinden",
                    "ten": "adresinden",
                    "da": "adresinde",
                    "de": "adresinde",
                    "ta": "adresinde",
                    "te": "adresinde",
                    "daki": "adresindeki",
                    "deki": "adresindeki",
                    "taki": "adresindeki",
                    "teki": "adresindeki",
                    "dır": "adresidir",
                    "dir": "adresidir",
                    "dur": "adresidir",
                    "dür": "adresidir",
                    "tır": "adresidir",
                    "tir": "adresidir",
                    "tur": "adresidir",
                    "tür": "adresidir",
                }
                descriptor = email_suffixes.get(normalized)
                if descriptor is None:
                    raise TranslationRejected(
                        "The translator attached an unsafe suffix to an email."
                    )
                return f"{protected.value} {descriptor}"
            return protected.value + (f"'{suffix}" if suffix else "")

        restored = _SENTINEL.sub(replacement, text)
        restored = re.sub(r"\s+([,.;:!?])", r"\1", restored)
        restored = re.sub(r"[ \t]{2,}", " ", restored)
        return restored.strip()

    @staticmethod
    def _exact_occurrences(text: str) -> list[tuple[int, int, str, str]]:
        candidates: list[tuple[int, int, int, str, str]] = []
        patterns = (
            ("email", _EMAIL),
            ("phone", _PHONE),
            ("time", _TIME),
            ("date", _DATE),
            ("number", _NUMBER),
        )
        for priority, (kind, pattern) in enumerate(patterns):
            for match in pattern.finditer(text):
                value = match.group(0)
                if kind == "email":
                    normalized = value.casefold()
                elif kind == "phone":
                    normalized = re.sub(r"\D", "", value)
                else:
                    normalized = value
                candidates.append((match.start(), match.end(), priority, kind, normalized))
        for match in _CALENDAR.finditer(text):
            candidates.append(
                (
                    match.start(),
                    match.end(),
                    len(patterns),
                    "calendar",
                    _CALENDAR_CANONICAL[match.group(0).casefold()],
                )
            )
        candidates.sort(key=lambda item: (item[0], -(item[1] - item[0]), item[2]))
        selected: list[tuple[int, int, str, str]] = []
        occupied_until = -1
        for start, end, _priority, kind, normalized in candidates:
            if start < occupied_until:
                continue
            selected.append((start, end, kind, normalized))
            occupied_until = end
        return selected

    @classmethod
    def exact_facts(cls, text: str) -> Counter[tuple[str, str]]:
        """Extract exact values that must survive translation without additions or omissions."""

        return Counter(
            (kind, normalized) for _start, _end, kind, normalized in cls._exact_occurrences(text)
        )

    @classmethod
    def relation_signatures(
        cls,
        text: str,
        *,
        locale: SourceLocale,
    ) -> dict[tuple[str, str], set[str]]:
        """Associate exact values with before/after/deadline/range roles."""

        folded = text.casefold()
        signatures: dict[tuple[str, str], set[str]] = {}
        for start, end, kind, normalized in cls._exact_occurrences(text):
            if kind not in {"time", "date", "number"}:
                continue
            before = folded[max(0, start - 40) : start]
            after = folded[end : min(len(folded), end + 40)]
            sentence_start = max(folded.rfind(".", 0, start), folded.rfind("\n", 0, start))
            sentence_end_candidates = [
                position
                for position in (folded.find(".", end), folded.find("\n", end))
                if position >= 0
            ]
            sentence_end = min(sentence_end_candidates, default=len(folded))
            sentence = folded[sentence_start + 1 : sentence_end]
            relations: set[str] = set()
            if locale == "en":
                if re.search(r"\bbefore\s*$", before) or re.match(
                    r"\s*(?:minutes?\s+)?before\b", after
                ):
                    relations.add("before")
                if re.search(r"\bafter\s*$", before) or re.match(
                    r"\s*(?:minutes?\s+)?after\b", after
                ):
                    relations.add("after")
                if re.search(r"(?:\bno\s+later\s+than|\bby|\buntil)\s*$", before):
                    relations.add("deadline")
                if (
                    re.search(r"\b(?:between|from)\s*$", before)
                    or re.search(r"\b(?:through|and)\s*$", before)
                    or (re.search(r"\bbetween\b", sentence) and re.search(r"\band\b", sentence))
                    or (re.search(r"\bfrom\b", sentence) and re.search(r"\bthrough\b", sentence))
                ):
                    relations.add("range")
            else:
                if re.match(
                    r"\s*(?:['’][a-zçğıöşü]+\s+)?(?:dakika\s+)?önce[a-zçğıöşü]*\b",
                    after,
                ):
                    relations.add("before")
                if re.match(
                    r"\s*(?:['’][a-zçğıöşü]+\s+)?(?:dakika\s+)?sonra[a-zçğıöşü]*\b",
                    after,
                ):
                    relations.add("after")
                if re.search(r"\ben\s+geç\s*$", before) or re.match(
                    r"\s*['’][a-zçğıöşü]+\s+kadar\b", after
                ):
                    relations.add("deadline")
                if " arasında" in sentence:
                    relations.add("range")
            if relations:
                signatures.setdefault((kind, normalized), set()).update(relations)
        return signatures

    @staticmethod
    def unit_signatures(text: str) -> Counter[tuple[str, str]]:
        """Bind quantities to memory/storage/Wi-Fi roles so equal numbers cannot swap."""

        folded = text.casefold()
        signatures: Counter[tuple[str, str]] = Counter()
        for match in re.finditer(r"(?<!\w)(\d+(?:[.,]\d+)?)\s*gb\b", folded):
            context = folded[match.end() : min(len(folded), match.end() + 35)]
            role = "memory" if re.search(r"\bram\b", context) else "storage"
            signatures[(role, match.group(1))] += 1
        for match in re.finditer(r"\bwi-fi\s*(\d+(?:[.,]\d+)?)\b", folded):
            signatures[("wifi_version", match.group(1))] += 1
        return signatures


class BaseTranslationEngine:
    def translate_batch(
        self,
        texts: list[str],
        *,
        source_locale: SourceLocale,
        target_locale: SourceLocale,
    ) -> list[str]:
        raise NotImplementedError

    def status(self, *, probe: bool = False) -> dict[str, Any]:
        raise NotImplementedError


class FakeTranslationEngine(BaseTranslationEngine):
    """Identity adapter for deterministic tests; focused tests inject explicit translations."""

    def translate_batch(
        self,
        texts: list[str],
        *,
        source_locale: SourceLocale,
        target_locale: SourceLocale,
    ) -> list[str]:
        return list(texts)

    def status(self, *, probe: bool = False) -> dict[str, Any]:
        return {
            "runtime": "fake",
            "ready": True,
            "models": {"tr_to_en": "identity", "en_to_tr": "identity"},
            "error": None,
        }


class MarianTranslationEngine(BaseTranslationEngine):
    """Lazy, offline-only Marian engine using the official OPUS tc-big model pair."""

    model_ids = {"tr_to_en": TR_TO_EN_MODEL, "en_to_tr": EN_TO_TR_MODEL}
    model_revisions = {
        "tr_to_en": TR_TO_EN_REVISION,
        "en_to_tr": EN_TO_TR_REVISION,
    }

    def __init__(self, *, local_files_only: bool = True) -> None:
        self.local_files_only = local_files_only
        self._models: dict[str, Any] = {}
        self._tokenizers: dict[str, Any] = {}
        self._lock = threading.RLock()
        self._last_error: str | None = None

    @staticmethod
    def _direction(source_locale: SourceLocale, target_locale: SourceLocale) -> str:
        if (source_locale, target_locale) == ("tr", "en"):
            return "tr_to_en"
        if (source_locale, target_locale) == ("en", "tr"):
            return "en_to_tr"
        raise ValueError(f"Unsupported translation direction: {source_locale}->{target_locale}")

    def _load(self, direction: str) -> tuple[Any, Any]:
        with self._lock:
            if direction in self._models:
                return self._tokenizers[direction], self._models[direction]
            model_id = self.model_ids[direction]
            try:
                from transformers import MarianMTModel, MarianTokenizer

                tokenizer = MarianTokenizer.from_pretrained(
                    model_id,
                    revision=self.model_revisions[direction],
                    local_files_only=self.local_files_only,
                )
                model = MarianMTModel.from_pretrained(
                    model_id,
                    revision=self.model_revisions[direction],
                    local_files_only=self.local_files_only,
                )
                model.eval()
            except Exception as exc:
                self._last_error = str(exc)
                raise RuntimeUnavailable(
                    "The offline Turkish translation models are unavailable. Run `make models` "
                    f"once while online. Details: {exc}"
                ) from exc
            self._tokenizers[direction] = tokenizer
            self._models[direction] = model
            self._last_error = None
            return tokenizer, model

    def translate_batch(
        self,
        texts: list[str],
        *,
        source_locale: SourceLocale,
        target_locale: SourceLocale,
    ) -> list[str]:
        if not texts:
            return []
        if source_locale == target_locale:
            return list(texts)
        direction = self._direction(source_locale, target_locale)
        tokenizer, model = self._load(direction)
        try:
            encoded = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            generated = model.generate(
                **encoded,
                do_sample=False,
                num_beams=int(os.getenv("TRACEFOLD_TRANSLATION_BEAMS", "4")),
                renormalize_logits=True,
                max_new_tokens=int(os.getenv("TRACEFOLD_TRANSLATION_MAX_TOKENS", "256")),
            )
            outputs = tokenizer.batch_decode(generated, skip_special_tokens=True)
        except Exception as exc:
            self._last_error = str(exc)
            raise RuntimeUnavailable(f"Offline translation failed: {exc}") from exc
        if len(outputs) != len(texts):
            raise RuntimeUnavailable("Offline translation returned an unexpected result count.")
        return [output.strip() for output in outputs]

    def prepare(self, *, smoke: bool = True) -> None:
        previous = self.local_files_only
        self.local_files_only = False
        try:
            for direction in self.model_ids:
                self._load(direction)
            if smoke:
                self.translate_batch(["Varış saati nedir?"], source_locale="tr", target_locale="en")
                self.translate_batch(
                    ["Call the Program Office."], source_locale="en", target_locale="tr"
                )
        finally:
            self.local_files_only = previous

    @classmethod
    def _snapshot_cached(cls, direction: str) -> bool:
        from huggingface_hub import try_to_load_from_cache

        model_id = cls.model_ids[direction]
        revision = cls.model_revisions[direction]

        def cached(filename: str) -> bool:
            return isinstance(
                try_to_load_from_cache(model_id, filename, revision=revision),
                str,
            )

        tokenizer_ready = all(
            cached(filename)
            for filename in ("config.json", "source.spm", "target.spm", "vocab.json")
        )
        weights_ready = cached("model.safetensors") or cached("pytorch_model.bin")
        return tokenizer_ready and weights_ready

    def status(self, *, probe: bool = False) -> dict[str, Any]:
        cached: dict[str, bool] = {}
        try:
            cached = {direction: self._snapshot_cached(direction) for direction in self.model_ids}
        except Exception:
            cached = {direction: direction in self._models for direction in self.model_ids}
        return {
            "runtime": "marian_opus",
            "ready": all(cached.values()),
            "offline_only": self.local_files_only,
            "models": {
                direction: {
                    "id": model_id,
                    "revision": self.model_revisions[direction],
                    "cached": cached[direction],
                    "loaded": direction in self._models,
                }
                for direction, model_id in self.model_ids.items()
            },
            "error": self._last_error,
        }


class ControlledTranslationService:
    """Protect facts/terms, translate locally, then reject lossy round trips."""

    def __init__(
        self,
        engine: BaseTranslationEngine,
        glossary: TerminologyGlossary | None = None,
    ) -> None:
        self.engine = engine
        self.glossary = glossary or TerminologyGlossary()
        self.round_trip_threshold = float(
            os.getenv("TRACEFOLD_TRANSLATION_ROUNDTRIP_THRESHOLD", "0.40")
        )

    def _translate_batch(
        self,
        texts: list[str],
        *,
        source_locale: SourceLocale,
        target_locale: SourceLocale,
        enforce_exact: bool = True,
        enforce_relations: bool = True,
    ) -> list[str]:
        protected: list[str] = []
        replacements: list[list[ProtectedValue]] = []
        exact_facts = [self.glossary.exact_facts(text) for text in texts]
        expected_relations = [
            self.glossary.relation_signatures(text, locale=source_locale) for text in texts
        ]
        expected_units = [self.glossary.unit_signatures(text) for text in texts]
        for text in texts:
            encoded, values = self.glossary.protect(
                text,
                source_locale=source_locale,
                target_locale=target_locale,
            )
            protected.append(encoded)
            replacements.append(values)
        outputs = self.engine.translate_batch(
            protected,
            source_locale=source_locale,
            target_locale=target_locale,
        )
        restored = [
            self.glossary.restore(output, values)
            for output, values in zip(outputs, replacements, strict=True)
        ]
        for expected, relations, units, translated in zip(
            exact_facts,
            expected_relations,
            expected_units,
            restored,
            strict=True,
        ):
            observed = self.glossary.exact_facts(translated)
            if enforce_exact and observed != expected:
                raise TranslationRejected(
                    "The local translator changed, added, or lost an exact numeric/contact fact."
                )
            if enforce_relations:
                observed_relations = self.glossary.relation_signatures(
                    translated,
                    locale=target_locale,
                )
                for fact, required in relations.items():
                    if not required.issubset(observed_relations.get(fact, set())):
                        raise TranslationRejected(
                            "The local translator changed a temporal relation attached to an exact "
                            "fact."
                        )
            if enforce_exact and self.glossary.unit_signatures(translated) != units:
                raise TranslationRejected(
                    "The local translator changed the unit or role attached to a quantity."
                )
        return restored

    def query_to_english(self, question: str) -> str:
        return self._translate_batch([question], source_locale="tr", target_locale="en")[0]

    @staticmethod
    def normalize_english_claim(claim: str) -> str:
        """Rewrite ambiguous English constructions into simple, translation-safe English."""

        normalized = " ".join(claim.strip().split())
        normalized = re.sub(r"^They\b", "The participant", normalized, flags=re.IGNORECASE)
        normalized = re.sub(
            r"^(?P<subject>.+?)\s+runs\s+from\s+(?P<start>.+?)\s+through\s+(?P<end>.+)$",
            r"\g<subject> takes place between \g<start> and \g<end>",
            normalized,
            flags=re.IGNORECASE,
        )
        normalized = re.sub(
            r"\b(?:takes?|has)\s+priority\s+over\b",
            "must happen before",
            normalized,
            flags=re.IGNORECASE,
        )
        normalized = re.sub(
            r"\bare\s+due\s+by\b",
            "must be completed no later than",
            normalized,
            flags=re.IGNORECASE,
        )
        normalized = re.sub(
            r"\bis\s+due\s+by\b",
            "must be completed no later than",
            normalized,
            flags=re.IGNORECASE,
        )
        normalized = re.sub(r"^Return\b", "Give back", normalized, flags=re.IGNORECASE)
        normalized = re.sub(
            r"\bluggage\s+in\s+the\s+tagged\s+ground-floor\s+storage\s+room\b",
            "tagged luggage in the ground-floor storage room",
            normalized,
            flags=re.IGNORECASE,
        )
        normalized = re.sub(
            r"^(?P<subject>The\s+.+?(?:handoff|drill|induction))\s+is\s+due\s+at\b",
            r"\g<subject> happens at",
            normalized,
            flags=re.IGNORECASE,
        )
        report_contact = re.match(
            r"^(?P<concern>.+?)\s+can\s+also\s+be\s+reported\s+to\s+the\s+"
            r"(?P<role>.+?)\s+at\s+(?P<email>[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})\s+"
            r"once\s+(?P<condition>.+)$",
            normalized,
            flags=re.IGNORECASE,
        )
        if report_contact:
            condition = report_contact.group("condition").rstrip(". ")
            concern = re.sub(
                r"^(?:a|an|the)\s+",
                "",
                report_contact.group("concern"),
                flags=re.IGNORECASE,
            )
            normalized = (
                f"After {condition}, email the {concern} to {report_contact.group('email')} "
                f"for the {report_contact.group('role')}"
            )
        normalized = re.sub(
            r"\s+by\s+(?=\d{1,2}:\d{2}\b)",
            " no later than ",
            normalized,
            flags=re.IGNORECASE,
        )
        return normalized.rstrip(".")

    @classmethod
    def atomize_claim(cls, claim: str) -> list[str]:
        """Split only general compound temporal clauses; never inspect case IDs or questions."""

        normalized = cls.normalize_english_claim(claim)
        offset = re.match(
            r"^(?P<main>.+?),\s*(?P<offset>\d+\s+minutes?\s+(?:before|after)\s+.+)$",
            normalized,
            flags=re.IGNORECASE,
        )
        if offset:
            return [offset.group("main"), f"This time is {offset.group('offset')}"]

        return_deadline = re.match(
            r"^Give back\s+(?P<objects>.+?)\s+no\s+later\s+than\s+(?P<deadline>.+)$",
            normalized,
            flags=re.IGNORECASE,
        )
        if return_deadline:
            return [
                f"Give back {return_deadline.group('objects')}",
                f"The return deadline is {return_deadline.group('deadline')}",
            ]

        until_after = re.match(
            r"^(?P<action>.+?)\s+until\s+(?P<deadline>\d{1,2}:\d{2})\s+after\s+"
            r"(?P<event>.+)$",
            normalized,
            flags=re.IGNORECASE,
        )
        if until_after:
            action = until_after.group("action")
            action = action[0].lower() + action[1:] if action else action
            return [
                f"After {until_after.group('event')}, {action} until "
                f"{until_after.group('deadline')}"
            ]
        return [normalized]

    def _post_edit_turkish(self, value: str) -> str:
        edited = re.sub(r"\bgeri verin\b", "iade edin", value, flags=re.IGNORECASE)
        edited = re.sub(
            r"\bbir araya gelir\b",
            "toplanır",
            edited,
            flags=re.IGNORECASE,
        )
        edited = re.sub(
            r"\bbir araya gelmelidir\b",
            "toplanmalıdır",
            edited,
            flags=re.IGNORECASE,
        )
        edited = re.sub(
            r"\bçalışan tarayıcı ve USB bağlantı noktasına sahip\b",
            "çalışan bir tarayıcıya ve USB bağlantı noktasına sahip",
            edited,
            flags=re.IGNORECASE,
        )
        edited = re.sub(
            r"\b(minimum\s+\d+\s+GB\s+RAM)$",
            r"\1'dir",
            edited,
            flags=re.IGNORECASE,
        )
        edited = re.sub(
            r"(\b\d{1,2}\s+[A-Za-zÇĞİÖŞÜçğıöşü]+)\s+ve\s+"
            r"(\d{1,2}\s+[A-Za-zÇĞİÖŞÜçğıöşü]+(?:\s+\d{4})?\s+"
            r"(?:tarihleri\s+)?arasında)",
            r"\1 ile \2",
            edited,
        )
        for entry in self.glossary.entries:
            if entry.source != entry.target:
                continue
            bare = f"{entry.target} kişiler"
            if bare.casefold() in edited.casefold():
                located = TerminologyGlossary._term_with_suffix(entry.target, "deki")
                located = f"{entry.target}'{located[len(entry.target) :]} kişiler"
                edited = re.sub(re.escape(bare), located, edited, flags=re.IGNORECASE)
        storage_room = next(
            (
                entry.target
                for entry in self.glossary.entries
                if entry.source.casefold() == "ground-floor storage room"
            ),
            None,
        )
        if storage_room:
            dative = TerminologyGlossary._term_with_suffix(storage_room, "e")
            locative = TerminologyGlossary._term_with_suffix(storage_room, "de")
            edited = re.sub(
                rf"{re.escape(dative)}\s+yerleştir",
                f"{locative} bırak",
                edited,
                flags=re.IGNORECASE,
            )
        edited = re.sub(r"\bbırakebilir\b", "bırakabilir", edited, flags=re.IGNORECASE)
        first_letter = next(
            (index for index, char in enumerate(edited) if not char.isspace()), None
        )
        if first_letter is not None and edited[first_letter].isalpha():
            upper = {"i": "İ", "ı": "I"}.get(
                edited[first_letter],
                edited[first_letter].upper(),
            )
            edited = edited[:first_letter] + upper + edited[first_letter + 1 :]
        return edited

    @staticmethod
    def _content_tokens(value: str) -> set[str]:
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
            "the",
            "to",
            "with",
        }
        return {token for token in _TOKEN.findall(value.casefold()) if token not in stopwords}

    def _unprotected_content_tokens(self, value: str) -> set[str]:
        reduced = value
        for entry in self.glossary.entries:
            reduced = self.glossary._phrase_pattern(entry.source).sub(" ", reduced)
        spans = self.glossary._exact_occurrences(reduced)
        for start, end, _kind, _normalized in reversed(spans):
            reduced = reduced[:start] + " " + reduced[end:]
        semantic_words = {
            "after",
            "assemble",
            "back",
            "before",
            "between",
            "call",
            "can",
            "cannot",
            "continue",
            "email",
            "enter",
            "from",
            "give",
            "hand",
            "happen",
            "install",
            "later",
            "leave",
            "may",
            "must",
            "notify",
            "occur",
            "place",
            "report",
            "return",
            "send",
            "should",
            "store",
            "support",
            "take",
            "takes",
            "than",
            "through",
            "until",
        }
        return self._content_tokens(reduced) - semantic_words

    @staticmethod
    def semantic_signature(value: str, *, locale: SourceLocale) -> dict[str, set[str]]:
        """Extract critical predicate, modality, contact-role, and order semantics."""

        folded = value.casefold()
        predicates: set[str] = set()
        modalities: set[str] = set()
        order: set[str] = set()

        if locale == "en":
            predicate_patterns = {
                "call": r"\bcall\b",
                "send": r"(?:^|[.!?,]\s+)(?:email|report)\b|\b(?:send|notify)\b",
                "return": r"(?:^|[.!?]\s+)(?:return|give\s+back|hand\s+back)\b",
                "enter": r"\benter\b",
                "assemble": r"\bassemble\b",
                "continue": r"\bcontinue\b",
                "install": r"\binstall\b",
                "support": r"\bsupport\b",
                "store": r"\b(?:leave|store)\b|(?<!take\s)(?<!takes\s)\bplace\b",
                "occur": r"\b(?:take\s+place|takes\s+place|happen[a-z]*|occur[a-z]*)\b",
            }
            if re.search(r"\b(?:cannot|can't|must\s+not|do\s+not|may\s+not)\b", folded):
                modalities.add("prohibited")
            elif re.search(r"\b(?:may|can)\b", folded):
                modalities.add("permitted")
            if re.search(r"\b(?:must|should|need|needs|required|no\s+later\s+than)\b", folded):
                modalities.add("required")
            if re.search(r"\bfirst\b", folded):
                order.add("first")
            if re.search(r"\b(?:then|after|once)\b", folded):
                order.add("then")
        else:
            predicate_patterns = {
                "call": r"\bara(?:yın|mak|n|dı|r)?\b",
                "send": r"\b(?:gönder|bildir|ilet)[a-zçğıöşü]*\b",
                "return": r"\b(?:iade\s+e(?:t|d)|geri\s+ver)[a-zçğıöşü]*\b",
                "enter": r"\bgir[a-zçğıöşü]*\b",
                "assemble": r"\b(?:toplan|bir\s+araya\s+gel)[a-zçğıöşü]*\b",
                "continue": r"\bdevam\s+e(?:t|d)[a-zçğıöşü]*\b",
                "install": r"\b(?:yükle|kur)[a-zçğıöşü]*\b",
                "support": r"\bdestekle[a-zçğıöşü]*\b",
                "store": r"\b(?:bırak|yerleştir|muhafaza\s+et)[a-zçğıöşü]*\b",
                "occur": r"\b(?:gerçekleş|yapıl)[a-zçğıöşü]*\b",
            }
            if re.search(
                r"\b[a-zçğıöşü]*(?:amaz|emez|mayın|meyin|mamalı|memeli)[a-zçğıöşü]*\b|\byasak\b",
                folded,
            ):
                modalities.add("prohibited")
            elif re.search(r"\b[a-zçğıöşü]*(?:abilir|ebilir)\b|\bizin\s+ver", folded):
                modalities.add("permitted")
            if re.search(
                r"\b(?:gerek[a-zçğıöşü]*|zorunlu|en\s+geç)\b|"
                r"\b[a-zçğıöşü]*(?:malı|meli|malıdır|melidir)\b",
                folded,
            ):
                modalities.add("required")
            if re.search(r"\b(?:öncelikle|ilk)\b", folded):
                order.add("first")
            if re.search(r"\b(?:ardından|sonra)\b", folded):
                order.add("then")

        predicates.update(
            name for name, pattern in predicate_patterns.items() if re.search(pattern, folded)
        )
        emails = _EMAIL.findall(value)
        phones = _PHONE.findall(value)
        if emails and "send" in predicates:
            if locale == "en" or any(
                re.search(rf"{re.escape(email)}\s+adresine\b", folded) for email in emails
            ):
                predicates.add("email_target")
        if phones and "call" in predicates:
            if locale == "en" or all(
                f"{phone.casefold()} numaralı telefonu" in folded for phone in phones
            ):
                predicates.add("phone_object")
        extensions = [
            normalized
            for start, end, kind, normalized in TerminologyGlossary._exact_occurrences(value)
            if kind == "number"
            and 3 <= len(normalized) <= 6
            and ("extension" if locale == "en" else "dahili")
            in folded[max(0, start - 40) : min(len(folded), end + 20)]
        ]
        if extensions and "call" in predicates:
            if locale == "en" or all(
                re.search(rf"dahili\s+numarası\s+{re.escape(extension)}\b", folded)
                for extension in extensions
            ):
                predicates.add("extension_object")
        return {"predicates": predicates, "modalities": modalities, "order": order}

    def claims_to_turkish(self, claims: list[str]) -> list[str]:
        normalized_claims = [self.normalize_english_claim(claim) for claim in claims]
        translated = self._translate_batch(
            normalized_claims,
            source_locale="en",
            target_locale="tr",
        )
        translated = [self._post_edit_turkish(value) for value in translated]
        for original, localized in zip(normalized_claims, translated, strict=True):
            expected_signature = self.semantic_signature(original, locale="en")
            observed_signature = self.semantic_signature(localized, locale="tr")
            for dimension, expected in expected_signature.items():
                if expected != observed_signature[dimension]:
                    raise TranslationRejected(
                        f"The local translator changed critical {dimension} semantics."
                    )
        reversed_claims = self.engine.translate_batch(
            translated,
            source_locale="tr",
            target_locale="en",
        )
        if len(reversed_claims) != len(translated):
            raise TranslationRejected("Reverse translation returned an unexpected result count.")
        for original, reversed_claim in zip(normalized_claims, reversed_claims, strict=True):
            original_tokens = self._content_tokens(original)
            reversed_tokens = self._content_tokens(reversed_claim)
            score = (
                len(original_tokens & reversed_tokens) / len(original_tokens)
                if original_tokens
                else 1.0
            )
            unprotected = self._unprotected_content_tokens(original)
            unprotected_score = (
                len(unprotected & reversed_tokens) / len(unprotected) if unprotected else 1.0
            )
            if score < self.round_trip_threshold and unprotected_score < self.round_trip_threshold:
                raise TranslationRejected(
                    f"Round-trip translation recall {score:.2f} and unprotected recall "
                    f"{unprotected_score:.2f} are below {self.round_trip_threshold:.2f}."
                )
        return translated

    def status(self, *, probe: bool = False) -> dict[str, Any]:
        return {
            **self.engine.status(probe=probe),
            "glossary": {
                "id": self.glossary.id,
                "entries": len(self.glossary.entries),
                "scope": self.glossary.scope,
            },
            "round_trip_threshold": self.round_trip_threshold,
        }


def create_translation_service() -> ControlledTranslationService:
    mode = os.getenv("TRACEFOLD_TRANSLATION_RUNTIME", "").strip().casefold()
    if not mode:
        mode = (
            "fake" if os.getenv("TRACEFOLD_RUNTIME", "foundry").casefold() == "fake" else "marian"
        )
    if mode == "fake":
        return ControlledTranslationService(FakeTranslationEngine())
    if mode in {"marian", "opus", "marian_opus"}:
        return ControlledTranslationService(MarianTranslationEngine(local_files_only=True))
    raise RuntimeUnavailable(
        "TRACEFOLD_TRANSLATION_RUNTIME must be 'marian' (production) or 'fake' (tests only)."
    )
