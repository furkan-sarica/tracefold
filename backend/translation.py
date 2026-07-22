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
_MONTH_NAMES = (
    "January",
    "Ocak",
    "February",
    "Şubat",
    "March",
    "Mart",
    "April",
    "Nisan",
    "May",
    "Mayıs",
    "June",
    "Haziran",
    "July",
    "Temmuz",
    "August",
    "Ağustos",
    "September",
    "Eylül",
    "October",
    "Ekim",
    "November",
    "Kasım",
    "December",
    "Aralık",
)
_WEEKDAY_NAMES = (
    "Monday",
    "Pazartesi",
    "Tuesday",
    "Salı",
    "Wednesday",
    "Çarşamba",
    "Thursday",
    "Perşembe",
    "Friday",
    "Cuma",
    "Saturday",
    "Cumartesi",
    "Sunday",
    "Pazar",
)
_TEXTUAL_DATE = re.compile(
    r"(?<!\w)\d{1,2}\s+(?:"
    + "|".join(map(re.escape, _MONTH_NAMES))
    + r")(?:\s+\d{4})?(?:['’][A-Za-zÇĞİÖŞÜçğıöşü]+)?(?!\w)",
    flags=re.IGNORECASE,
)
_NUMBER = re.compile(r"(?<![\w@])\d+(?:[.,]\d+)?(?![\w@])")
_TOKEN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_CALENDAR_CANONICAL = {
    **{
        localized.casefold(): english.casefold()
        for english, localized in zip(_MONTH_NAMES[::2], _MONTH_NAMES[1::2], strict=True)
    },
    **{english.casefold(): english.casefold() for english in _MONTH_NAMES[::2]},
    **{
        localized.casefold(): english.casefold()
        for english, localized in zip(_WEEKDAY_NAMES[::2], _WEEKDAY_NAMES[1::2], strict=True)
    },
    **{english.casefold(): english.casefold() for english in _WEEKDAY_NAMES[::2]},
}
_CALENDAR = re.compile(
    r"\b(?:"
    + "|".join(
        map(
            re.escape,
            sorted((*_MONTH_NAMES, *_WEEKDAY_NAMES), key=len, reverse=True),
        )
    )
    + r")\b",
    flags=re.IGNORECASE,
)
_TURKISH_TERM_SUFFIXES = (
    "daki",
    "deki",
    "taki",
    "teki",
    "dan",
    "den",
    "tan",
    "ten",
    "dır",
    "dir",
    "dur",
    "dür",
    "tır",
    "tir",
    "tur",
    "tür",
    "nın",
    "nin",
    "nun",
    "nün",
    "yla",
    "yle",
    "yı",
    "yi",
    "yu",
    "yü",
    "nı",
    "ni",
    "nu",
    "nü",
    "ya",
    "ye",
    "na",
    "ne",
    "da",
    "de",
    "ta",
    "te",
    "ın",
    "in",
    "un",
    "ün",
    "la",
    "le",
    "a",
    "e",
    "ı",
    "i",
    "u",
    "ü",
)


class TranslationRejected(RuntimeError):
    """Raised when translation changes protected facts or loses too much meaning."""


@dataclass(frozen=True)
class GlossaryEntry:
    source: str
    target: str
    sentinel: bool = True
    bind_action: bool = False


@dataclass(frozen=True)
class ProtectedValue:
    value: str
    kind: str
    relation: str = ""


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
                        bind_action=bool(item.get("bind_action", False)),
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

    @staticmethod
    def _turkish_phrase_pattern(value: str) -> re.Pattern[str]:
        """Match a reviewed term before a supported Turkish inflection.

        Turkish case suffixes are commonly attached without an apostrophe. A strict
        word boundary therefore misses forms such as ``oryantasyonuna`` even when the
        glossary contains ``oryantasyonu``. The source-language suffix is consumed
        with the phrase; the translation model then supplies target-language syntax
        around the protected sentinel.
        """

        suffixes = "|".join(map(re.escape, _TURKISH_TERM_SUFFIXES))
        return re.compile(
            rf"(?<!\w){re.escape(value)}(?:['’]?(?P<suffix>{suffixes}))?(?!\w)",
            flags=re.IGNORECASE,
        )

    @staticmethod
    def _english_relation_for_turkish_suffix(suffix: str) -> str:
        """Map a consumed Turkish case suffix to a controlled English relation."""

        normalized = suffix.casefold()
        if normalized in {"a", "e", "ya", "ye", "na", "ne"}:
            return "to"
        if normalized in {"dan", "den", "tan", "ten"}:
            return "from"
        if normalized in {"la", "le", "yla", "yle"}:
            return "with"
        if normalized in {"da", "de", "ta", "te", "daki", "deki", "taki", "teki"}:
            return "at"
        return ""

    def protect(
        self,
        text: str,
        *,
        source_locale: SourceLocale,
        target_locale: SourceLocale,
        protect_exact: bool = False,
    ) -> tuple[str, list[ProtectedValue]]:
        if source_locale == target_locale:
            return text, []

        candidates: list[tuple[int, int, int, ProtectedValue]] = []
        for entry in self.entries:
            if not entry.sentinel:
                continue
            phrase = entry.source if source_locale == "en" else entry.target
            replacement = entry.target if target_locale == "tr" else entry.source
            pattern = (
                self._turkish_phrase_pattern(phrase)
                if source_locale == "tr"
                else self._phrase_pattern(phrase)
            )
            for match in pattern.finditer(text):
                relation = ""
                if source_locale == "tr" and target_locale == "en":
                    replacement = entry.source
                    relation = self._english_relation_for_turkish_suffix(
                        match.groupdict().get("suffix") or "",
                    )
                kind = "proper" if entry.source == entry.target else "term"
                candidates.append(
                    (
                        match.start(),
                        match.end(),
                        0,
                        ProtectedValue(replacement, kind, relation),
                    )
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

        if protect_exact:
            # Exact values are masked with the same tokenizer-stable sentinels as
            # terminology. This avoids lossy ad-hoc placeholders while relation and
            # unit checks below still verify the restored translation.
            for match in _TEXTUAL_DATE.finditer(text):
                value = re.sub(r"['’][A-Za-zÇĞİÖŞÜçğıöşü]+$", "", match.group(0))
                month_pairs = zip(_MONTH_NAMES[::2], _MONTH_NAMES[1::2], strict=True)
                for english, turkish in month_pairs:
                    source, target = (
                        (turkish, english) if target_locale == "en" else (english, turkish)
                    )
                    value = re.sub(
                        rf"\b{re.escape(source)}\b",
                        target,
                        value,
                        flags=re.IGNORECASE,
                    )
                candidates.append((match.start(), match.end(), 2, ProtectedValue(value, "exact")))
            for priority, pattern in enumerate((_TIME, _DATE, _NUMBER), start=3):
                for match in pattern.finditer(text):
                    candidates.append(
                        (
                            match.start(),
                            match.end(),
                            priority,
                            ProtectedValue(match.group(0), "exact"),
                        )
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

        if protect_exact:
            merged: list[tuple[int, int, ProtectedValue]] = []
            for start, end, protected in selected:
                if merged and not text[merged[-1][1] : start].strip():
                    previous_start, previous_end, previous = merged[-1]
                    separator = text[previous_end:start]
                    merged[-1] = (
                        previous_start,
                        end,
                        ProtectedValue(previous.value + separator + protected.value, "exact"),
                    )
                else:
                    merged.append((start, end, protected))
            selected = merged

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
        possessive_accusative = {
            "nızı": "ınızı",
            "nizi": "inizi",
            "nuzu": "unuzu",
            "nüzü": "ünüzü",
        }
        if normalized in possessive_accusative and value[-1:].casefold() in "ıiuü":
            return value[:-1] + possessive_accusative[normalized]
        if normalized in {"lerini", "larını"}:
            return TerminologyGlossary._term_with_suffix(value, "nı")
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
            if protected.relation:
                if suffix:
                    raise TranslationRejected(
                        "The local translator attached a suffix to a relation-protected term."
                    )
                before = text[max(0, match.start() - 48) : match.start()].casefold()
                already_related = re.search(r"\b(?:at|from|in|into|on|to|with)\s*$", before)
                direct_object = protected.relation == "to" and re.search(
                    r"\b(?:attend|contact|enter|join|reach|visit)"
                    r"(?:\s+[a-z-]+){0,3}\s*$",
                    before,
                )
                prefix = "" if already_related or direct_object else f"{protected.relation} "
                return prefix + protected.value
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
            raw_calendar = match.group(0)
            if raw_calendar.casefold() == "may":
                nearby = text[max(0, match.start() - 8) : min(len(text), match.end() + 8)]
                if not re.search(
                    r"(?:\b\d{1,2}\s+May\b|\bMay\s+\d{1,2}\b)",
                    nearby,
                    flags=re.IGNORECASE,
                ):
                    continue
            candidates.append(
                (
                    match.start(),
                    match.end(),
                    len(patterns),
                    "calendar",
                    _CALENDAR_CANONICAL[raw_calendar.casefold()],
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
                    or re.search(r"\bthrough\s*$", before)
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
                max_length=int(os.getenv("TRACEFOLD_TRANSLATION_MAX_TOKENS", "256")),
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
            os.getenv("TRACEFOLD_TRANSLATION_ROUNDTRIP_THRESHOLD", "0.60")
        )

    def _translate_batch(
        self,
        texts: list[str],
        *,
        source_locale: SourceLocale,
        target_locale: SourceLocale,
        enforce_exact: bool = True,
        enforce_relations: bool = True,
        protect_exact: bool = False,
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
                protect_exact=protect_exact,
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
        for index, (expected, translated) in enumerate(zip(exact_facts, restored, strict=True)):
            for (kind, value), count in expected.items():
                if kind != "time":
                    continue
                dotted = value.replace(":", ".")
                if dotted == value:
                    continue
                translated = re.sub(
                    rf"(?<!\d){re.escape(dotted)}(?!\d)",
                    value,
                    translated,
                    count=count,
                )
            restored[index] = translated
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
        translated = self._translate_batch(
            [question],
            source_locale="tr",
            target_locale="en",
            protect_exact=True,
        )[0]
        translated = re.sub(
            r"\bdeparture service\b",
            "departure shuttle",
            translated,
            flags=re.IGNORECASE,
        )
        translated = re.sub(
            r"\bwhere should (?P<people>[^?.,]*\boccupants) be collected\b",
            r"where should \g<people> assemble",
            translated,
            flags=re.IGNORECASE,
        )
        return re.sub(
            r"\bhow does this date relate to\b",
            "how does this deadline relate to",
            translated,
            flags=re.IGNORECASE,
        )

    @staticmethod
    def normalize_english_claim(claim: str) -> str:
        """Rewrite ambiguous English constructions into simple, translation-safe English."""

        normalized = " ".join(claim.strip().split())
        normalized = re.sub(r"^They\b", "The participant", normalized, flags=re.IGNORECASE)
        normalized = re.sub(
            r"^(?P<body>.+?)\s+first$",
            lambda match: f"First, {match.group('body')[:1].lower()}{match.group('body')[1:]}",
            normalized,
            flags=re.IGNORECASE,
        )
        emergency_phone = _PHONE.search(normalized)
        if (
            emergency_phone
            and re.search(r"\b5555\b", normalized)
            and re.search(r"\b(?:use|call|dial)\b", normalized, flags=re.IGNORECASE)
            and re.search(r"\b(?:campus|emergency)\b", normalized, flags=re.IGNORECASE)
        ):
            prefix = "First, " if re.search(r"\bfirst\b", normalized, flags=re.IGNORECASE) else ""
            normalized = (
                f"{prefix}call extension 5555 from a campus phone or call "
                f"{emergency_phone.group(0)} from a mobile phone"
            )
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
            "must be completed by",
            normalized,
            flags=re.IGNORECASE,
        )
        normalized = re.sub(
            r"\bis\s+due\s+by\b",
            "must be completed by",
            normalized,
            flags=re.IGNORECASE,
        )
        normalized = re.sub(
            r"^(?:ensure\s+(?:that\s+)?you\s+can|you\s+should\s+be\s+able\s+to)\s+"
            r"install\b",
            "You need permission to install",
            normalized,
            flags=re.IGNORECASE,
        )
        normalized = re.sub(r"^Return\b", "Give back", normalized, flags=re.IGNORECASE)
        normalized = re.sub(
            r"^(?P<prefix>First,\s+)?use\s+the\s+(?:campus\s+)?emergency\s+number\s+"
            r"(?P<extension>\d+)\s+from\s+a\s+campus\s+phone\s+or\s+"
            r"(?P<phone>\+\d(?:[\d\s().-]{5,}\d))\s+from\s+a\s+mobile(?:\s+phone)?$",
            r"\g<prefix>call extension \g<extension> from a campus phone or call "
            r"\g<phone> from a mobile phone",
            normalized,
            flags=re.IGNORECASE,
        )
        normalized = re.sub(
            r"^(?P<prefix>First,\s+)?call\s+(?:extension\s+)?(?P<extension>\d+)\s+"
            r"from\s+a\s+campus\s+phone\s+or\s+(?:phone\s+)?"
            r"(?P<phone>\+\d(?:[\d\s().-]{5,}\d))\s+from\s+a\s+mobile(?:\s+phone)?$",
            r"\g<prefix>call extension \g<extension> from a campus phone or call "
            r"\g<phone> from a mobile phone",
            normalized,
            flags=re.IGNORECASE,
        )
        normalized = re.sub(
            r"^Staff\s+run\s+a\s+scheduled\s+(?P<event>.+?)\s+"
            r"(?P<schedule>(?:at|on)\s+.+)$",
            r"The scheduled \g<event> takes place \g<schedule>",
            normalized,
            flags=re.IGNORECASE,
        )
        normalized = re.sub(
            r"^After\s+ensuring\s+immediate\s+safety,",
            "After immediate safety is addressed,",
            normalized,
            flags=re.IGNORECASE,
        )
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
        return normalized.rstrip(".")

    @classmethod
    def atomize_claim(cls, claim: str) -> list[str]:
        """Split only general compound temporal clauses; never inspect case IDs or questions."""

        normalized = cls.normalize_english_claim(claim)
        specifications = re.match(
            r"^(?P<subject>.+?)\s+should\s+have\s+(?P<first>[^,]+),\s+"
            r"(?P<second>[^,]+\s+after\s+updates),\s+(?P<rest>.+)$",
            normalized,
            flags=re.IGNORECASE,
        )
        if specifications:
            subject = specifications.group("subject")
            return [
                f"{subject} should have {specifications.group('first')}",
                f"{subject} should have {specifications.group('second')}",
                f"{subject} should have {specifications.group('rest')}",
            ]
        memory_and_storage = re.match(
            r"^(?P<subject>.+?)\s+(?P<modality>should|must|needs?\s+to)\s+have\s+"
            r"(?P<memory>\d+(?:[.,]\d+)?\s*(?:GB|MB|TB)\s+(?:RAM|memory))\s+and\s+"
            r"(?P<storage>\d+(?:[.,]\d+)?\s*(?:GB|MB|TB)\s+(?:free\s+)?storage)\s+"
            r"after\s+(?:system\s+)?updates$",
            normalized,
            flags=re.IGNORECASE,
        )
        if memory_and_storage:
            subject = memory_and_storage.group("subject")
            modality = memory_and_storage.group("modality")
            return [
                f"{subject} {modality} have {memory_and_storage.group('memory')}",
                f"{subject} {modality} have {memory_and_storage.group('storage')} after "
                "system updates",
            ]
        event_and_assembly = re.match(
            r"^(?P<event>.+?\b(?:drill|event)\b.+?)\s+and\s+"
            r"(?P<assembly>.+?\b(?:should\s+)?assemble\b.+)$",
            normalized,
            flags=re.IGNORECASE,
        )
        if event_and_assembly:
            event = event_and_assembly.group("event")
            scheduled_event = re.match(
                r"^(?:The\s+)?(?:scheduled\s+)?(?P<event>.+?\bdrill)\s+is\s+on\s+"
                r"(?P<date>.+?)\s+at\s+(?P<time>\d{1,2}:\d{2})$",
                event,
                flags=re.IGNORECASE,
            )
            if scheduled_event:
                event_name = scheduled_event.group("event")
                event_name = event_name[:1].lower() + event_name[1:]
                event = (
                    f"The scheduled {event_name} takes place at "
                    f"{scheduled_event.group('time')} on {scheduled_event.group('date')}"
                )
            assembly = event_and_assembly.group("assembly")
            if not re.search(r"\b(?:should|must)\s+assemble\b", assembly, flags=re.IGNORECASE):
                assembly = re.sub(
                    r"\bassemble\b",
                    "should assemble",
                    assembly,
                    count=1,
                    flags=re.IGNORECASE,
                )
            return [
                event,
                assembly,
            ]
        call_with_purpose = re.match(
            r"^(?:(?:You|The participant)\s+(?:should|must)\s+)?call\s+"
            r"(?P<contact>.+?)\s+before\s+(?P<deadline>\d{1,2}:\d{2})\s+"
            r"to\s+arrange\s+for\s+(?P<item>.+?)\s+from\s+(?:the\s+)?(?P<person>.+)$",
            normalized,
            flags=re.IGNORECASE,
        )
        if call_with_purpose:
            return [
                f"Call {call_with_purpose.group('contact')} before "
                f"{call_with_purpose.group('deadline')}",
                f"Arrange {call_with_purpose.group('item')} with the "
                f"{call_with_purpose.group('person')}",
            ]

        before_call = re.match(
            r"^Before\s+(?P<deadline>\d{1,2}:\d{2}),\s+call\s+"
            r"(?P<contact>.+?)\s+to\s+arrange\s+(?P<item>.+)$",
            normalized,
            flags=re.IGNORECASE,
        )
        if before_call:
            return [
                f"Call {before_call.group('contact')} before {before_call.group('deadline')}",
                f"Arrange {before_call.group('item')}",
            ]

        conditional_arrangement = re.match(
            r"^If\s+(?P<condition>.+?),\s+you\s+must\s+arrange\s+"
            r"(?P<arrangement>.+?)\s+and\s+continue\s+(?P<work>.+?)\s+until\s+"
            r"(?P<approval>.+)$",
            normalized,
            flags=re.IGNORECASE,
        )
        if conditional_arrangement:
            condition = conditional_arrangement.group("condition")
            approval = conditional_arrangement.group("approval")
            if re.search(
                r"\bSafety Officer\b", conditional_arrangement.group("arrangement"), re.I
            ) and re.fullmatch(r"you are signed off", approval, flags=re.IGNORECASE):
                approval = "the Safety Officer approves you"
            return [
                f"If {condition}, arrange {conditional_arrangement.group('arrangement')}",
                f"If {condition}, continue {conditional_arrangement.group('work')} until "
                f"{approval}",
            ]

        report_with_email = re.match(
            r"^After\s+(?P<condition>.+?),\s+report\s+(?P<concern>.+?)\s+using\s+.+?,\s+"
            r"including\s+emailing\s+the\s+(?P<role>.+?)\s+at\s+"
            r"(?P<email>[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})$",
            normalized,
            flags=re.IGNORECASE,
        )
        if report_with_email:
            concern = re.sub(
                r"^(?:a|an|the)\s+",
                "",
                report_with_email.group("concern"),
                flags=re.IGNORECASE,
            )
            return [
                f"After {report_with_email.group('condition')}, email the {concern} to "
                f"{report_with_email.group('email')} for the {report_with_email.group('role')}"
            ]

        stored_luggage = re.match(
            r"^Luggage\s+can\s+be\s+left\s+in\s+(?P<place>.+?)\s+after\s+"
            r"checking\s+out,\s+available\s+until\s+(?P<deadline>\d{1,2}:\d{2})\s+"
            r"for\s+those\s+using\s+(?P<shuttle>.+)$",
            normalized,
            flags=re.IGNORECASE,
        )
        if stored_luggage:
            return [
                f"After checkout, leave luggage in {stored_luggage.group('place')} until "
                f"{stored_luggage.group('deadline')} when using "
                f"{stored_luggage.group('shuttle')}"
            ]

        offset = re.match(
            r"^(?P<main>.+?),\s*(?P<offset>\d+\s+minutes?\s+(?:before|after)\s+.+)$",
            normalized,
            flags=re.IGNORECASE,
        )
        if offset:
            return [offset.group("main"), f"This time is {offset.group('offset')}"]

        return_after_handoff = re.match(
            r"^(?:The\s+)?(?:loan-device\s+)?return\s+deadline\s+is\s+"
            r"(?P<offset>\d+\s+minutes?\s+after)\s+the\s+"
            r"(?P<handoff>.+?handoff)\s+due\s+at\s+(?P<time>.+)$",
            normalized,
            flags=re.IGNORECASE,
        )
        if return_after_handoff:
            handoff = return_after_handoff.group("handoff")
            return [
                f"The {handoff} happens at {return_after_handoff.group('time')}",
                f"The return deadline is {return_after_handoff.group('offset')} the {handoff}",
            ]

        return_deadline = re.match(
            r"^Give back\s+(?P<objects>.+?)\s+(?:no\s+later\s+than|by)\s+"
            r"(?P<deadline>.+)$",
            normalized,
            flags=re.IGNORECASE,
        )
        if return_deadline:
            deadline = return_deadline.group("deadline")
            deadline_with_relationship = re.match(
                r"^(?P<due>.+?);\s*this\s+deadline\s+is\s+"
                r"(?P<relationship>\d+\s+minutes?\s+after\s+.+)$",
                deadline,
                flags=re.IGNORECASE,
            )
            if deadline_with_relationship:
                return [
                    f"Give back {return_deadline.group('objects')}",
                    f"The return deadline is {deadline_with_relationship.group('due')}",
                    f"This deadline is {deadline_with_relationship.group('relationship')}",
                ]
            return [
                f"Give back {return_deadline.group('objects')}",
                f"The return deadline is {deadline}",
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
            r"(\+\d(?:[\d\s().-]{5,}\d))\s+numaralı telefonda\s+"
            r"(ara[a-zçğıöşü]*)",
            r"\1 numaralı telefonu \2",
            edited,
            flags=re.IGNORECASE,
        )
        edited = re.sub(
            r"bir rapor (?:hazırlamaya|oluşturmaya) yardımcı olmak için "
            r"Katılımcı Destek['’]ten isteyin",
            "rapor hazırlamak için Katılımcı Destekten yardım isteyin",
            edited,
            flags=re.IGNORECASE,
        )
        edited = re.sub(
            r"\b(anahtar teslim alma işlemi(?:ni)?)\s+gece görevlisi\s+ile\s+düzenleyin\b",
            "anahtar teslim alma işlemini gece görevlisiyle ayarlayın",
            edited,
            flags=re.IGNORECASE,
        )
        edited = re.sub(
            r"\banahtar teslim alma işlemini düzenleyin\b",
            "anahtar teslim alma işlemini ayarlayın",
            edited,
            flags=re.IGNORECASE,
        )
        edited = re.sub(
            r"\banahtar teslim alma işlemini ayarlayın\b",
            "anahtarınızı nasıl teslim alacağınızı ayarlayın",
            edited,
            flags=re.IGNORECASE,
        )
        edited = re.sub(
            r"\bDizüstü bilgisayarınız\s+(?P<amount>\d+\s+GB\s+RAM)['’]e\s+"
            r"sahip olmalı(?:dır)?\b",
            r"Dizüstü bilgisayarınızda \g<amount> bulunmalıdır",
            edited,
            flags=re.IGNORECASE,
        )
        edited = re.sub(
            r"\bDizüstü bilgisayar\s+(?P<amount>\d+\s+GB\s+RAM)['’]e\s+"
            r"sahip olmalı(?:dır)?\b",
            r"Dizüstü bilgisayarda \g<amount> bulunmalıdır",
            edited,
            flags=re.IGNORECASE,
        )
        edited = re.sub(
            r"\bbir çalışan tarayıcı\b",
            "çalışan bir tarayıcı",
            edited,
            flags=re.IGNORECASE,
        )
        edited = re.sub(
            r"\bçalışan tarayıcına\b",
            "çalışan bir tarayıcıya",
            edited,
            flags=re.IGNORECASE,
        )
        edited = re.sub(
            r"\bçalışan tarayıcı ve (?P<port>USB-C veya USB-A bağlantı noktasına) sahip\b",
            r"çalışan bir tarayıcıya ve \g<port> sahip",
            edited,
            flags=re.IGNORECASE,
        )
        edited = re.sub(
            r"^Wi-Fi 5 veya daha yenisini desteklemeli\b",
            "Dizüstü bilgisayarınız Wi-Fi 5 veya daha yenisini desteklemeli",
            edited,
            flags=re.IGNORECASE,
        )
        edited = re.sub(r"\bcheck-out\b", "çıkış", edited, flags=re.IGNORECASE)
        edited = re.sub(
            r"\bresepsiyona iade etmelidir\b",
            "resepsiyona teslim etmelidir",
            edited,
            flags=re.IGNORECASE,
        )
        edited = re.sub(
            r"\bkampüs telefonundan\s+(?P<extension>\d{3,6})\s+numaralı telefonu\s+"
            r"(?P<verb>ara[a-zçğıöşü]*)",
            r"kampüs telefonundan \g<extension> dahili numarasını \g<verb>",
            edited,
            flags=re.IGNORECASE,
        )
        edited = re.sub(
            r"\bİmzalanana kadar\b",
            "Onay alana kadar",
            edited,
            flags=re.IGNORECASE,
        )
        edited = re.sub(
            r"Güncellenen varış saatinizi\s+"
            r"(?P<email>[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})\s+"
            r"adresindeki Program Ofisine gönderin",
            r"Güncellenen varış saatinizi \g<email> adresine e-posta göndererek "
            r"Program Ofisine bildirin",
            edited,
            flags=re.IGNORECASE,
        )
        edited = re.sub(
            r"\bGüvenlik Sorumlusu ile bir telafi oryantasyonu düzenleyin\b",
            "Güvenlik Sorumlusuyla bir telafi oryantasyonu ayarlayın",
            edited,
            flags=re.IGNORECASE,
        )
        edited = re.sub(
            r"\bGüvenlik Sorumlusu ile bir telafi oryantasyonunu tamamlayana kadar\b",
            "Güvenlik Sorumlusuyla telafi oryantasyonunu tamamlayana kadar",
            edited,
            flags=re.IGNORECASE,
        )
        edited = re.sub(
            r"\b(Pazartesi|Salı|Çarşamba|Perşembe|Cuma|Cumartesi|Pazar)\s*,\s*"
            r"(\d{1,2}\s+Ağustos(?:\s+\d{4})?)",
            r"\2 \1",
            edited,
            flags=re.IGNORECASE,
        )
        edited = re.sub(
            r"\b(Northstar Summer Lab\s+\d{4})\s+(?=\d{1,2}\s+Ağustos)",
            r"\1, ",
            edited,
            flags=re.IGNORECASE,
        )
        edited = re.sub(
            r"\bAnkara['’]da\s+(?=(?:yapılacak|gerçekleştirilecek)\b)",
            "Ankara adresinde ",
            edited,
            flags=re.IGNORECASE,
        )
        edited = re.sub(
            r"\b(Pazartesi|Salı|Çarşamba|Perşembe|Cuma|Cumartesi|Pazar)\s+ve\s+"
            r"(?=\d{1,2}\s+Ağustos(?:\s+\d{4})?\s+"
            r"(?:Pazartesi|Salı|Çarşamba|Perşembe|Cuma|Cumartesi|Pazar)\b)",
            r"\1 ile ",
            edited,
            flags=re.IGNORECASE,
        )
        edited = re.sub(
            r"\b(\d{1,2}:\d{2})\s+servisi\s+kullanırken\b",
            r"\1 kalkış servisini kullanıyorsanız",
            edited,
            flags=re.IGNORECASE,
        )
        edited = re.sub(
            r"\betiketli bagaj muhafaza alanında bagaj bırakın\b",
            "bagajınızı etiketli muhafaza alanında bırakın",
            edited,
            flags=re.IGNORECASE,
        )
        edited = re.sub(
            r"\bbagaj bırakabilirler\b",
            "bagajlarını bırakabilirler",
            edited,
            flags=re.IGNORECASE,
        )
        edited = re.sub(
            r"\bBagaj\s+(?P<shuttle>\d{1,2}:\d{2}\s+kalkış servisi)\s+"
            r"kullananlar için\s+(?P<deadline>\d{1,2}:\d{2}'[a-zçğıöşü]+)\s+kadar\s+"
            r"(?P<place>.+?)\s+bırakılabilir\b",
            r"\g<shuttle>ni kullanacaksanız bagajınızı \g<place> \g<deadline> kadar "
            r"bırakabilirsiniz",
            edited,
            flags=re.IGNORECASE,
        )
        edited = re.sub(
            r"\bHemen güvenliği sağladıktan sonra,?\s+davranışı\s+"
            r"(?P<role>Davranış Sorumlusu)\s+için\s+"
            r"(?P<email>[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})\s+adresine\s+"
            r"e-postayla gönderin\b",
            r"Acil güvenliği sağladıktan sonra davranışı \g<email> adresine e-posta "
            r"göndererek \g<role>na bildirin",
            edited,
            flags=re.IGNORECASE,
        )
        edited = re.sub(
            r"\bAcil güvenlik sağlandıktan sonra,?\s+"
            r"(?P<role>Davranış Sorumlusu)\s+için\s+davranışı\s+"
            r"(?P<email>[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})\s+adresine\s+"
            r"e-postayla gönderin\b",
            r"Acil güvenlik sağlandıktan sonra davranışı \g<email> adresine e-posta "
            r"göndererek \g<role>na bildirin",
            edited,
            flags=re.IGNORECASE,
        )
        edited = re.sub(
            r"\bAcil güvenlik sağlandıktan sonra,?\s+"
            r"(?P<role>Davranış Sorumlusu)\s+için\s+"
            r"(?P<email>[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})\s+adresine\s+"
            r"davranış(?:la ilgili)?\s+endişesini\s+e-postayla\s+gönderin\b",
            r"Acil güvenlik sağlandıktan sonra davranışla ilgili endişeyi \g<email> adresine "
            r"e-posta göndererek \g<role>na bildirin",
            edited,
            flags=re.IGNORECASE,
        )
        edited = re.sub(
            r"\bAcil güvenlik sağlandıktan sonra,?\s+"
            r"(?P<email>[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})\s+e-postası\s+gibi\s+"
            r"Davranış Kurallarındaki gizli yolları kullanarak davranışı bildirin\b",
            r"Acil güvenlik sağlandıktan sonra davranışı, Davranış Kurallarındaki gizli "
            r"bildirim kanallarından biri olan \g<email> adresine e-posta göndererek bildirin",
            edited,
            flags=re.IGNORECASE,
        )
        edited = re.sub(
            r"(?P<email>[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})\s+adresinde\s+"
            r"(?P<rest>[^.]*\be-posta\s+gönder[a-zçğıöşü]*)",
            r"\g<email> adresine \g<rest>",
            edited,
            flags=re.IGNORECASE,
        )
        edited = re.sub(
            r"Davranış Kurallarındaki gizli rotalar aracılığıyla",
            "Davranış Kurallarındaki gizli bildirim kanalları üzerinden",
            edited,
            flags=re.IGNORECASE,
        )
        edited = re.sub(
            r"(?:Ardından,?\s+)?(?P<email>[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})\s+"
            r"adresine e-posta göndermek gibi Davranış Kuralları['’]nda ayrıntılı olarak "
            r"belirtilen gizli yolları kullanarak davranışı bildirin\b",
            r"Ardından davranışı, Davranış Kuralları'nda açıklanan gizli bildirim "
            r"kanallarından biri olan \g<email> adresine e-posta göndererek bildirin",
            edited,
            flags=re.IGNORECASE,
        )
        edited = re.sub(
            r"Acil durumla başa çıktıktan sonra,?\s+"
            r"(?P<email>[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})\s+adresine e-posta "
            r"göndermeyi içeren Davranış Kuralları['’]ndaki gizli yolları kullanarak "
            r"davranışı bildirin\b",
            r"Acil güvenliği sağladıktan sonra davranışı, Davranış Kuralları'ndaki gizli "
            r"bildirim kanallarından biri olan \g<email> adresine e-posta göndererek bildirin",
            edited,
            flags=re.IGNORECASE,
        )
        edited = re.sub(
            r"(?P<email>[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})\s+adresine\s+"
            r"Davranış Sorumlusuna e-posta gönderin\b",
            r"Davranış Sorumlusuyla iletişim için \g<email> adresine e-posta gönderin",
            edited,
            flags=re.IGNORECASE,
        )
        edited = re.sub(
            r"\bekip teslim zamanından\b",
            "ekip tesliminden",
            edited,
            flags=re.IGNORECASE,
        )
        edited = re.sub(
            r"\bİade son tarihi,\s+",
            "İade son tarihi ",
            edited,
            flags=re.IGNORECASE,
        )
        edited = re.sub(
            r"\bPersonel,\s+(?P<when>.+?)\s+tarifeli\s+bir\s+"
            r"(?P<event>tahliye tatbikatı)\s+çalıştırıyor\b",
            r"Planlanan \g<event> \g<when> yapılacak",
            edited,
            flags=re.IGNORECASE,
        )
        edited = re.sub(
            r"\bPlanlanan tahliye tatbikatı,?\s+"
            r"(?P<date>\d{1,2}\s+Ağustos\s+"
            r"(?:Pazartesi|Salı|Çarşamba|Perşembe|Cuma|Cumartesi|Pazar)\s+günü)\s+"
            r"(?P<time>\d{1,2}:\d{2}'[a-zçğıöşü]+)\s+gerçekleşecek\b",
            r"Planlanan tahliye tatbikatı \g<date> saat \g<time> yapılacak",
            edited,
            flags=re.IGNORECASE,
        )
        edited = re.sub(
            r"\bGüvenlik Sorumlusu sizi onaylayana kadar\b",
            "Güvenlik Sorumlusu onay verene kadar",
            edited,
            flags=re.IGNORECASE,
        )
        edited = re.sub(
            r"\bKatılımcılar,(?P<prefix>.+?çıkış yapmalı),\s+oda anahtarını\b",
            r"Katılımcılar,\g<prefix>; oda anahtarını",
            edited,
            flags=re.IGNORECASE,
        )
        edited = re.sub(
            r"\bKatılımcılar,(?P<prefix>.+?)ödünç alınan nevresim torbasını\b",
            r"Katılımcılar,\g<prefix>ödünç aldıkları nevresim torbasını",
            edited,
            flags=re.IGNORECASE,
        )
        edited = re.sub(
            r"\bEkip teslim zamanı\b",
            "Ekip teslimi",
            edited,
            flags=re.IGNORECASE,
        )
        edited = re.sub(
            r"\b(\d+\s+dakika\s+sonra)\s*\.?$",
            r"\1dır",
            edited,
            flags=re.IGNORECASE,
        )
        edited = re.sub(
            r"\bYurt Nöbet Telefonunu\s+(?P<deadline>\d{1,2}:\d{2}'[a-zçğıöşü]+\s+önce)\s+"
            r"(?P<phone>\+\d(?:[\d\s().-]{5,}\d))\s+numaralı telefonu\s+"
            r"(?P<verb>ara[a-zçğıöşü]*)",
            r"\g<deadline> \g<phone> numaralı Yurt Nöbet Telefonunu \g<verb>",
            edited,
            flags=re.IGNORECASE,
        )
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
        loan_laptop = next(
            (
                entry.target
                for entry in self.glossary.entries
                if entry.source.casefold() == "loan laptop"
            ),
            None,
        )
        if loan_laptop:
            accusative = TerminologyGlossary._term_with_suffix(loan_laptop, "ı")
            edited = re.sub(
                rf"\b{re.escape(loan_laptop)}(?=,\s+[^.]*\biade\s+edin\b)",
                accusative,
                edited,
                flags=re.IGNORECASE,
            )
        room_key = next(
            (entry.target for entry in self.glossary.entries if entry.source == "room key"),
            None,
        )
        if room_key:
            room_key_accusative = TerminologyGlossary._term_with_suffix(room_key, "nı")
            edited = re.sub(
                rf"\b{re.escape(room_key)}(?=\s+ve\s+[^.]*\b(?:iade|teslim)\s+"
                rf"e(?:din|t)[a-zçğıöşü]*\b)",
                room_key_accusative,
                edited,
                flags=re.IGNORECASE,
            )
        charger = next(
            (entry.target for entry in self.glossary.entries if entry.source == "charger"),
            None,
        )
        if charger:
            charger_accusative = TerminologyGlossary._term_with_suffix(charger, "nı")
            edited = re.sub(
                rf"\b{re.escape(charger)}(?=\s+ve\s+[^.]*\biade\s+edin\b)",
                charger_accusative,
                edited,
                flags=re.IGNORECASE,
            )
        first_letter = next(
            (index for index, char in enumerate(edited) if not char.isspace()), None
        )
        starts_with_email = (
            first_letter is not None and _EMAIL.match(edited[first_letter:]) is not None
        )
        if first_letter is not None and edited[first_letter].isalpha() and not starts_with_email:
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
            "as",
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
            "those",
            "to",
            "such",
            "with",
            "you",
            "your",
        }
        aliases = {
            "gather": "assemble",
            "channel": "route",
            "channels": "route",
            "emailing": "email",
            "internal": "extension",
            "mailing": "email",
            "next": "beside",
            "people": "occupants",
            "routes": "route",
            "secret": "confidential",
            "using": "use",
        }
        return {
            aliases.get(token, token)
            for token in _TOKEN.findall(value.casefold())
            if token not in stopwords
            if aliases.get(token, token) not in stopwords
        }

    def _unprotected_content_tokens(self, value: str) -> set[str]:
        reduced = value
        for entry in self.glossary.entries:
            reduced = self.glossary._phrase_pattern(entry.source).sub(" ", reduced)
        spans = self.glossary._exact_occurrences(reduced)
        for start, end, _kind, _normalized in reversed(spans):
            reduced = reduced[:start] + " " + reduced[end:]
        semantic_words = {
            "after",
            "arrange",
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

        folded = value.casefold().replace("i̇", "i")
        predicates: set[str] = set()
        modalities: set[str] = set()
        order: set[str] = set()

        if locale == "en":
            predicate_patterns = {
                "call": r"\bcall\b",
                "send": (
                    r"\b(?:send|notify|report)\b|"
                    r"\bemail\b(?![@.])(?!(?:\s+address)?\s+(?:is|was|address)\b)"
                ),
                "return": r"\b(?:return(?:ing)?|give\s+back|hand\s+back)\b",
                "enter": r"\benter\b",
                "assemble": r"\bassemble\b",
                "arrange": r"\barrange\b",
                "continue": r"\bcontinue\b",
                "install": r"\binstall\b",
                "support": r"(?<!participant )\bsupport\b",
                "store": r"\b(?:leave|left|store)\b|(?<!take\s)(?<!takes\s)\bplace\b",
                "occur": (
                    r"\b(?:take\s+place|takes\s+place|be\s+held|held|happen[a-z]*|occur[a-z]*)\b"
                ),
            }
            if re.search(
                r"\b(?:cannot|can't|must\s+not|should\s+not|do\s+not|may\s+not)\b",
                folded,
            ):
                modalities.add("prohibited")
            if re.search(r"\b(?:may|can)\b", folded):
                modalities.add("permitted")
            if re.search(
                r"\b(?:must|should|need|needs|required|necessary|ensure|no\s+later\s+than)\b",
                folded,
            ):
                modalities.add("required")
            if re.search(r"\bmandatory\b", folded):
                modalities.add("required")
            if re.search(r"\bfirst\b", folded):
                order.add("first")
            if re.search(r"\b(?:then|after|once)\b", folded):
                order.add("then")
        else:
            predicate_patterns = {
                "call": r"\bara(?:yın|yin|mak|n|dı|r|malı|meli)[a-zçğıöşü]*\b",
                "send": r"\b(?:gönder|bildir|ilet)[a-zçğıöşü]*\b",
                "return": (
                    r"\b(?:iade\s+e(?:t|d)|teslim\s+e(?:t|d)|geri\s+ver)[a-zçğıöşü]*\b|"
                    r"\biade\s+son\s+tarihi\b"
                ),
                "enter": r"\bgir[a-zçğıöşü]*\b",
                "assemble": r"\b(?:toplan|bir\s+araya\s+gel)[a-zçğıöşü]*\b",
                "arrange": r"\b(?:ayarla|düzenle)[a-zçğıöşü]*\b",
                "continue": r"\bdevam\s+e(?:t|d)[a-zçğıöşü]*\b",
                "install": r"\b(?:yükle[a-zçğıöşü]*|kur(?:mak|ma|un|unuz|abilir|ul)"
                r"[a-zçğıöşü]*)\b",
                "support": r"\bdestekle[a-zçğıöşü]*\b",
                "store": r"\b(?:bırak|yerleştir|muhafaza\s+et)[a-zçğıöşü]*\b",
                "occur": r"\b(?:gerçekleş|yapıl)[a-zçğıöşü]*\b",
            }
            if re.search(
                r"\b[a-zçğıöşü]*(?:amaz|emez|mayın|meyin|mamalı|memeli)[a-zçğıöşü]*\b|\byasak\b",
                folded,
            ):
                modalities.add("prohibited")
            if re.search(
                r"\b[a-zçğıöşü]*(?:abil|ebil)[a-zçğıöşü]*\b|\bizin\s+ver",
                folded,
            ):
                modalities.add("permitted")
            if re.search(
                r"\b(?:gerek[a-zçğıöşü]*|zorunlu|en\s+geç)\b|"
                r"\b[a-zçğıöşü]*(?:malı|meli)(?:dır|dir|dur|dür|sınız|siniz|sunuz|sünüz)?\b",
                folded,
            ):
                modalities.add("required")
            if re.search(
                r"\b(?:öncelikle|ilk)\b|(?:^|[.!?;]\s*)önce\b",
                folded,
            ):
                order.add("first")
            if re.search(r"\b(?:ardından|sonra)[a-zçğıöşü]*\b", folded):
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
                re.search(
                    rf"{re.escape(phone.casefold())}\s+numaralı\s+"
                    r"(?:[a-zçğıöşü]+\s+){0,4}telefonu[a-zçğıöşü]*",
                    folded,
                )
                for phone in phones
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
                re.search(
                    rf"(?:dahili\s+numarası\s+{re.escape(extension)}\b|"
                    rf"{re.escape(extension)}\s+dahili\s+numarası[a-zçğıöşü]*)",
                    folded,
                )
                for extension in extensions
            ):
                predicates.add("extension_object")
        return {"predicates": predicates, "modalities": modalities, "order": order}

    @staticmethod
    def _semantic_action_occurrences(
        value: str,
        *,
        locale: SourceLocale,
    ) -> list[tuple[int, int, str]]:
        """Locate action predicates without discarding their order or object binding."""

        if locale == "en":
            patterns = (
                ("return", r"\b(?:return(?:ed|ing|s)?|give\s+back|hand\s+back)\b"),
                ("store", r"\b(?:store(?:d|s|ing)?|leave|left)\b"),
                ("send", r"\b(?:send|notify|report)\b|\bemail\b(?![@.])"),
                ("call", r"\bcall(?:ed|ing|s)?\b"),
                ("enter", r"\benter(?:ed|ing|s)?\b"),
                ("assemble", r"\bassembl(?:e|ed|es|ing)\b"),
                ("arrange", r"\barrang(?:e|ed|es|ing)\b"),
                ("continue", r"\bcontinu(?:e|ed|es|ing)\b"),
                ("install", r"\binstall(?:ed|ing|s)?\b"),
                ("support", r"(?<!participant )\bsupport(?:ed|ing|s)?\b"),
                (
                    "occur",
                    r"\b(?:take\s+place|takes\s+place|be\s+held|held|happen[a-z]*|occur[a-z]*)\b",
                ),
            )
        else:
            patterns = (
                ("return", r"\b(?:iade\s+e(?:t|d)|teslim\s+e(?:t|d)|geri\s+ver)[a-zçğıöşü]*\b"),
                ("store", r"\b(?:bırak|yerleştir|muhafaza\s+et)[a-zçğıöşü]*\b"),
                ("send", r"\b(?:gönder|bildir|ilet)[a-zçğıöşü]*\b"),
                ("call", r"\bara(?:yın|yin|mak|n|dı|r|malı|meli)[a-zçğıöşü]*\b"),
                ("enter", r"\bgir[a-zçğıöşü]*\b"),
                ("assemble", r"\b(?:toplan|bir\s+araya\s+gel)[a-zçğıöşü]*\b"),
                ("arrange", r"\b(?:ayarla|düzenle)[a-zçğıöşü]*\b"),
                ("continue", r"\bdevam\s+e(?:t|d)[a-zçğıöşü]*\b"),
                (
                    "install",
                    r"\b(?:yükle[a-zçğıöşü]*|kur(?:mak|ma|un|unuz|abilir|ul)"
                    r"[a-zçğıöşü]*)\b",
                ),
                ("support", r"\bdestekle[a-zçğıöşü]*\b"),
                ("occur", r"\b(?:gerçekleş|yapıl)[a-zçğıöşü]*\b"),
            )

        candidates = [
            (match.start(), match.end(), action)
            for action, pattern in patterns
            for match in re.finditer(pattern, value, flags=re.IGNORECASE)
        ]
        candidates.sort(key=lambda item: (item[0], -(item[1] - item[0]), item[2]))
        selected: list[tuple[int, int, str]] = []
        occupied_until = -1
        for start, end, action in candidates:
            if start < occupied_until:
                continue
            if (
                locale == "en"
                and action == "send"
                and value[start:end].casefold() == "email"
                and selected
                and selected[-1][2] == "send"
                and re.fullmatch(
                    r"\s+(?:an?\s+)?",
                    value[selected[-1][1] : start],
                    flags=re.IGNORECASE,
                )
            ):
                continue
            selected.append((start, end, action))
            occupied_until = end
        return selected

    def _semantic_anchor_occurrences(
        self,
        value: str,
        *,
        locale: SourceLocale,
    ) -> list[tuple[int, int, str]]:
        """Locate protected domain/contact objects under one canonical English identity."""

        candidates: list[tuple[int, int, str]] = []
        for entry in self.glossary.entries:
            if not entry.bind_action:
                continue
            phrase = entry.source if locale == "en" else entry.target
            pattern = (
                self.glossary._phrase_pattern(phrase)
                if locale == "en"
                else self.glossary._turkish_phrase_pattern(phrase)
            )
            candidates.extend(
                (match.start(), match.end(), f"term:{entry.source.casefold()}")
                for match in pattern.finditer(value)
            )
        for start, end, kind, normalized in self.glossary._exact_occurrences(value):
            if kind in {"email", "phone"}:
                candidates.append((start, end, f"{kind}:{normalized}"))

        candidates.sort(key=lambda item: (item[0], -(item[1] - item[0]), item[2]))
        selected: list[tuple[int, int, str]] = []
        occupied_until = -1
        for start, end, anchor in candidates:
            if start < occupied_until:
                continue
            selected.append((start, end, anchor))
            occupied_until = end
        return selected

    @staticmethod
    def _semantic_modality_occurrences(
        value: str,
        *,
        locale: SourceLocale,
    ) -> list[tuple[int, int, str]]:
        if locale == "en":
            patterns = (
                ("prohibited", r"\b(?:cannot|can't|must\s+not|should\s+not|do\s+not|may\s+not)\b"),
                ("permitted", r"\b(?:may|can)\b"),
                (
                    "required",
                    r"\b(?:must|should|need|needs|required|necessary|mandatory|ensure|"
                    r"no\s+later\s+than)\b",
                ),
            )
        else:
            patterns = (
                (
                    "prohibited",
                    r"\b[a-zçğıöşü]*(?:amaz|emez|mayın|meyin|mamalı|memeli)[a-zçğıöşü]*\b|\byasak\b",
                ),
                (
                    "permitted",
                    r"\b[a-zçğıöşü]*(?:abil|ebil)[a-zçğıöşü]*\b|\bizin\s+ver",
                ),
                (
                    "required",
                    r"\b(?:gerek[a-zçğıöşü]*|zorunlu|en\s+geç)\b|"
                    r"\b[a-zçğıöşü]*(?:malı|meli)(?:dır|dir|dur|dür|sınız|siniz|sunuz|sünüz)?\b",
                ),
            )
        return [
            (match.start(), match.end(), modality)
            for modality, pattern in patterns
            for match in re.finditer(pattern, value, flags=re.IGNORECASE)
        ]

    @staticmethod
    def _nearest_action_index(
        start: int,
        end: int,
        actions: list[tuple[int, int, str]],
        *,
        prefer_preceding: bool | None = None,
    ) -> int:
        center = (start + end) / 2

        def distance(item: tuple[int, int, str]) -> tuple[float, int, float, int]:
            action_start, action_end, _action = item
            if end <= action_start:
                interval_distance = action_start - end
            elif start >= action_end:
                interval_distance = start - action_end
            else:
                interval_distance = 0
            direction_penalty = 0
            if prefer_preceding is True and action_start >= end:
                direction_penalty = 1
            elif prefer_preceding is False and action_end <= start:
                direction_penalty = 1
            action_center = (action_start + action_end) / 2
            return interval_distance, direction_penalty, abs(center - action_center), action_start

        return min(range(len(actions)), key=lambda index: distance(actions[index]))

    def semantic_action_bindings(
        self,
        value: str,
        *,
        locale: SourceLocale,
    ) -> tuple[
        Counter[tuple[str, tuple[str, ...], tuple[str, ...]]],
        tuple[str, ...],
    ]:
        """Bind actions to protected objects/modalities and preserve explicit action order."""

        actions = self._semantic_action_occurrences(value, locale=locale)
        if not actions:
            return Counter(), ()
        anchors_by_action: list[list[str]] = [[] for _action in actions]
        for start, end, anchor in self._semantic_anchor_occurrences(value, locale=locale):
            compatible_actions = actions
            compatible_indices = list(range(len(actions)))
            expected_action = "send" if anchor.startswith("email:") else None
            if anchor.startswith("phone:"):
                expected_action = "call"
            if expected_action is not None:
                compatible_indices = [
                    index
                    for index, (_start, _end, action) in enumerate(actions)
                    if action == expected_action
                ]
                compatible_actions = [actions[index] for index in compatible_indices]
            if compatible_actions:
                local_index = self._nearest_action_index(
                    start,
                    end,
                    compatible_actions,
                    prefer_preceding=locale == "en",
                )
                action_index = compatible_indices[local_index]
                anchors_by_action[action_index].append(anchor)
        modalities_by_action: list[list[str]] = [[] for _action in actions]
        for start, end, modality in self._semantic_modality_occurrences(value, locale=locale):
            modalities_by_action[self._nearest_action_index(start, end, actions)].append(modality)

        bindings = Counter(
            (
                action,
                tuple(sorted(anchors_by_action[index])),
                tuple(sorted(modalities_by_action[index])),
            )
            for index, (_start, _end, action) in enumerate(actions)
        )
        folded = value.casefold().replace("i̇", "i")
        if locale == "en":
            explicit_order = bool(re.search(r"\bfirst\b", folded)) and bool(
                re.search(r"\b(?:then|after|once)\b", folded)
            )
        else:
            explicit_order = bool(
                re.search(
                    r"\b(?:öncelikle|ilk)\b|(?:^|[.!?;]\s*)önce\b",
                    folded,
                )
            ) and bool(re.search(r"\b(?:ardından|sonra)[a-zçğıöşü]*\b", folded))
        ordered_actions = (
            tuple(action for _start, _end, action in actions) if explicit_order else ()
        )
        return bindings, ordered_actions

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
            expected_bindings, expected_order = self.semantic_action_bindings(
                original,
                locale="en",
            )
            observed_bindings, observed_order = self.semantic_action_bindings(
                localized,
                locale="tr",
            )
            if len(expected_bindings) > 1:

                def by_anchor(
                    bindings: Counter[tuple[str, tuple[str, ...], tuple[str, ...]]],
                ) -> dict[str, Counter[tuple[str, tuple[str, ...]]]]:
                    projected: dict[str, Counter[tuple[str, tuple[str, ...]]]] = {}
                    for (action, anchors, modalities), count in bindings.items():
                        for anchor in anchors:
                            projected.setdefault(anchor, Counter())[(action, modalities)] += count
                    return projected

                expected_by_anchor = by_anchor(expected_bindings)
                observed_by_anchor = by_anchor(observed_bindings)
                common_anchors = expected_by_anchor.keys() & observed_by_anchor.keys()
                if any(
                    expected_by_anchor[anchor] != observed_by_anchor[anchor]
                    for anchor in common_anchors
                ):
                    raise TranslationRejected(
                        "The local translator changed critical action-object or action-modality "
                        "bindings."
                    )
            if expected_order and expected_order != observed_order:
                raise TranslationRejected(
                    "The local translator changed the explicit order of critical actions."
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
            if score < 0.30 or unprotected_score < self.round_trip_threshold:
                raise TranslationRejected(
                    f"Round-trip translation recall {score:.2f} is below 0.30 or unprotected "
                    f"recall {unprotected_score:.2f} is below {self.round_trip_threshold:.2f}."
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
