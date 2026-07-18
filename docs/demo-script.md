# Tracefold Live Demo Script

Target duration: 5 minutes. The demo assumes the two configured Foundry Local models, both OPUS-MT translation checkpoints, and the eight synthetic documents are already downloaded and indexed. Do not rely on network access or a fallback runtime.

Preflight: run `make models`, then `make demo-ready` once to rebuild the public corpus and persist a fresh 24-case real-model evaluation in the default local database. Start the app with `make demo`.

## 0:00–0:35 — Set the promise

Open **Ask** with the empty welcome state visible.

> “Tracefold turns a folder of documents into English and Turkish answers you can trace. Foundry Local handles retrieval embeddings and grounded English claims; dedicated OPUS-MT models handle controlled Turkish translation on the same machine. The documents, vectors, prompts, translations, and answers stay here, and every factual sentence must earn its citations.”

Briefly disconnect Wi-Fi or show that network access is disabled if this can be done without interrupting the presentation setup.

## 0:35–1:05 — Verify the local runtime

Open **Setup**.

Point out:

- Foundry Local reports ready.
- The exact `qwen3-embedding-0.6b` and `qwen2.5-7b` Foundry identifiers, device, and execution provider are visible.
- The `Helsinki-NLP/opus-mt-tc-big-tr-en` and `Helsinki-NLP/opus-mt-tc-big-en-tr` translation checkpoints are local and ready.
- The application has no cloud-provider configuration or API key.

> “Tracefold fails visibly if a required Foundry or translation model is missing. It never swaps in a hidden fake answer path or hosted translation service.”

## 1:05–1:40 — Show the library and ingestion evidence

Open **Library**. Show the eight Northstar Summer Lab documents, their indexed status, and the 53-chunk release index. If time allows, remove and re-add `03-travel-guide.md` to show ingestion progress and duplicate-safe indexing.

> “The parser preserves headings and source locations, then chunks each document, creates local embeddings, and stores text plus Float32 vectors in SQLite.”

## 1:40–2:30 — Ask an English multi-document question

Open **Ask** and enter:

```text
I will reach campus after 20:00 on arrival day. What must I do, and whom should I contact?
```

After generation and citation validation finish, the SSE response emits retrieval metadata and the validated answer in word-sized events:

1. Point out the cited Travel Guide and Accommodation Guide passages. They already contain the required phone and email, so the answer does not need to display a separate Contacts Directory citation.
2. Open a citation and show the exact supporting passage and similarity score.
3. Confirm the answer says to call the Residence Duty Phone before 19:30 and notify the Program Office; it should not imply that the final shuttle waits.

> “This is more than keyword search. The evaluation verifies that top-five retrieval covers the relevant travel, accommodation, and contacts sources. The answer itself displays only the passages it actually references.”

## 2:30–3:20 — Prove the Turkish path

Open preferences, switch the interface language to **Türkçe**, and ask the paired question:

```text
Varış günü kampüse saat 20:00'den sonra ulaşacağım. Ne yapmalıyım ve kiminle iletişime geçmeliyim?
```

Confirm that the response is natural Turkish, preserves `19:30`, `+90 312 555 0149`, and `program@northstar.example` exactly, and cites the same stored evidence. Open one citation and show its source passage.

> “Türkçe, yalnızca arayüz çevirisi değil. Soru önce koruma belirteçleriyle İngilizceye çevriliyor; Foundry Local kanıta dayalı İngilizce iddiaları üretiyor ve doğruluyor. Yalnızca doğrulanmış iddialar Türkçeye çevriliyor; atıflar yeniden yerleştiriliyor, ters çeviri ile kritik olgu ve dil kontrolleri tamamlanıyor.”

## 3:20–3:50 — Prove safe refusal

Ask:

```text
Northstar bursunun tutarı ne kadar?
```

Expected response:

> “Kütüphanenizde bu soruyu yanıtlamak için yeterli kanıt bulamadım.”

> “Güvenli ret, sabit bilgi getirme eşiğiyle model yanıtı üretilmeden önce belirleniyor. Belgelerde burs tutarı yok; Tracefold bu nedenle bir rakam uydurmuyor.”

## 3:50–4:35 — Show quality evidence

Open **Kalite** and display the latest accepted 24-case evaluation. Clearly separate acceptance targets from measured values.

Point out:

- Dataset `northstar-summer-lab-2026-bilingual-parity-v3`: 12 unique English/Turkish pairs, 24 cases total; each language has eight answerable and four unanswerable cases.
- Hit@5, expected-source coverage, and per-locale results.
- Citation validity, sentence-level citation coverage, protected-fact grounding, and post-translation citation reconstruction.
- Safe-refusal accuracy, bilingual paired-case accuracy, reverse-translation/language gates, and warm retrieval latency.
- The exact Foundry/OPUS-MT identities, retrieval/translation configuration, evaluated Git revision, and clean-tree provenance in `docs/evaluation-report.json`.

Never describe a target as achieved unless the current recorded run shows it. If the report's revision is not the intended release code commit or its dirty flag is true, treat the report as stale and rerun `make eval`.

## 4:35–5:00 — Close on the differentiator

Return to the cited answer.

> “A fluent answer is not enough for document Q&A. Tracefold makes the evidence inspectable, keeps the inference path local, validates both English and Turkish claims, and treats ‘I don't know’ as a feature. That gives us a small RAG system that is private, testable, and honest.”

## Presenter contingency

- If Setup is not `ready`, do not use mock results. Show the actionable model error and continue with screenshots from a previously recorded real-model run.
- If either OPUS-MT checkpoint or the translation controls are not ready, do not demo Turkish as accepted; show the visible failure and use evidence from the latest valid release run.
- If ingestion is slow, keep the pre-indexed database and narrate the pipeline rather than re-indexing all eight documents.
- If a response misses a required fact, returns an invalid citation, or the evaluation misses an expected retrieval source, call it a failed case; do not hand-wave or restate an unsupported answer.
- If the Turkish answer is awkward, changes a protected value, or leaks English interface copy, do not present it as accepted; record the failure and use only evidence from the latest valid run.
- Do not generalize Northstar parity results to an arbitrary domain. Explain that the checked-in glossary is extensible, then require reviewed domain terms and a new bilingual evaluation for another corpus.
