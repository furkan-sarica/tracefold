# Tracefold Privacy and Data Handling

## Plain-language promise

Tracefold is designed for local document question answering in English and Turkish. Uploaded files, extracted text, embeddings, retrieved passages, prompts, controlled intermediate translations, and generated answers stay on the user's machine during the normal workflow. Embedding and grounded claim generation use Foundry Local; Turkish↔English translation uses two locally cached Helsinki-NLP OPUS-MT checkpoints rather than a hosted model endpoint.

“Local” does not mean “automatically secure.” Tracefold is a single-user localhost demonstration, not a hardened multi-user document service. The person running it remains responsible for device security, operating-system accounts, backups, and the suitability of uploaded material.

## Data lifecycle

| Data | Why it exists | Where it lives | Retention |
|---|---|---|---|
| Uploaded source file | Re-indexing and citation context | Local data directory | Until the user deletes the document or data directory |
| Extracted chunks and metadata | Retrieval and citation display | Local SQLite database | Deleted with the owning document |
| Float32 embedding vectors | Cosine retrieval | Local SQLite BLOBs | Deleted with the owning document or on re-index |
| Ingestion job state | Progress and troubleshooting | Local SQLite database | Until local data is cleared |
| Evaluation questions, answers, evidence, metrics, and timings | Reproducible quality evidence | Local SQLite database; exported release runs may also be written to `docs/evaluation-report.json` | Until local data is cleared; an explicitly exported report remains until deleted from the project |
| Foundry and OPUS-MT model files | Local inference | Their respective on-device model caches | Until the operator removes those caches |
| Northstar glossary | Stable reviewed terminology for the release corpus | Checked-in project file | Until changed or removed from the project |
| Application logs | Operational diagnosis | Local process output or configured log file | Controlled by the local operator |

Turkish queries and answers may exist transiently in both languages while the controlled translation, reverse-translation, and validation stages run. Typed sentinels temporarily replace protected values such as dates, times, phone numbers, email addresses, and glossary-controlled terms; the pipeline rejects output when they cannot be restored exactly. Citation markers are detached before answer translation and reconstructed from the validated sentence map afterward. Normal intermediate translations are not written to a conversation history.

The default operational logging policy records event type, duration, model status, and safe identifiers. It should not record full document text, raw question bodies, translations, generated answers, sentinel mappings, or embedding arrays. Evaluation is the deliberate exception: its checked-in synthetic questions, generated answers, retrieved/cited source names, metrics, and code provenance are persisted and may be exported so a release can be audited. Do not run or export that evaluation with private material.

## Network behavior

Model acquisition may require network access during explicit setup: Foundry Local supplies the embedding/chat variants and Hugging Face supplies the two Helsinki-NLP OPUS-MT checkpoints. After those assets are cached, the accepted query path uses local files, SQLite, localhost HTTP, Foundry Local, and the on-device translation runtime. Neither query text nor document text is sent to Hugging Face. The application includes no cloud inference fallback, analytics beacon, web search, remote font, or CDN dependency.

An offline smoke test is part of release acceptance. A missing local model produces a visible setup error rather than transmitting data elsewhere or silently returning synthetic output.

## User controls

- The Library view lists indexed documents and allows a document and all of its chunks to be deleted.
- Clearing the local application data removes the SQLite database, uploaded copies, job history, and evaluation history. Shared Foundry/Hugging Face model caches are separate and must be removed explicitly if the operator also wants to reclaim model files.
- Changing embedding models requires re-indexing; Tracefold does not mix or export vectors across incompatible model spaces.
- Demo content and the checked-in bilingual evaluation questions are synthetic and intentionally safe to commit. Any document, database, screenshot, or evaluation export produced from private material must stay out of version control.

## Limitations and recommended use

- Bind the server to localhost. Do not expose it to a shared network without authentication, authorization, transport security, upload limits, and a security review.
- Do not ingest secrets, passwords, full medical records, identity scans, or data the operator is not authorized to process.
- Deleting Tracefold's local copy cannot delete copies created by operating-system backups or external tools.
- Citations improve inspectability but do not guarantee that a generated answer is correct; users should open the evidence for consequential decisions.
- Protected sentinels, citation reconstruction, reverse translation, and language gates reduce common translation and transcription errors, but they do not prove semantic equivalence in every domain.
- The checked-in glossary covers Tracefold and the synthetic Northstar corpus. It is extensible, but it is not a universal English–Turkish terminology dictionary. The 24-case release report validates this corpus only; new professional or specialist material needs reviewed glossary entries and a domain-specific bilingual evaluation.
- Tracefold is not an emergency, medical, legal, or real-time information service.

## Demo corpus

The checked-in Northstar Summer Lab documents are fictional. The `.example` email addresses, 555-series phone numbers, people, venues, and program details are synthetic and must not be treated as real-world instructions.

The OPUS-MT model weights retain their upstream CC BY 4.0 license and attribution. See [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).
