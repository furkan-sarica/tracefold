# Tracefold Privacy and Data Handling

## Plain-language promise

Tracefold is designed for local document question answering. Uploaded files, extracted text, embeddings, retrieved passages, prompts, and generated answers stay on the user's machine during the normal workflow. Embedding and chat inference use Foundry Local rather than a hosted model endpoint.

“Local” does not mean “automatically secure.” Tracefold is a single-user localhost demonstration, not a hardened multi-user document service. The person running it remains responsible for device security, operating-system accounts, backups, and the suitability of uploaded material.

## Data lifecycle

| Data | Why it exists | Where it lives | Retention |
|---|---|---|---|
| Uploaded source file | Re-indexing and citation context | Local data directory | Until the user deletes the document or data directory |
| Extracted chunks and metadata | Retrieval and citation display | Local SQLite database | Deleted with the owning document |
| Float32 embedding vectors | Cosine retrieval | Local SQLite BLOBs | Deleted with the owning document or on re-index |
| Ingestion job state | Progress and troubleshooting | Local SQLite database | Until local data is cleared |
| Evaluation outcomes and timings | Reproducible quality evidence | Local SQLite database | Until local data is cleared |
| Application logs | Operational diagnosis | Local process output or configured log file | Controlled by the local operator |

The default logging policy records event type, duration, model status, and safe identifiers. It should not record full document text, raw question bodies, generated answers, or embedding arrays.

## Network behavior

Model acquisition may require network access during explicit setup. After the required assets are downloaded, the accepted query path uses local files, SQLite, localhost HTTP, and Foundry Local. The application includes no cloud inference fallback, analytics beacon, web search, remote font, or CDN dependency.

An offline smoke test is part of release acceptance. A missing local model produces a visible setup error rather than transmitting data elsewhere or silently returning synthetic output.

## User controls

- The Library view lists indexed documents and allows a document and all of its chunks to be deleted.
- Clearing the local application data removes the SQLite database, uploaded copies, job history, and evaluation history.
- Changing embedding models requires re-indexing; Tracefold does not mix or export vectors across incompatible model spaces.
- Demo content is synthetic and intentionally safe to commit. Private user documents must stay out of version control.

## Limitations and recommended use

- Bind the server to localhost. Do not expose it to a shared network without authentication, authorization, transport security, upload limits, and a security review.
- Do not ingest secrets, passwords, full medical records, identity scans, or data the operator is not authorized to process.
- Deleting Tracefold's local copy cannot delete copies created by operating-system backups or external tools.
- Citations improve inspectability but do not guarantee that a generated answer is correct; users should open the evidence for consequential decisions.
- Tracefold is not an emergency, medical, legal, or real-time information service.

## Demo corpus

The checked-in Northstar Summer Lab documents are fictional. The `.example` email addresses, 555-series phone numbers, people, venues, and program details are synthetic and must not be treated as real-world instructions.
