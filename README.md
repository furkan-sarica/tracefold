# Tracefold

[**English**](README.md) · [Türkçe](README.tr.md)

> **Answers you can trace.** A private, source-grounded document assistant with Microsoft Foundry Local at its RAG core.

![Tracefold — Answers you can trace](frontend/public/og.png)

Tracefold turns a small document library into inspectable English and Turkish answers. It parses and chunks files, creates embeddings on the device with Foundry Local `qwen3-embedding-0.6b`, stores Float32 vectors in SQLite, retrieves evidence with cosine similarity, and asks Foundry Local `qwen2.5-14b` to form source-grounded English claims. A controlled offline OPUS-MT layer translates Turkish queries into English and validated English claims into Turkish. If retrieval is too weak—or translation, protected facts, language, or citations fail validation—the application refuses instead of guessing.

The result is an assignment-complete local RAG system presented as a polished product: runtime setup, document library, evidence-first chat, citation drawer, deterministic safe refusal, and a reproducible quality dashboard.

## Product preview

![Tracefold answer with inspectable local evidence](docs/screenshots/tracefold-ask-evidence.png)

## Why Tracefold

- **Local by design:** documents, embeddings, prompts, intermediate translations, and answers stay on the machine during the normal query path.
- **Inspectable evidence:** every displayed citation opens the exact stored passage, section/page, chunk ID, and score.
- **Grounded failure:** each factual sentence needs a valid citation; unknown, missing, or out-of-range IDs are rejected, and insufficient retrieval returns a locale-specific safe refusal.
- **Controlled bilingual path:** retrieval keeps the original Turkish query alongside a protected OPUS-MT English translation; validated English claims return through a separate OPUS-MT model, citation reconstruction, and post-translation gates.
- **Honest evaluation:** 24 checked-in cases form 12 one-to-one English/Turkish pairs, with 16 answerable and eight unanswerable cases, so language parity is measured rather than inferred. The latest verified release run passed 24/24 cases with the real Foundry Local runtime.
- **Presentation ready:** responsive Tracefold branding, a five-minute talk track, architecture/evaluation documentation, and final deck assets.

## Assignment coverage

| Requested capability | Tracefold implementation |
| --- | --- |
| 5–10 source documents | 8 cohesive, synthetic Northstar Summer Lab documents |
| Local embeddings and chat | Foundry Local `qwen3-embedding-0.6b` retrieval + `qwen2.5-14b` grounded claim generation |
| Controlled Turkish path | Helsinki-NLP `opus-mt-tc-big-tr-en` + `opus-mt-tc-big-en-tr`, protected sentinels, extensible glossary, citation reconstruction, and reverse/protected-fact/language gates |
| Document chunking | Heading-aware 240-token chunks with 45-token overlap; sections never merge |
| Vector persistence | Little-endian Float32 embedding BLOBs in SQLite |
| Retrieval | Stable cosine top-k with a per-document diversity cap |
| Grounded answers | Evidence-only prompt, sentence-level citation validation, protected-fact checks, exact evidence drawer |
| Safe unknown handling | Calibrated pre-generation threshold plus deterministic English/Turkish refusal |
| Evaluation and testing | 24-case bilingual parity dataset, persisted and exported metrics, lint, unit, coverage, build, and browser E2E gates |
| Final presentation | Six-slide English and Turkish PPTX/PDF decks with matching speaker material |

## Requirements

- Python 3.11
- Node.js 20 or newer
- Microsoft Foundry Local supported on the host machine
- Enough free storage for the Foundry Local model variants selected for the host
- 16 GB unified/system memory recommended for the verified release setup
- Network access only for the first dependency/model download

The release evaluation records its host, execution provider, exact Foundry variants, translation model IDs, retrieval/translation configuration, and code provenance in [`docs/evaluation-report.json`](docs/evaluation-report.json). Foundry Local selects a compatible cached variant on other supported machines; set `TRACEFOLD_DEVICE=cpu` or `gpu` only when you need to override automatic selection.

## Quick start

```bash
git clone https://github.com/furkan-sarica/tracefold.git
cd tracefold

make bootstrap   # hash-locked Python/npm dependencies + Playwright Chromium
make models      # one-time Foundry + OPUS-MT model download and smoke test
make seed-reset  # index the 8 synthetic demo documents
make demo        # build UI and bind the complete app to localhost
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). After `make models`, both Foundry models and both OPUS-MT translation checkpoints are local, so the complete query path can run with the network disabled.

Python 3.11 installs use the checked-in, hash-verified `requirements-lock.txt` and `requirements-dev-lock.txt`; npm installs use `frontend/package-lock.json`. The human-maintained `requirements*.txt` files define supported ranges. Run `make lock` deliberately after changing those ranges to regenerate both cross-platform Python locks with `uv`, then review and test the resulting diff.

The `seed-reset` command intentionally replaces only Tracefold's local `data/tracefold.db` and `data/documents/` index. Use `make seed` to add any missing demo documents without resetting an existing index.

Before a judged presentation, run `make demo-ready` once instead of `make seed-reset`. It rebuilds the synthetic corpus and persists a fresh real-model evaluation in the default local database, so **Quality** opens with measured results. This command is intentionally destructive to the current Tracefold demo index and can take several minutes on a cold model load.

## Bilingual demo path

1. Open **Setup** and show the exact Foundry Local variants/device plus the ready status of both OPUS-MT checkpoints.
2. Open **Library** and confirm eight documents and 53 chunks are indexed locally.
3. In English, ask:

   ```text
   I will reach campus after 20:00 on arrival day. What must I do, and whom should I contact?
   ```

4. Open the citation drawer to inspect the Travel Guide and Accommodation Guide passages referenced by the validated answer. Those passages contain the required phone and email; the three-source coverage target applies to top-five retrieval, not to the final citation-chip count.
5. Switch the interface to Turkish and ask the paired question:

   ```text
   Varış günü kampüse saat 20:00'den sonra ulaşacağım. Ne yapmalıyım ve kiminle iletişime geçmeliyim?
   ```

   Confirm that the Turkish answer cites the same stored evidence and preserves the exact time, phone number, and email address. The controlled path masks those values before translation, validates the grounded English claim, reconstructs its citations after Turkish translation, then runs reverse-translation, protected-fact, and language gates.
6. Ask the Turkish negative case:

   ```text
   Northstar bursunun tutarı ne kadar?
   ```

7. Open **Quality** and inspect the checked-in 24-case evaluation, including the 12/12 language split and paired-case accuracy.

Expected refusal:

```text
Kütüphanenizde bu soruyu yanıtlamak için yeterli kanıt bulamadım.
```

## Commands

| Command | Purpose |
| --- | --- |
| `make bootstrap` | Create `.venv`, install hash-locked Python/npm dependencies, and download Playwright Chromium |
| `make lock` | Regenerate the Python 3.11 cross-platform, hash-verified runtime and development locks with `uv` |
| `make frontend-browser` | Download the Chromium build pinned by the Playwright lock |
| `make models` | Cache the two Foundry models and two OPUS-MT translation models, then run local smoke checks |
| `make build` | Type-check and create the production frontend bundle |
| `make seed` | Add missing synthetic demo documents to the local index |
| `make seed-reset` | Rebuild the local demo index from scratch |
| `make demo-ready` | Rebuild demo data, run the 24-case real-model evaluation, persist it locally, and export `docs/evaluation-report.json` |
| `make demo` | Serve the complete same-origin app at `127.0.0.1:8000` |
| `make api` / `make web` | Run backend and Vite separately for development |
| `make lint` | Run Ruff and ESLint with warnings treated as failures |
| `make test` | Run deterministic backend coverage tests, frontend unit tests, and the production build |
| `make test-e2e` | Run Playwright browser tests against the built product flow |
| `make release-check` | Audit the locked Python development and complete npm dependency sets, then run lint, unit/coverage, production build, and browser E2E gates |
| `make eval` | Rebuild isolated `.tracefold-eval/` data, run the 24-case real-model evaluation, and export `docs/evaluation-report.json` |
| `make release-checksums` | Validate the final report and EN/TR decks, then atomically write `release/SHA256SUMS` |
| `make release-artifacts` | Verify the clean Git provenance, report approval, deck structure, PDFs, and every release checksum |
| `make release-ready` | Run both the deterministic release gates and the final read-only artifact verification |

The deterministic `TRACEFOLD_RUNTIME=fake` adapter is available only for automated tests. The user-facing runtime defaults to Foundry Local and exposes failures; there is no silent production fallback.

GitHub Actions repeats the Python and frontend quality gates, development-dependency security audits, and final release-artifact integrity check on pushes to `main` and on pull requests. `make release-check` intentionally excludes the artifact check so its deterministic code gates remain usable before the report and decks exist; `make release-ready` is the complete final gate. The real-model evaluation stays a deliberate local release step because it requires Foundry Local and downloaded model assets. Its JSON report records the evaluated Git revision, whether that tree was dirty, dataset ID `northstar-summer-lab-2026-bilingual-parity-v3`, exact Foundry and OPUS-MT identities, platform, retrieval/translation configuration, document/chunk counts, per-locale metrics, and case-level evidence. A release report is accepted only when it names the intended code commit and records `dirty: false`.

For a release, first commit the behavior candidate and run `make eval` from that clean revision. Review every printed Turkish case before recording approval with `scripts/review_evaluation.py --report-json docs/evaluation-report.json --data-dir .tracefold-eval --reviewer "<reviewer>" --approve`. After the final EN/TR PPTX and PDF files are exported, run `make release-checksums`, commit only the resulting evidence/assets, and run `make release-ready` from the clean final commit. The artifact gate requires the evaluated behavior revision to be an ancestor of that commit and rejects intervening runtime-code changes.

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Service and local index counts |
| `GET` | `/api/runtime` | Exact Foundry and OPUS-MT model, device/cache/load, execution-provider, and glossary state |
| `POST` | `/api/documents` | Validate, parse, chunk, embed, and persist one document |
| `GET` | `/api/documents` | List local sources and chunk counts |
| `DELETE` | `/api/documents/{id}` | Delete one document and its chunks |
| `DELETE` | `/api/documents` | Clear the local library |
| `GET` | `/api/jobs/{id}` | Read the persisted ingestion outcome |
| `POST` | `/api/chat` | Return a validated grounded answer as JSON |
| `POST` | `/api/chat/stream` | Return retrieval metadata, then post-validation answer-token and citation SSE events |
| `POST` | `/api/evaluations/run` | Run the built-in or a supplied evaluation set |
| `GET` | `/api/evaluations/latest` | Read the latest persisted quality report |

Document ingestion is intentionally synchronous in this single-user demonstration. A completed/failed job record is still persisted so the UI and API expose durable progress and errors without claiming a background queue.

## Architecture

```text
React + TypeScript UI (same-origin localhost)
            │
         FastAPI
            │
   ┌────────┼───────────┐
   │        │           │
Parser   RAG policy   Evaluator
   │        │           │
   └── SQLite text + Float32 vectors
            │
   Foundry Local RAG core
 qwen3 embedding + qwen2.5 chat
            │
 local OPUS-MT TR↔EN control layer
```

See [docs/architecture.md](docs/architecture.md), [docs/privacy.md](docs/privacy.md), and [docs/evaluation.md](docs/evaluation.md) for implementation boundaries and acceptance evidence.

## Project map

```text
backend/           FastAPI, Foundry/OPUS runtimes, parsing, RAG, translation controls, SQLite, evaluation
frontend/          React/TypeScript/Vite Tracefold product UI
demo_docs/         8 synthetic documents safe for public demos
docs/              architecture, evaluation, privacy, demo and talk track
scripts/           model preparation and reproducible demo seeding
tests/             deterministic unit and API contract tests
presentation/      final PPTX/PDF deliverables
```

## Privacy and limitations

Tracefold binds to `127.0.0.1` and does not configure cloud inference, telemetry, web search, authentication, multi-user isolation, OCR, or persistent conversation memory. Image-only PDFs are rejected with an OCR hint. The local SQLite index is tied operationally to the configured embedding model; rebuild the index after changing that model. Treat document content as untrusted—the prompt instructs the model to ignore embedded instructions, but this demo is not a complete adversarial prompt-injection sandbox. Grounded citations improve inspectability; they do not turn the application into a medical, legal, emergency, or real-time information service.

The OPUS-MT checkpoints provide general English–Turkish translation, while Tracefold's checked-in glossary only controls selected product and Northstar corpus terminology. It is intentionally extensible, not a claim of universal domain coverage. The accepted 24-case report validates the synthetic release corpus; a new domain needs its own reviewed glossary entries and bilingual evaluation before equivalent quality can be claimed.

All checked-in Northstar content is fictional. Do not commit private source documents or the generated local database.

## License

Tracefold source code is [MIT licensed](LICENSE). Foundry/Qwen and OPUS-MT assets retain their upstream terms and attribution; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
