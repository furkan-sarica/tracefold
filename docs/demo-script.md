# Tracefold Live Demo Script

Target duration: 4 minutes 30 seconds. The demo assumes Foundry Local models and the eight synthetic documents are already downloaded/indexed. Do not rely on network access or a fallback runtime.

Preflight: run `make demo-ready` once to rebuild the public corpus and persist a fresh real-model evaluation in the default local database, then start the app with `make demo`.

## 0:00–0:35 — Set the promise

Open **Ask** with the empty welcome state visible.

> “Tracefold turns a folder of documents into answers you can trace. Both embeddings and generation run through Foundry Local on this machine. The documents, vectors, prompts, and answers stay here, and every answer must earn its citations.”

Briefly disconnect Wi-Fi or show that network access is disabled if this can be done without interrupting the presentation setup.

## 0:35–1:05 — Verify the local runtime

Open **Setup**.

Point out:

- Foundry Local reports ready.
- The exact embedding and chat model identifiers are visible.
- The application has no cloud-provider configuration or API key.

> “Tracefold fails visibly if a required model is missing. It never swaps in a hidden fake answer path.”

## 1:05–1:40 — Show the library and ingestion evidence

Open **Library**. Show the eight Northstar Summer Lab documents, their indexed status, and the 53-chunk release index. If time allows, remove and re-add `03-travel-guide.md` to show ingestion progress and duplicate-safe indexing.

> “The parser preserves headings and source locations, then chunks each document, creates local embeddings, and stores text plus Float32 vectors in SQLite.”

## 1:40–2:35 — Ask a multi-document question

Open **Ask** and enter:

```text
I will reach campus after 20:00 on arrival day. What must I do, and whom should I contact?
```

After generation and citation validation finish, the SSE response emits the retrieval metadata and validated answer in word-sized events:

1. Point out the cited Travel Guide and Accommodation Guide passages. They already contain the required phone and email, so the answer does not need to display a separate Contacts Directory citation.
2. Open a citation and show the exact supporting passage and similarity score.
3. Confirm the answer says to call the Residence Duty Phone before 19:30 and notify the Program Office; it should not imply that the final shuttle waits.

> “This is more than keyword search. The evaluation verifies that top-five retrieval covers the relevant travel, accommodation, and contacts sources. The answer itself displays only the passages it actually references.”

## 2:35–3:10 — Prove abstention

Ask:

```text
How much is the Northstar scholarship stipend?
```

Expected response:

> “I couldn't find enough evidence in the indexed documents to answer that.”

> “The refusal is decided before generation by a fixed retrieval policy. The corpus never states a stipend, so Tracefold does not improvise one.”

## 3:10–3:55 — Show quality evidence

Open **Quality** and run or display the latest 12-case evaluation. Clearly label the acceptance targets and the measured values from the current run.

Point out:

- Eight answerable and four unanswerable cases.
- Retrieval Hit@5 and multi-source coverage.
- Citation validity and informational required-fact recall.
- Negative refusal accuracy and warm latency.

Never describe a target as achieved unless the current recorded run shows it.

## 3:55–4:30 — Close on the differentiator

Return to the cited answer.

> “A fluent answer is not enough for document Q&A. Tracefold makes the evidence inspectable, keeps the inference path local, and treats ‘I don't know’ as a feature. That gives us a small RAG system that is private, testable, and honest.”

## Presenter contingency

- If Setup is not `ready`, do not use mock results. Show the actionable model error and continue with screenshots from a previously recorded real-model run.
- If ingestion is slow, keep the pre-indexed database and narrate the pipeline rather than re-indexing all eight documents.
- If a response misses a required fact, returns an invalid citation, or the evaluation misses an expected retrieval source, call it a failed case; do not hand-wave or restate an unsupported answer.
