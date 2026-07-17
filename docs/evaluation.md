# Tracefold Evaluation

## Release result

Tracefold is evaluated as a grounded local RAG system, not as a general chatbot. The checked-in `eval-dataset.json` contains eight answerable and four deliberately unanswerable questions over eight synthetic Northstar Summer Lab documents.

The latest verified release run passed all 12 cases with the real Foundry Local runtime.

| Metric | Definition | Target | Measured |
| --- | --- | ---: | ---: |
| Case pass rate | Cases meeting retrieval, citation, refusal, and required-fact gates | 100% | **12/12 (100%)** |
| Retrieval Hit@5 | Answerable cases with at least one expected source in top five | ≥ 90% | **100%** |
| Expected-source recall@5 | Expected sources represented in retrieved evidence | 100% | **100%** |
| Citation validity | Answerable outputs containing only resolvable retrieved evidence IDs | 100% | **100%** |
| Citation source precision | Cited files included in the case's accepted source set | 100% | **100%** |
| Negative refusal accuracy | Unanswerable cases returning only the standard refusal | 100% | **100%** |
| Required-fact token recall | Supplemental automated recall with numeric-fact guard | Informational | **76.04%** |
| Warm retrieval p95 | Query embedding, SQLite vector load, cosine ranking, and top-k | < 500 ms | **95.56 ms** |
| Model warm-up | One-time load before timed retrieval samples | Informational | **17.14 s** |
| Raw model TTFT p95 | Model request to first unvalidated token | Not a release claim | **Not measured** |

The supplemental token-recall number is not used as a claim that 23.96% of facts are wrong. It is a conservative lexical metric that penalizes valid paraphrases. Every final answer was also reviewed against its cited passages; the recorded 12-case run contained no unsupported material fact.

Expected-source recall is computed over the top-five retrieved passages. The final citation list is intentionally narrower: it contains only evidence IDs the validated answer actually references, so a 100% three-source retrieval case does not imply three visible citation chips.

## Run record

```text
Run ID: eval_31ce4ce4275545a0a4b3526ab085b0f1
Dataset: northstar-summer-lab-2026-v1
Date: 17 July 2026, 14:52 Europe/Istanbul
Code state: codex/tracefold-rag working tree based on c0503a5
OS / hardware: macOS 26.5.2, Apple M1 Pro, 16 GB, arm64
Foundry execution provider: WebGpuExecutionProvider
Embedding: qwen3-embedding-0.6b-generic-gpu:1
Chat: qwen2.5-7b-instruct-generic-gpu:4
Documents / chunks: 8 / 53
Chunk size / overlap: 240 / 45
Top-k / per-document cap: 5 / 2
Abstention threshold / evidence floor: 0.60 / 0.30
Context character budget: 1800
```

The run was produced from the current local working tree, before a final Git commit; it is not presented as evidence for an unpublished commit hash. Re-run `make eval` after committing or changing any model, corpus, chunk, retrieval, prompt, or threshold configuration.

## Pass rules

An answerable case passes when:

- at least one accepted source appears in retrieved evidence;
- the model does not refuse;
- every displayed citation ID resolves to a retrieved passage;
- required-fact recall is at least `0.50` after a hard check that every expected numeric fact appears.

An unanswerable case passes only when the system returns exactly:

```text
I couldn't find enough evidence in the indexed documents to answer that.
```

and returns no citations.

The numeric guard catches failures that ordinary token overlap can miss—for example, an answer that contains the correct 17:00 and 17:30 times but invents an incorrect duration instead of the expected 30 minutes.

## Reproduce

```bash
make bootstrap
make models
make eval
```

`make eval` uses isolated `.tracefold-eval/` data, so it does not overwrite a user's normal Tracefold library. The evaluator warms the real runtime before timed retrieval samples and persists the exact model/configuration context with the case outputs.

Automated tests use the explicit deterministic fake adapter to verify application contracts without downloading models:

```bash
make test
```

Fake-adapter results are never combined with the measured Foundry Local report.

## Calibration method

The threshold is corpus/model/chunk specific. After switching to heading-isolated 240-token chunks, the release dataset showed a clear top-score separation: all answerable cases were at or above approximately `0.64`, while the highest negative case was approximately `0.57`. Tracefold freezes the release gate at `0.60`, preserving all answerable Hit@5 cases and all four negative refusals.

This threshold must be recalibrated after changing the embedding model, corpus characteristics, or chunk policy. It is not a universal semantic-similarity constant.

## Known measurement limits

- Citation validity proves that IDs resolve; it does not replace human review of whether a passage entails every phrase.
- Required-fact token recall is lexical and under-scores correct paraphrases.
- The current API validates a complete answer before emitting SSE tokens, so raw model TTFT is intentionally not reported as user-visible streaming latency.
- The dataset is small and presentation-focused. It is appropriate evidence for this assignment, not a production-domain benchmark.
- Offline behavior was verified with cached models and localhost-only application paths; model download itself requires an initial network connection.
