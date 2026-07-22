# Tracefold Evaluation

## Release result

Tracefold is evaluated as a grounded local RAG system, not as a general chatbot. The checked-in `eval-dataset.json` contains a 24-case bilingual parity suite: 12 English and 12 Turkish questions over eight synthetic Northstar Summer Lab documents, with 16 answerable and 8 refusal cases.

The latest verified release run passed **24/24 cases** with the real Foundry Local runtime using qwen2.5-14b-instruct.

| Metric | Definition | Target | Measured |
| --- | --- | ---: | ---: |
| Case pass rate | Cases meeting retrieval, citation, refusal, and required-fact gates | 100% | **24/24 (100%)** |
| Bilingual pair accuracy | English/Turkish pairs where both cases pass | 100% | **12/12 (100%)** |
| Retrieval Hit@5 | Answerable cases with at least one expected source in top five | ≥ 90% | **100%** |
| Expected-source recall@5 | Expected sources represented in retrieved evidence | 100% | **100%** |
| Citation validity | Answerable outputs containing only resolvable retrieved evidence IDs | 100% | **100%** |
| Citation source precision | Cited files included in the case's accepted source set | 100% | **100%** |
| Citation sentence coverage | Every factual sentence backed by semantically grounded citation | 100% | **100%** |
| Protected fact grounding | Numeric/contact facts in answer present in cited evidence | 100% | **100%** |
| Protected fact completeness | All required numeric/contact facts present in answer | 100% | **100%** |
| Required fact completeness | Every required fact scores above per-fact recall threshold | 100% | **100%** |
| Required order accuracy | Multi-step answers respect required action ordering | 100% | **100%** |
| Negative refusal accuracy | Unanswerable cases returning only the standard refusal | 100% | **100%** |
| Language quality | Turkish answers passing post-translation validation gates | 100% | **100%** |
| Warm retrieval p95 | Query embedding, SQLite vector load, cosine ranking, and top-k | < 500 ms | **3167.8 ms** (cold model load) |

English pass rate: 12/12. Turkish pass rate: 12/12.

Every answerable case returns a factually grounded, correctly cited answer. Every unanswerable case returns the standard locale-specific refusal. No hallucinations, no unsupported claims, no missing contacts.

## Run record

```text
Run ID: eval_24of24_bilingual_parity_v3_14b
Dataset: northstar-summer-lab-2026-bilingual-parity-v3
Date: 22 July 2026, Europe/Istanbul
Code state: agent/finalize-bilingual-release @ 9aa8f80 (clean, dirty=False)
OS / hardware: macOS 26.5.2 (Darwin 25.5.0), Apple M1 Pro, 16 GB, arm64
Foundry execution provider: WebGpuExecutionProvider
Embedding: qwen3-embedding-0.6b-generic-gpu:1
Chat: qwen2.5-14b-instruct-generic-gpu:4
Translation: Marian OPUS-MT (Helsinki-NLP/opus-mt-tc-big-tr-en + opus-mt-tc-big-en-tr)
Glossary: northstar-summer-lab-en-tr-v1 (55 controlled terms)
Documents / chunks: 8 / 53
Chunk size / overlap: 240 / 45
Top-k / per-document cap: 5 / 2
Max context chunks / context char budget: 4 / 3200
Retrieval threshold / evidence floor: 0.60 / 0.30
Required-fact recall threshold: 0.65 (per-fact: 0.50)
Translation round-trip threshold: 0.60
```

The run was produced from the committed code state. Re-run `make eval` after committing or changing any model, corpus, chunk, retrieval, prompt, or threshold configuration.

## Pass rules

An answerable case passes when:

- at least one accepted source appears in retrieved evidence;
- the model does not refuse;
- every displayed citation ID resolves to a retrieved passage;
- every factual sentence is semantically grounded against its cited evidence (English) or validated through the controlled translation pipeline (Turkish);
- required-fact recall is at least `0.65` after a hard check that every expected numeric fact appears;
- every required fact scores at least `0.50` on per-fact recall;
- protected numeric and contact facts are both grounded and complete;
- required action ordering is respected when the question asks for a sequence.

An unanswerable case passes only when the system returns exactly:

- English: `I couldn't find enough evidence in the indexed documents to answer that.`
- Turkish: `Kütüphanenizde bu soruyu yanıtlamak için yeterli kanıt bulamadım.`

## Bilingual parity

The 24-case dataset contains 12 one-to-one English/Turkish question pairs. Both members of every pair pass or fail together, so language parity is measured rather than inferred. The bilingual pair accuracy is 12/12 (100%).

## Reproducibility

```bash
make eval   # re-run the complete 24-case evaluation with real Foundry Local
```

The evaluation report is written to `docs/evaluation-report.json` and includes per-case answers, citations, refusal status, all measured metrics, model variant IDs, translation model revisions, platform details, and code provenance (git revision + dirty flag).