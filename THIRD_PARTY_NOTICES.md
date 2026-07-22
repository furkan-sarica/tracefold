# Third-Party Notices

Tracefold source code is distributed under the repository's [MIT License](LICENSE). That license does not relicense third-party model checkpoints downloaded by the application.

## Microsoft Foundry Local and Qwen models

Tracefold uses the MIT-licensed [Microsoft Foundry Local SDK](https://github.com/microsoft/Foundry-Local) to discover, download, and run compatible on-device variants of these models:

| Tracefold alias | Upstream model | Foundry catalog license | License terms |
| --- | --- | --- | --- |
| `qwen3-embedding-0.6b` | [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | Apache License 2.0 | [Upstream license](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/blob/main/LICENSE) |
| `qwen2.5-14b` | [Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) | Apache License 2.0 | [Upstream license](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct/blob/main/LICENSE) |

The aliases resolve to a hardware-compatible Foundry variant at setup time. The exact variant ID, device, and execution provider used for a release evaluation are recorded in `docs/evaluation-report.json` rather than represented as a portable model binary in this repository.

Tracefold does not modify or redistribute Foundry or Qwen model weights. `make models` explicitly downloads them to Foundry Local's on-device cache. The upstream model terms and the Foundry catalog entry are authoritative if this notice differs from them; use of these models does not imply endorsement by Microsoft, Qwen, Alibaba Cloud, or the upstream contributors.

## Helsinki-NLP OPUS-MT translation models

Tracefold's controlled offline Turkish path uses these unmodified upstream checkpoints:

| Model | Direction | Publisher | Upstream release | Source | Declared license |
| --- | --- | --- | --- | --- | --- |
| `Helsinki-NLP/opus-mt-tc-big-tr-en` | Turkish to English | Helsinki-NLP / OPUS-MT | 17 March 2022 | [Model card](https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-tr-en) | [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/) (`CC BY 4.0`) |
| `Helsinki-NLP/opus-mt-tc-big-en-tr` | English to Turkish | Helsinki-NLP / OPUS-MT | 25 February 2022 | [Model card](https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-en-tr) | [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/) (`CC BY 4.0`) |

The checkpoints are part of the [OPUS-MT project](https://github.com/Helsinki-NLP/OPUS-MT), were originally trained with [Marian NMT](https://marian-nmt.github.io/), use OPUS training data, and were converted for the Transformers ecosystem as described by their model cards.

Tracefold does not modify or redistribute the model weights in this repository. Explicit setup downloads the checkpoints to an on-device cache. Tracefold adds its own input/output controls around the models—protected-value sentinels, an extensible Northstar glossary, citation reconstruction, reverse translation, and validation gates. Those controls are part of Tracefold and are not features claimed by the upstream checkpoints.

Please retain this attribution, the model names, the source links, and the CC BY 4.0 license link when distributing a package that includes or automatically acquires these checkpoints. Use of the models does not imply endorsement of Tracefold by Helsinki-NLP, OPUS-MT, the University of Helsinki, Hugging Face, or the upstream authors.

Requested academic citations from the upstream model cards:

- Jörg Tiedemann and Santhosh Thottingal. [OPUS-MT — Building open translation services for the World](https://aclanthology.org/2020.eamt-1.61/). Proceedings of the 22nd Annual Conference of the European Association for Machine Translation, 2020.
- Jörg Tiedemann. [The Tatoeba Translation Challenge — Realistic Data Sets for Low Resource and Multilingual MT](https://aclanthology.org/2020.wmt-1.139/). Proceedings of the Fifth Conference on Machine Translation, 2020.

The upstream model cards and license are authoritative if this summary differs from them.
