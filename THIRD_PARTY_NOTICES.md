# Third-Party Notices

Tracefold source code is distributed under the repository's [MIT License](LICENSE). That license does not relicense third-party model checkpoints downloaded by the application.

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
