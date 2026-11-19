# Third-party code, models and data

The MIT licence in `LICENSE` applies to original MAPPER code. It does not
relicense external code, pretrained weights, publications or datasets.

`src/mapper/models/tabm_reference.py` is the vendored TabM implementation
from Yandex Research, copyright 2024 Yandex LLC, under Apache-2.0. Its source
notice is retained. The implementation is copied without changes; see
`licenses/TabM-Apache-2.0.txt` and <https://github.com/yandex-research/tabm>.

InfoAlign, BioBERT, BERT and PubChemDeBERTa are third-party pretrained models.
Generating their embeddings requires the respective software and model terms.
Cached embeddings do not require installing those encoders at inference time.

Processed external proteomics measurements and fixed STRING enrichment
artwork retain their source terms and attribution. Dataset publications and
accessions are recorded in the accompanying dataset metadata. No MIT claim is
made over those materials. Proprietary DIA-NN/Perseus applications are not
included in this source tree.
