# SCC–PSRC

Research implementation of fixed-code candidate rescoring for filtered vector search. SCC calibrates a shared scoring codebook; PSRC predicts candidate-level corrections from codeword identities, query statistics and pool-standardized subspace scores. Stored code assignments and candidate generation remain fixed.

## Install and check

Use Python 3.10–3.12 in a separate virtual environment. For CPU software checks:

```sh
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install ".[test,training]"
python -m fvsr.cli demo
python -m pytest tests/unit tests/parity -q -p no:cacheprovider
```

These commands use synthetic inputs for software checks. Benchmark execution requires compatible datasets, trained states, codecs, indices and case manifests, which are not included. Input formats and commands are in [USAGE.md](docs/USAGE.md).

## Documentation

| Document | Contents |
|---|---|
| [Usage](docs/USAGE.md) | Inputs, training/evaluation commands, backend support and verification |
| [Implementation details](docs/IMPLEMENTATION_DETAILS.md) | Equations, exact training recipes, selection rules, data roles, timing and external backend versions |
| [Datasets](docs/DATASETS.md) | Dataset descriptions, preprocessing and provider links |
| [Saved results](paper_results/README.md) | Recorded numerical views, units and interpretation |

## Source layout

Open this directory in PyCharm or another Python editor. Package metadata and dependency constraints are in [pyproject.toml](pyproject.toml).

| Source | Purpose |
|---|---|
| [parent_bound.py](src/fvsr/core/parent_bound.py) | SCC codebook calibration and lookup scores |
| [r4_model_bound.py](src/fvsr/core/r4_model_bound.py) | PSRC statistics, features and residual head |
| [trainer_v2.py](src/fvsr/core/trainer_v2.py), [selection.py](src/fvsr/core/selection.py) | Training, checkpoint and coefficient selection |
| [model.py](src/fvsr/model.py), [controls.py](src/fvsr/controls.py) | Model loading, identity checks and comparison methods |
| [backends.py](src/fvsr/backends.py) | Non-residual squared-L2 Faiss IVF-PQ adapter |
| [scripts/](scripts/) | Data/artifact checks and training/evaluation entry points |
| [tests/](tests/), [frozen_reference/](frozen_reference/) | Software tests and retained numerical reference modules |
| [configs/](configs/), [manifests/](manifests/), [environments/](environments/) | Recipes, input/source identities and environment records |

## Project status and licensing

Repository: [SCC-PSRC](https://github.com/tothedocksandgetsomefries/SCC-PSRC), public. Python package: `fvsr-frozen-reference`, version `0.1.2rc3`. Associated manuscript: *Fixed-code candidate rescoring for filtered vector search*.

A software license for the project's original code has not yet been selected. Dependencies, external backends and datasets remain subject to their own terms. Their source and binaries are not vendored here; dataset links are in [DATASETS.md](docs/DATASETS.md). Dataset, model and index assets are not bundled.
