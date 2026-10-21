# Installation and execution

Run commands from the project root with Python 3.10–3.12. The base package requires NumPy 1.26.4 and PyTorch >=2.1,<2.9. Training adds SciPy; native IVF search adds Faiss. Detailed version and measurement settings are in [IMPLEMENTATION_DETAILS.md](IMPLEMENTATION_DETAILS.md).

```bash
python -m pip install -e ".[test,training]"
python -m fvsr.cli demo
python -m pytest tests/unit tests/parity -q -p no:cacheprovider
```

The demo and tests use artificial software fixtures. Install the `ivf` extra separately in a compatible Faiss CPU environment for native search:

```bash
python -m pip install -e ".[ivf]"
```

## Artifact inputs

Supply a case manifest and the corresponding artifacts. `CASE_MANIFEST.json`, `ARTIFACT_DIR` and other capitalized names below denote user-provided paths or values. They are not files supplied by this repository.

```bash
python scripts/verify_artifacts.py --manifest CASE_MANIFEST.json --artifact-dir ARTIFACT_DIR
python scripts/prepare_codec.py --manifest CASE_MANIFEST.json --artifact-dir ARTIFACT_DIR
```

Artifact references have `path`, `sha256` and `protected` fields. Paths resolve inside `ARTIFACT_DIR`; the loader checks the referenced file hashes. A serving case supplies `geometry`, `index` and `models`. The geometry NPZ contains `C`, `codes`, `rotation` and `residual`. The supported native route is non-residual, squared-L2 Faiss `IndexIVFPQ`, with 8-bit PQ codes. The index binding records its artifact hash and geometry identity; loading also checks all database IDs and their codes.

Learned-model entries and their binding certificates must match the checkpoint's identity, selected state, scaler, parent state and coefficient. The `--seed` value is supplied by the caller and must agree with that identity. No historical value is selected automatically. Model weights, trained codecs and indices are not distributed here.

## Native retrieval and rescoring

Query NPZ contains `queries` and unique `qids`. Attribute NPZ contains `update_date`, `category_labels` and `category_membership` in database-row order. Each request contains `qid`, `filter` and `request`: filter `R` uses integer `range_start`/`range_end`; `EMIS` uses `label`. Optional `legal_ids_sha256` verifies the resulting eligible IDs. These are the real-metadata predicates implemented by this entry point.

```bash
python scripts/evaluate.py --case CASE_MANIFEST.json --artifact-dir ARTIFACT_DIR --queries QUERIES.npz --attributes ATTRIBUTES.npz --requests REQUESTS.json --method full --seed USER_SEED --nprobe NPROBE --L POOL_SIZE --k TOP_K --output NEW_OUTPUT_DIR
```

`USER_SEED`, `NPROBE`, `POOL_SIZE` and `TOP_K` must be replaced by values for the supplied case. `L >= k >= 1` and `1 <= nprobe <= index.nlist` are required. `--method Native_L` ranks the L-candidate pool with native scores; `Native_direct_k` asks the backend for k candidates. Those two methods need no `--seed`.

The output directory must be new. It receives `PREDICTIONS.npz` (`ranked_ids`, `pool_ids`, padded with -1) and `REQUESTS.json` (query IDs, filter labels and legal/pool counts). Recall calculation additionally needs matching offline legal ground truth and the selected query-role/aggregation protocol. This entry point writes rankings rather than recall or timing summaries.

The default execution mode is `reference`. `--execution-mode full-cm` accepts the Full scorer for CPU global PQ with identity rotation and materializes its calibrated codebook once. An artificial check is available without external inputs:

```bash
python scripts/evaluate.py --demo --execution-mode full-cm
```

## Training

This command prints the requested recipe without loading data or starting an optimizer:

```bash
python scripts/train_reference.py --phase QC
```

Execution requires a caller-supplied random-state identity, a training manifest and a fresh output directory:

```bash
python scripts/train_reference.py --phase A --manifest TRAINING_MANIFEST.json --artifact-dir ARTIFACT_DIR --seed USER_SEED --output NEW_TRAINING_DIR --execute
```

The manifest supplies `dataset`, `backend_identity`, `evidence_role=FIT_SELECT_ONLY`, `role_schema=FIT0_SELECT_CAL1_SELECT_CHECK2`, and hashed `database`, `geometry` and `cache` references. The cache NPZ supplies:

- `role`, `qids`, `predicate`, `query`;
- candidate `ids`, boolean `valid`, native `base` and `exact` scores, legal ground-truth `gt`, and `legal_count`;
- unique, disjoint `db_learn_ids` and `db_cal_ids`;
- `labels100` for SCC positive/negative sampling.

FIT query IDs and SELECT query IDs must be disjoint. The schema assigns roles 0, 1 and 2 to FIT, SELECT_CAL and SELECT_CHECK. Role 2 is part of selection. `B` and `QC` additionally require a hashed selected SCC `parent`; `QC` and `NN_QUERY_MATCHED` require a hashed selected `initial` checkpoint. `NN_RECON` fits database reconstruction; `NN_QUERY_MATCHED` continues from the selected reconstruction checkpoint. The current CLI uses CPU training.

`--reference POOL_REF` or `FIT_REF` is available for `B`/`QC` with `--arm FINAL` and requires `full_scaler_checkpoint`. `--resume` requires a hashed recovery checkpoint in the manifest. The saved configuration must match; output goes to a new directory. Recovery files, per-epoch metrics, the selected state and result metadata are retained.

Exact objectives, initialization, sampling, optimizer settings and selection rules are in [IMPLEMENTATION_DETAILS.md](IMPLEMENTATION_DETAILS.md). That document also distinguishes the common trainer from the separately recorded W3 decoder protocol.

## File verification

```bash
python scripts/verify_release_files.py
```

This compares files with `FILE_MANIFEST.json`. After intentional source changes, maintainers regenerate that inventory using `scripts/refresh_manifest.py` before verifying it.

## Backend support

| Component | Included interface | Required inputs |
|---|---|---|
| Global-PQ SCC–PSRC | Python scoring and training implementations | Matching codec, codes and model states |
| OPQ and list-residual scoring | Coordinate-aware scoring implementations | Original transform, coarse centers and compatible candidate pools |
| Non-residual squared-L2 IVF-PQ | Native Faiss adapter | Matching Faiss index, codes, IDs, query and attribute files |
| HNSW-PQ, Rii, RangePQ, ACORN/RACORN-PQ | Recorded source/build information | Separate backend integration and compatible inputs |

The native adapter checks index type, metric, codebook and stored ID/code consistency. See [USAGE.md](USAGE.md) for the supported evaluation entry and [external backend versions](IMPLEMENTATION_DETAILS.md#external-backend-versions) for recorded external versions.

## Execution and verification

The demo and unit/parity tests check score identities, artifact binding and synthetic scoring behavior. Training dry runs display recipes and validate arguments. Environment and verification records are in [QUALIFIED_RUNTIMES.json](../environments/QUALIFIED_RUNTIMES.json) and [RELEASE_STATUS.json](../RELEASE_STATUS.json). These software checks do not measure real-data recall or latency.
