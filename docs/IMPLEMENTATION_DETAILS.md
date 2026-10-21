# Training, selection and timing

This document describes the implemented SCC–PSRC equations, training stages and recorded measurement settings. [README source layout](../README.md#source-layout) locates the corresponding functions; [USAGE.md](USAGE.md) describes the input files and commands.

## Scoring and initialization

For global PQ, the stored codebook is `C[M,256,d]`, with `D=M*d`. SCC learns a rank-eight adjustment while keeping the database codes fixed:

```text
b_m = 0.10 * max(sqrt(mean(C_m**2)), 1e-12)
alpha_mr = tanh(g_mr)
Delta_mkd = b_m * tanh(sum_r(alpha_mr * W_mrkd) / b_m)
Ccal = C + Delta
U_im = 2 * dot(q_m, Ccal[m,z_im]) - ||Ccal[m,z_im]||²
s_c,i = sum_m U_im
```

`g` starts at zero and `W` uses Xavier-uniform initialization, so SCC initially uses the original codebook. Scores omit the query-only squared norm. The implementation retains an explicit left-to-right FP32 subspace sum in `Parent.score_fast` ([parent_bound.py](../src/fvsr/core/parent_bound.py)).

PSRC uses eight-dimensional embeddings for each subspace/codeword. Its input concatenates those embeddings, current-pool standardized `U`, and standardized query RMS for each subspace. The MLP has widths `M*(8+2) → 32 → 32 → 1`, with ReLU after the first two linear layers. Embeddings are initialized from `Normal(0,0.02)`; hidden linear layers use the PyTorch default initialization; the last layer's weights and bias start at zero ([MatchedMLP](../src/fvsr/core/r4_model_bound.py#L639)).

All pool means and standard deviations use the valid candidates and population variance, with standard-deviation floor `epsilon=1e-6`. Query RMS is the square root of the mean squared query coordinates in each subspace. Its fixed population mean and standard deviation are fitted on FIT records with equal record weights, including repeated query records. The centered residual and final score are:

```text
t_i = Z_pool(s_exact)_i - Z_pool(s_c)_i
r_i = h_theta(phi_i) - mean_valid(h_theta(phi))
s_i = s_c,i + gamma * std_pool(s_c) * r_i
```

`gamma=0` returns the parent scores directly. [r4_model_bound.py](../src/fvsr/core/r4_model_bound.py) implements pool statistics, query-RMS scaling, targets, centering and score composition. Transformations use row vectors `q @ rotation.T`; the identity rotation gives ordinary PQ.

## Training objectives and sampling

An operation is a query–predicate pair. FIT supplies updates; SELECT is the union of SELECT_CAL and SELECT_CHECK. The dataset role schema is detailed in [data roles](IMPLEMENTATION_DETAILS.md#data-roles-and-statistical-units).

SCC (`A`) uses operations with at least one exact legal top-100 target and one non-target in the valid candidate pool. Each update samples 64 positives from reachable top-100 targets and 64 negatives from up to 256 non-target candidates with the highest native scores. Sampling uses replacement when the available set is smaller than the requested count. In the current trainer, ineligible W3 operations are recorded and excluded; an ineligible operation in another dataset raises an error.

For an operation, define `span=Q90(exact)-Q10(exact)` over valid candidates. With positive set `P+` and negative set `P-`:

```text
a_o = max((min(exact[P+]) - max(exact[P-])) / span, 0)
tau = median(positive finite a_o), or 1 if there are none
w_o = clip(1 + 3*exp(-a_o / max(tau,1e-6)), 1, 4)
L_SCC = mean_o[w_o * mean_(64×64 pairs) softplus(1-s_pos+s_neg)]
        + 0.01 * mean_(sampled scores)(Z_sample(s_c)-Z_sample(s_native))²
```

A span at most `1e-12`, or non-finite separation, receives weight 4. The anchor uses population standard deviation floored at `1e-6`. Sampling and weights are in [training_primitives.py](../src/fvsr/core/training_primitives.py); the loss is in [trainer_v2.py](../src/fvsr/core/trainer_v2.py#L204).

For initial PSRC (`B`) and continuation (`QC`), the selected SCC parameters remain frozen. Each operation's loss is the mean SmoothL1 loss (`beta=1`) between the centered residual and `t`, over its valid candidates. Empty pools have zero operation loss. The outer weights balance query IDs, predicates within each query and repeated records:

```text
omega_o = N_operations / (N_queries * N_predicates_for_query * N_records_for_query_predicate)
L_PSRC = mean_o[omega_o * mean_valid(SmoothL1_beta1(r_i,t_i))]
```

FIT fixes the query-RMS scaler. `QC` loads the selected `B` checkpoint, including its scaler state, and starts with that checkpoint's selected coefficient. [SharedTrainer](../src/fvsr/core/trainer_v2.py#L84) defines the actual update order and cached targets.

## Optimizer recipes

The values below agree with [original_recipes.json](../configs/training/original_recipes.json) and [`RECIPES`](../src/fvsr/core/training_reference.py#L7).

| Stage / key | Optimizer | Learning rate | Weight decay | Batch | Minimum / maximum epochs | Patience | Gradient-norm clip |
|---|---|---:|---:|---:|---:|---:|---:|
| SCC / `A` | AdamW | 1e-3 | 0 | 16 operations | 2 / 30 | 5 | 1 |
| Initial PSRC / `B` | AdamW | 1e-3 | 1e-4 | 16 operations | 10 / 30 | 8 | 1 |
| Continuation / `QC` | AdamW | 1e-4 | 1e-4 | 16 operations | 4 / 12 | 4 | 1 |
| Neural reconstruction / `NN_RECON` | Adam | 5e-4 | 0 | 256 database vectors | 1 / 300 | 300 | none |
| Query-matched decoder / `NN_QUERY_MATCHED` | AdamW | 1e-4 | 1e-4 | 16 operations | 4 / 12 | 4 | 1 |

These are epoch limits and early-stopping patience, not realized run lengths. `NN_RECON` additionally uses `ReduceLROnPlateau` on database-calibration MSE, factor 0.5 and patience 10. Other stages have no learning-rate scheduler in this trainer. Optimizer fields not listed use the installed PyTorch defaults. CPU training uses FP32, one PyTorch thread, deterministic algorithms, and disables TF32 and mixed precision.

`TECHNICAL_QUALIFICATION` is a separate software-fixture mode: the trainer substitutes two epochs and minimum/patience two, and can cap updates. The command-line `--execute` route selects `FORMAL` recipes.

## Checkpoint and coefficient selection

Selection computes exact legal Recall@100 and Recall@10 with denominator `min(k, legal_count)` for each operation. It averages repeated records within each query/predicate, then predicates within each query, then query IDs. Operations with no legal objects have no estimable recall. Score ties use ascending database IDs ([metric_summary](../src/fvsr/core/trainer_v2.py#L60), [top](../src/fvsr/core/replay_bound.py)).

The coefficient grid is `{0, 0.25, 0.5, 0.75, 1}`. All stages evaluate epoch 0, accept strict improvements only and retain the earlier checkpoint on ties.

1. SCC selects the checkpoint with maximum SELECT R100.
2. Initial PSRC selects its checkpoint with maximum SELECT R100 at `gamma=1`. It then evaluates the coefficient grid. Eligible coefficients allow R100 to decrease by at most 0.001 and R10 by at most 0.0025 relative to the frozen parent: 0.1 and 0.25 percentage points, respectively. Among eligible coefficients it maximizes R100, choosing the smaller coefficient on a tie. The selected value is `gamma_init`.
3. Continuation starts from that checkpoint and holds `gamma_init` fixed during updates and checkpoint evaluation. A new checkpoint must strictly increase `J=(R100+R10)/2` over the current best and meet both tolerances relative to the starting model at `gamma_init`.
4. After restoring the selected continuation checkpoint, final coefficient selection starts at `gamma_init` and visits the grid in ascending order. A replacement must strictly increase J and meet both tolerances against **both** the restored checkpoint at `gamma_init` and the native scores from the same SELECT operations. Otherwise the current coefficient stays selected. A tie never replaces the current choice.

The references in steps 2 and 4 differ: step 2 uses SCC parent scores; step 4 uses the native scores as its second reference. Implementations: [initial_gamma and EpochSelector](../src/fvsr/core/trainer_v2.py), [select_epoch](../src/fvsr/core/training_primitives.py), [select_gamma](../src/fvsr/core/selection.py).

Early stopping counts consecutive epochs without an accepted checkpoint improvement and applies once the stage minimum epoch is reached. Recovery checkpoints retain model, optimizer, scheduler, sampling/RNG state, selection history and configuration identity. The local storage adapter preserves every epoch recovery file, plus the selected checkpoint and per-epoch metrics. Resume requires a matching configuration hash and a new output directory.

## Fixed-code and input-reference controls

The neural decoder initializes an additive `M×256×D` lookup by placing each original codeword in its corresponding coordinate block. Lookup outputs are summed, followed by `BatchNorm(D) → Linear(D,2D) → ReLU → Linear(2D,D)`, without a residual skip. Reconstruction minimizes coordinate MSE on DB_LEARN and selects the lowest DB_CAL MSE, including epoch 0. The query-matched continuation loads the selected reconstruction model and minimizes weighted operation SmoothL1 between standardized negative squared reconstruction distances and standardized exact scores. The common trainer applies the continuation J/tolerance checkpoint rule; its coefficient remains 1 ([NNDecoder](../src/fvsr/core/training_reference.py#L15), [SharedTrainer](../src/fvsr/core/trainer_v2.py)).

The separately recorded W3 neural-decoder protocol selected query-matched checkpoints on SELECT_CAL and used SELECT_CHECK descriptively. Its reconstruction stage used a maximum of 300 epochs and early stopping after at least 50 epochs, two learning-rate reductions and 30 epochs without relative MSE improvement of `1e-4`. The common command-line trainer above does not implement that distinct stopping/selection protocol.

The additive decoder fits the fixed-code linear reconstruction system with ridge coefficient `1e-5`, solving each coordinate by SciPy LSQR (`atol=btol=1e-8`, iteration limit 200; coordinate blocks of 16) in [aq_fixed](../src/fvsr/core/training_reference.py#L30).

POOL-ref and FIT-ref retrain separate PSRC heads with a shared SCC state and query-RMS scaler. FIT-ref replaces current-pool subspace-input moments with fixed FIT moments; supervision, output centering and score rescaling remain pool-based. Its moment weights balance query IDs, predicates and repeated records over the complete FIT operation set, with equal candidate weight inside each pool; empty pools contribute no moment denominator. This differs from equal-record query-RMS normalization. See [reference_factory](../src/fvsr/reference_training.py) and [fit_reference_moments](../src/fvsr/core/reference_control.py#L13).

## Environment and measured request boundaries

| Environment | Recorded settings |
|---|---|
| Historical measurements | Intel Xeon Gold 6226R, 2.90 GHz; about 62.4 GiB RAM; Python 3.10.13; NumPy 1.26.4; Faiss 1.9.0; PyTorch 2.1.2 |
| Saved software-check environment | Windows CPU; Python 3.12 environment; NumPy 1.26.4; PyTorch 2.8.0+cpu; SciPy 1.15.3; full package list in [environments](../environments/requirements-clean-cpu-py312.txt) |
| Package constraints | Python 3.10–3.12; NumPy 1.26.4; PyTorch >=2.1,<2.9; optional Faiss >=1.9,<1.10; see [pyproject.toml](../pyproject.toml) |

Search and SCC–PSRC scoring were measured in CPU FP32 on one pinned thread. Affinity was logical CPU2 for SIFT1M/IVF/OPQ32 and DEEP1M/residual-IVFPQ32, and CPU6 for the other recorded settings. Latency ratios pair methods within the same setting. ExactPool used FP64 distances as an offline same-pool reference.

Complete-request timing includes predicate preparation, retrieval, query transformations, code gathering, lookup evaluation, valid-pool statistics, residual scoring and final top-k selection. The recorded SCC–PSRC path also reconstructs the calibrated codebook and performs the additional model-side query transform on every request. That codebook contraction costs `O(R*K*D)`, with rank `R=8` and `K=256`. Loading, initialization, warmup, offline ground truth, export and network/API/IPC are excluded. Quantiles pool valid request observations across the separately evaluated models rather than averaging their quantiles.

The saved protocol states 32 warmup requests, five repetitions for the labels W3/PQ16 and DEEP1M/PQ32, and three for the other labels. It does not resolve the repetition mapping for every backend or whether the warmup count applies per repetition or per model. The metadata cost evaluation uses a 128-query subset and three timing repetitions per separately evaluated fixed model.

The offline neural-decoder implementation supports temporary GPU FP32 reconstructions; complete-request CPU timings for native scoring, SCC–PSRC and RAM Exact exclude that cache. The saved description does not identify the execution device for every offline decoder run. Historical OS/BLAS versions and a universal query-to-database self-match exclusion rule also remain unverified.

The optional `full-cm` serving mode materializes the calibrated codebook once for CPU global PQ with identity rotation. Its scores are checked against the reference implementation, but this cached mode is distinct from the per-request construction used in the recorded latency measurements ([materialized.py](../src/fvsr/materialized.py)).

## Data roles and statistical units

The evaluated datasets are SIFT1M, DEEP1M, GLOVE100, NYTimes and W3 arXiv embeddings. Dataset descriptions and available upstream links are in [DATASETS.md](DATASETS.md). [DATA_SOURCES.json](../manifests/DATA_SOURCES.json) records the pinned W3 files.

GLOVE100 and NYTimes vectors use FP64 normalization followed by FP32 storage. The other datasets receive no additional normalization. NYTimes preprocessing excludes 239 zero-valued database rows and one zero-valued development query; removing three further development queries that duplicate FIT content leaves 1,020 development queries. The FIT cohort remains 1,024.

| Role / cohort | Use |
|---|---|
| FIT | SCC/PSRC parameter updates and fixed query-RMS population statistics. |
| SELECT_CAL and SELECT_CHECK | Their union forms SELECT for SCC–PSRC checkpoint and coefficient selection. |
| DB_LEARN / DB_CAL | Disjoint database-vector subsets for decoder reconstruction updates and MSE-based selection. |
| 28-configuration evaluation | Development evaluation; configuration summaries weight settings equally. |
| Four-domain new-query evaluation | 512 content groups per domain: DEEP1M/PQ16, GLOVE100/PQ20, NYTimes/PQ16 and W3/PQ16, using fixed domain-specific models and the original databases. |
| W3 decoder and metadata quality | Reuses the evaluated 512 groups. Decoder comparisons were added after Full results were available and are post hoc. |
| Metadata cost | A 128-query subset of the W3 512-group cohort; its quality, latency and resources refer to that subset. |
| Input-reference NQ | Additional query groups used for exploratory evaluation. |
| Selectivity study | Fixed-model SIFT1M/DEEP1M development evaluation. |
| Pool-size study | 128 development queries per W3/SIFT domain, with fixed models and search effort. |

The main development workload uses aligned and permuted attributes at 1%, 5% and 10% selectivity, `L=400` and `k=100`. Aligned attributes follow a fixed normalized Gaussian projection; intervals are centered on the query insertion position. Permuted attributes use a fixed database permutation and deterministic query-dependent intervals. Each interval contains `round(N*selectivity)` database rows within valid bounds. These synthetic workloads differ from the real-metadata `R` and `EMIS` predicates exposed by the current evaluation CLI.

The common training cache encodes FIT=0, SELECT_CAL=1 and SELECT_CHECK=2 under `FIT0_SELECT_CAL1_SELECT_CHECK2`. FIT and SELECT query IDs must be disjoint. The recorded W3 neural-decoder protocol instead selected its query-matched model on SELECT_CAL and used SELECT_CHECK descriptively; that method-specific use does not change SCC–PSRC's union rule. Full selection details are in [IMPLEMENTATION_DETAILS.md](IMPLEMENTATION_DETAILS.md).

Query-RMS normalization weights each FIT record equally, including repeated queries. FIT-ref subspace-input moments use the complete FIT operation set with weights balancing query IDs, predicates and repeated records; candidates within a pool have equal weight, and empty pools contribute no moment denominator. Those moment operations need not equal the operations eligible for SCC loss updates.

Unless a comparison is explicitly a single-model control, learned-scorer metrics average three separately evaluated models with equal weights. Query-content groups are the unit of paired 95% intervals from 10,000 bootstrap resamples; predicates and model associations stay within their groups. Intervals are marginal and conditional on the selected models. Reused queries across code lengths remain reused content groups. The 7,032 query-gain observations are configuration-specific, and the 200 histogram records are bins.

Recall@k uses exact legal top-k targets with denominator `min(k,legal_count)`. Ground truth scans the legal collection in FP64 squared Euclidean distance and resolves ties by ascending database ID. A zero-legal-object operation has no estimable recall. The within-pool net recovery ratio uses identical request/model weights for recall and candidate coverage; a zero available-gain denominator is recorded as unavailable.

Protected TEST and unselected FRESH inputs are outside FIT/SELECT. Complete historical prior-use coverage, every timing-subset member ID and a universal query-to-database self-match exclusion rule have not been established from the retained records.

## External backend versions

The included native adapter supports non-residual squared-L2 Faiss IVF-PQ. The following records identify the external implementations used in historical experiments. Their backend-specific integrations and bound assets are not included in this source tree.

| Component | Exact source/runtime | Original build and boundaries |
|---|---|---|
| Faiss IVF / historical HNSW-PQ | Faiss 1.9.0; Python 3.10.13; NumPy 1.26.4 | `pip install faiss-cpu==1.9.0`; the public IVF adapter checks native type, metric, codebook and every stored ID/code. Original HNSW selection/capture workers remain separate. Installing Faiss does not recreate the selected index. |
| Rii | [Rii 0.2.12](https://pypi.org/project/rii/0.2.12/), nanopq 0.2.2 | Upstream sdist SHA-256 `85ccd0fb5a8a193990128843596bee2306792ff8b520d1f77c01de1f6bece50c`. Original source, `setup.py`, C++ files, runtime Python files and selected binary are internally archived. The source build uses pybind11, OpenMP, C++17 and host SIMD flags; a rebuild is not guaranteed to have the original binary hash. It must reuse the frozen OPQ codes and original configured posting state, not fit a fresh quantizer. |
| RangePQ | [Author repository](https://github.com/CUHK-DBGroup/RangePQ), commit `c932134d2b4b3c414dc1636050443d3a0819777f` | Original adapted `main.cpp` and `segr.h`, patch and remaining upstream source are archived. g++ 11.1.0, pybind11 2.13.6, Python 3.10 headers; `g++ -shared -fPIC -O3 -std=c++17 -fopenmp -march=native -mtune=native -I<pybind11/include> -I<python3.10> src/main.cpp src/pqkmeans.cpp -o main.cpython-310-x86_64-linux-gnu.so`. The exact old binary SHA is `333364833a4ce4ff87af75b386b19d39e6d0f0c5b06302d1362b95c3fda37b91`. The recorded bridge and fixed coarse state are required for that historical configuration. |
| ACORN/RACORN PQ adaptations | [Author artifact](https://github.com/naver/racorn), commit `04223a28e09d46e7e7708038a0083bdd34731edd` | Original project-owned PQ bridge plus upstream headers and FP16 dependency are archived. Original shared-library flags: `g++ -O3 -std=c++17 -pthread -march=native -DUSEARCH_ENABLE_3HOP=0 -DUSEARCH_USE_FP16LIB=1 -I include -shared -fPIC racorn_pq_bridge.cpp -o libracorn_pq.so`. Binary SHA `919ec8798252d77d42eb25bcf6ee563c381b6fa0a7e9e55c2173d24166719e39`. This is an explicitly adapted PQ representation; upstream F32 and other representations are not substituted. |


Pinned versions identify the recorded implementation, rather than the latest upstream version. External code and binaries remain subject to their respective licenses. Backend support in this source tree is summarized in [backend support](USAGE.md#backend-support).
