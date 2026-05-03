# Safe Feedback Tuning Handoff

This document is an execution plan for remote agents that continue DSAFC tuning.
The goal is to improve the real 10-run mean/std results for:

`reut, uat, amap, usps, eat, cora, cite`

Do not treat this as a seed roll task. AE graphs and training seeds must stay fixed
while tuning one or a few safe hyperparameters at a time.

## Hard Contract

Keep the paper narrative unchanged:

- Use `graph_mode=dual`.
- Use `fusion_mode=attn`.
- Keep `enable_dcgl_negative_loss=True`.
- Keep these modules OFF:
  - dynamic threshold module
  - EMA prototypes module
  - DCGL cluster-level module
  - GCN backbone module
- Use fixed AE graphs generated with `pretrain_seed=42` and `graph_seed=42`.
  If the graph files are missing on the remote machine, generate them once with
  those seeds and then reuse them. Do not roll AE graphs during tuning.
- Use 10-run training results with `seed_start=0`, so seeds are `0..9`.
- Give each dataset a maximum active feedback-tuning budget of 6 hours.
- Do not update paper tables from a single best seed.
- Do not manually edit bad seeds or remove outliers.

The fixed AE graphs are:

```text
data/ae_graph/reut_ae_graph.txt
data/ae_graph/uat_ae_graph.txt
data/ae_graph/amap_ae_graph.txt
data/ae_graph/usps_ae_graph.txt
data/ae_graph/eat_ae_graph.txt
data/ae_graph/cora_ae_graph.txt
data/ae_graph/cite_ae_graph.txt
```

The remote agent does not need any prior local summary file. The required
condition is only that the AE graphs are generated/reused under:

```text
--pretrain_seed 42 --graph_seed 42
```

## Target Lines

Use the current main-table Ours values as the minimum replacement line.

| Dataset | ACC | NMI | ARI | F1 |
| --- | ---: | ---: | ---: | ---: |
| reut | 83.20 | 59.82 | 66.01 | 70.57 |
| uat | 56.24 | 27.31 | 21.98 | 56.44 |
| amap | 77.39 | 67.22 | 58.25 | 71.69 |
| usps | 82.40 | 73.29 | 68.39 | 82.16 |
| eat | 54.76 | 31.97 | 24.32 | 52.98 |
| cora | 73.49 | 55.55 | 50.14 | 71.17 |
| cite | 70.74 | 45.28 | 45.18 | 61.56 |

The actual push target is stricter: the attention result should exceed the best
number already present in the main table for that dataset and metric. Use the
following "main-table best" values as the upper target.

| Dataset | ACC best | NMI best | ARI best | F1 best | Best-source note |
| --- | ---: | ---: | ---: | ---: | --- |
| reut | 83.20 | 59.90 | 66.01 | 70.57 | ACC/ARI/F1 from Ours, NMI from DFCN |
| uat | 56.58 | 28.15 | 25.52 | 56.44 | ACC from SCGC-S, NMI/ARI from CCGC, F1 from Ours |
| amap | 77.48 | 67.67 | 58.48 | 72.22 | SCGC-S |
| usps | 84.91 | 84.16 | 79.50 | 82.16 | ACC/NMI/ARI from SCGC-N*, F1 from Ours |
| eat | 57.94 | 33.91 | 27.71 | 57.96 | ACC/NMI/F1 from SCGC-S, ARI from CCGC |
| cora | 73.88 | 57.58 | 52.51 | 71.17 | ACC from CCGC/SCGC-S, NMI from AGE, ARI from CCGC, F1 from Ours |
| cite | 73.29 | 46.92 | 50.21 | 64.80 | ACC/NMI/ARI from SCGC-N*, F1 from SCGC-S |

For a replacement candidate, prefer four-metric improvement over current Ours.
For a headline push candidate, prefer exceeding the main-table best line above.
If only two or three metrics improve, keep it as a local note, not a table replacement.

## No Required Prior Sweep

This handoff is standalone. The remote machine does not need to run or read any
previous local safety sweep before starting feedback tuning.

Start directly from the dataset base parameters in this document. Use short
one-parameter probes, update the local base from positive/negative feedback, and
only run small pairwise probes after a useful direction is found.

If the remote workspace already contains a prior complete 10-run result that
preserves the hard contract, it may be recorded as optional evidence. It is not a
dependency, and it must not replace the feedback procedure or the exact 10-run
confirmation rule.

Optional local grid clues may be provided in:

```text
scripts/SAFE_GRID_COMPLETED_RESULTS.md
```

If that file exists, read it after this handoff and use it only as positive or
negative direction feedback for completed datasets. If it is absent, do not
recreate the local grid; continue directly with the standalone feedback procedure
below.

## Safe Parameters

The following parameters are safe because they tune strength, temperature, weighting,
or confidence behavior without changing the narrative.

### Shared Safe Parameters

| Parameter | Meaning | Safe use |
| --- | --- | --- |
| `fusion_temp` | attention fusion temperature | tunes softness/sharpness of branch weighting |
| `fusion_balance` | balance regularizer/branch balance strength | tunes branch usage pressure |
| `fusion_min_weight` | minimum branch floor | prevents branch collapse |
| `lambda_inst` | instance consistency strength | tunes regularization strength |
| `lambda_clu` | cluster distribution consistency strength | tunes regularization strength |
| `dcgl_neg_tau` | negative center contrast temperature | tunes negative-loss sharpness |
| `dcgl_neg_weight` | negative-loss coefficient | tunes DCGL-negative strength |
| `threshold` | confidence threshold | safe only as a scalar training hyperparameter |
| `branch_bias_cap` | raw-branch bias cap for citation datasets | safe when `enable_branch_bias_fusion=True` already belongs to the dataset config |
| `warmup_epochs` | warmup duration | safe, but change cautiously |
| `alpha` | cluster-guided loss coefficient | safe, but change cautiously |
| `lr` | optimizer step size | safe, but use only as late-stage refinement |
| `epochs` | training horizon | safe, but use only as late-stage refinement |

Unsafe for this handoff:

- enabling dynamic threshold module
- enabling EMA prototypes
- enabling DCGL cluster-level loss
- enabling GCN backbone
- changing `graph_mode`
- changing `fusion_mode`
- regenerating or rolling AE graphs unless explicitly requested
- replacing the 10-run protocol with selected seeds

## Dataset Starting Points And Safe Ranges

Use the current `experiment.py` values as the base point. Start with the listed
safe ranges, then narrow or extend only after feedback.

### Reuters `reut`

Base:

```text
fusion_temp=1.6
fusion_balance=0.25
lambda_inst=0.08
lambda_clu=0.06
warmup_epochs=35
fusion_min_weight=0.15
dcgl_neg_tau=0.5
dcgl_neg_weight=0.6
```

First-pass safe axes:

```text
fusion_min_weight: 0.10, 0.15, 0.20
dcgl_neg_tau: 0.5, 0.75, 1.0
dcgl_neg_weight: 0.3, 0.4, 0.5, 0.6
fusion_balance: 0.20, 0.25, 0.30
lambda_inst: 0.06, 0.08, 0.10
lambda_clu: 0.04, 0.06, 0.08
```

Direction note: Reuters appears sensitive to negative strength. Prioritize lowering
`dcgl_neg_weight` and increasing `dcgl_neg_tau` to reduce seed sensitivity.

### UAT `uat`

Base:

```text
fusion_temp=1.9
fusion_balance=0.35
lambda_inst=0.08
lambda_clu=0.07
warmup_epochs=35
fusion_min_weight=0.20
dcgl_neg_tau=0.5
dcgl_neg_weight=0.6
```

First-pass safe axes:

```text
fusion_temp: 1.8, 1.9, 2.0, 2.1
fusion_balance: 0.30, 0.35, 0.40, 0.45
fusion_min_weight: 0.18, 0.20, 0.22, 0.25
lambda_clu: 0.06, 0.07, 0.075, 0.08
dcgl_neg_tau: 0.5, 0.75, 1.0
dcgl_neg_weight: 0.3, 0.4, 0.5, 0.6
```

Direction note: UAT should test negative strength early because it often trades
off accuracy and stability.

### AMAP `amap`

Base:

```text
fusion_temp=1.25
fusion_balance=0.08
lambda_inst=0.07
lambda_clu=0.035
warmup_epochs=35
fusion_min_weight=0.05
dcgl_neg_tau=0.5
dcgl_neg_weight=0.6
```

First-pass safe axes:

```text
fusion_balance: 0.05, 0.08, 0.10
lambda_inst: 0.0, 0.03, 0.07
fusion_min_weight: 0.0, 0.05, 0.10
dcgl_neg_tau: 0.5, 1.0
dcgl_neg_weight: 0.6, 0.8, 1.0
lambda_clu: 0.02, 0.035, 0.05
```

Direction note: AMAP is usually a good early dataset because the safe axes are
compact. Probe `fusion_balance`, `fusion_min_weight`, and `dcgl_neg_weight`
before widening the search.

### USPS `usps`

Base:

```text
fusion_temp=1.8
fusion_balance=0.35
lambda_inst=0.09
lambda_clu=0.09
warmup_epochs=35
fusion_min_weight=0.20
dcgl_neg_tau=0.5
dcgl_neg_weight=0.6
```

First-pass safe axes:

```text
fusion_temp: 1.8, 2.0, 2.2
fusion_balance: 0.30, 0.35, 0.45
fusion_min_weight: 0.15, 0.20, 0.25
dcgl_neg_tau: 0.5, 0.75, 1.0
dcgl_neg_weight: 0.3, 0.4, 0.5, 0.6, 0.8
lambda_inst: 0.07, 0.09, 0.11
lambda_clu: 0.07, 0.09, 0.11
```

Direction note: USPS can be sensitive to AE graph quality and negative-loss
strength. Keep AE fixed during this handoff, and diagnose stability with
`dcgl_neg_weight`, `dcgl_neg_tau`, `fusion_temp`, and `fusion_min_weight`.

### EAT `eat`

Base:

```text
fusion_temp=2.0
fusion_balance=0.35
lambda_inst=0.08
lambda_clu=0.08
warmup_epochs=35
fusion_min_weight=0.20
threshold=0.4
dcgl_neg_tau=0.5
dcgl_neg_weight=0.6
```

First-pass safe axes:

```text
threshold: 0.35, 0.4, 0.45, 0.5
fusion_temp: 1.8, 2.0, 2.2
fusion_balance: 0.25, 0.35
fusion_min_weight: 0.15, 0.20
dcgl_neg_tau: 0.35, 0.5, 0.75
dcgl_neg_weight: 0.3, 0.4, 0.5, 0.6, 0.8
lambda_inst: 0.06, 0.08, 0.10
lambda_clu: 0.06, 0.08, 0.10
```

Direction note: EAT is small and volatile. Optimize for `mean - 0.5 * std`, not
raw mean only.

### Cora `cora`

Base:

```text
fusion_temp=1.3
fusion_balance=0.0
lambda_inst=0.03
lambda_clu=0.01
warmup_epochs=70
fusion_min_weight=0.0
enable_branch_bias_fusion=True
branch_bias_target=raw
branch_bias_cap=0.10
threshold=0.4
dcgl_neg_tau=0.5
dcgl_neg_weight=0.6
```

First-pass safe axes:

```text
threshold: 0.35, 0.4
branch_bias_cap: 0.08, 0.10, 0.12
dcgl_neg_tau: 0.35, 0.5, 0.75
dcgl_neg_weight: 0.3, 0.4, 0.6, 0.8, 1.0
lambda_inst: 0.02, 0.03, 0.04
lambda_clu: 0.005, 0.01, 0.02
warmup_epochs: 55, 70, 85
```

Direction note: keep raw branch bias enabled. Do not add new modules.

### Citeseer `cite`

Base:

```text
fusion_temp=1.8
fusion_balance=0.15
lambda_inst=0.045
lambda_clu=0.02
warmup_epochs=55
fusion_min_weight=0.10
enable_branch_bias_fusion=True
branch_bias_target=raw
branch_bias_cap=0.15
dcgl_neg_tau=0.5
dcgl_neg_weight=0.6
```

First-pass safe axes:

```text
fusion_balance: 0.10, 0.15, 0.25, 0.35
fusion_min_weight: 0.05, 0.10, 0.15, 0.20
branch_bias_cap: 0.12, 0.15, 0.18
dcgl_neg_tau: 0.35, 0.5, 0.75, 1.0
dcgl_neg_weight: 0.3, 0.4, 0.6, 0.8, 1.0
lambda_inst: 0.03, 0.045, 0.06
lambda_clu: 0.01, 0.02, 0.03
warmup_epochs: 45, 55, 65
```

Direction note: citation datasets are sensitive to raw-vs-AE branch bias. Keep
`branch_bias_target=raw`; only tune `branch_bias_cap`.

## Feedback Tuning Procedure

Do not run the full Cartesian product unless explicitly asked. Use coordinate
search with positive/negative feedback.

For each dataset, run feedback tuning in 6-hour blocks at most. After every 5 to
10 completed candidates, stop candidate generation, update the local base, and
write a short decision note before launching more tests. Do not continue a static
candidate list when a better direction has already been identified.

### Step 1: Establish Local Base

For each dataset, start from the base listed in this document. If the remote
workspace has its own prior verified 10-run candidate under the hard contract,
record it in the ledger and optionally use it as the local base after one exact
10-run recheck.

Every candidate must run:

```text
--runs 10 --seed_start 0
```

### Step 2: One-Parameter Probes

Change exactly one parameter at a time around the base. Example:

```text
base: dcgl_neg_weight=0.6
probe: 0.4
probe: 0.8
```

Interpretation:

- If mean improves and std does not expand materially, move base to that value.
- If mean slightly drops but std improves a lot, keep it as a stability candidate.
- If both mean and std worsen, reject that direction.
- If only F1 improves but ACC/NMI/ARI drop, record it but do not replace the base.

### Step 3: Directional Extension

If a direction works, extend one step:

```text
0.6 -> 0.4 works, try 0.3
0.4 -> 0.3 works, try 0.2 only if narrative remains credible
```

For `dcgl_neg_weight`, avoid going below `0.2` unless explicitly requested.
For `fusion_min_weight`, avoid making the branch floor so high that attention
fusion becomes effectively fixed averaging.

### Step 4: Pairwise Coupling

Only after one-parameter probes identify two useful axes, run small pairwise tests.

Good pairings:

- `dcgl_neg_weight` x `dcgl_neg_tau`
- `fusion_min_weight` x `fusion_balance`
- `lambda_inst` x `lambda_clu`
- `branch_bias_cap` x `fusion_min_weight` for Cora/Citeseer
- `threshold` x `dcgl_neg_weight` for EAT/Cora

Keep pairwise grids tiny: 4 to 9 candidates.

### Step 5: Confirmation

When a candidate beats the current base, rerun the exact same candidate as a
complete 10-run reproduction with the same fixed AE and same seed window `0..9`.
This is mandatory before the result can be used as a table candidate.

If the candidate was first discovered from a 10-run probe, still run one exact
10-run confirmation before table update. If exact rerun differs materially,
inspect `train.py` for remaining nondeterminism or GPU nondeterminism before
updating tables.

## Ranking Rule

Use this score for local comparison:

```text
score = ACC_mean + 0.4 * F1_mean + 0.2 * NMI_mean + 0.2 * ARI_mean
        - 0.25 * (ACC_std + F1_std)
```

But table replacement still requires metric-by-metric review against current Ours.

Suggested decision labels:

- `ACCEPT`: improves at least three metrics and does not damage std.
- `STRONG_ACCEPT`: improves all four metrics.
- `HEADLINE_ACCEPT`: exceeds the main-table best line for the target metric set.
- `STABILITY_KEEP`: similar mean but much lower std.
- `REJECT`: worse mean and worse/equal std.
- `SIDE_NOTE`: improves one metric only.

## Command Template

Use `train.py` directly for feedback probes. Replace dataset, cluster number, AE graph,
and candidate parameters.

```powershell
C:\Users\stern\anaconda3\envs\SCGC_1\python.exe train.py `
  --dataset amap `
  --cluster_num 8 `
  --graph_mode dual `
  --fusion_mode attn `
  --t 4 `
  --linlayers 1 `
  --epochs 400 `
  --dims 500 `
  --lr 0.0001 `
  --threshold 0.4 `
  --alpha 0.5 `
  --knn_k 5 `
  --device cuda `
  --runs 10 `
  --seed_start 0 `
  --ae_graph_path data\ae_graph\amap_ae_graph.txt `
  --enable_dcgl_negative_loss `
  --dcgl_neg_tau 0.5 `
  --dcgl_neg_weight 0.6 `
  --warmup_epochs 35 `
  --fusion_hidden 64 `
  --fusion_temp 1.25 `
  --fusion_balance 0.08 `
  --lambda_inst 0.07 `
  --lambda_clu 0.035 `
  --fusion_min_weight 0.05
```

For Cora/Citeseer include:

```powershell
--enable_branch_bias_fusion `
--branch_bias_target raw `
--branch_bias_cap 0.15
```

## Logging Requirements

Create a per-dataset tuning folder under:

```text
experiment_output/feedback_safe_tuning/<dataset>/
```

For every run, save:

- full command
- timestamp
- dataset
- base candidate id
- changed parameter(s)
- ACC/NMI/ARI/F1 mean and std
- decision label
- short interpretation

Maintain a markdown ledger:

```text
experiment_output/feedback_safe_tuning/<dataset>/feedback_log.md
```

Minimum row format:

| Step | Base | Changed | ACC | NMI | ARI | F1 | Std ACC/NMI/ARI/F1 | Decision | Note | Log |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |

Also maintain one global summary:

```text
experiment_output/feedback_safe_tuning/summary.md
```

## Table Update Rule

Do not update paper tables until:

1. The candidate is a complete 10-run mean/std result.
2. The AE graph is fixed from seed 42.
3. Training seeds are `0..9`.
4. The command preserves the hard contract.
5. The candidate has an exact 10-run confirmation run recorded.
6. The result is better than current Ours by metric-by-metric review.
7. The headline target check against the main-table best line is recorded.
8. The source log path and summary are recorded.

When updating docs, update only Ours/DSAFC values and corresponding source notes.
Do not change baseline columns.

## Immediate Next Action For Remote Agent

1. Verify that the fixed AE graph files exist. If any are missing, generate them
with `pretrain_seed=42` and `graph_seed=42`, then reuse the generated files.
2. If `scripts/SAFE_GRID_COMPLETED_RESULTS.md` exists, read it as optional
direction feedback. If it does not exist, skip this step.
3. Create the logging folders:

```text
experiment_output/feedback_safe_tuning/<dataset>/feedback_log.md
experiment_output/feedback_safe_tuning/summary.md
```

4. Start from the base parameters listed in this document.
5. Run one-parameter probes first, then update the local base after every 5 to 10
completed candidates.
6. Start feedback tuning in this suggested order:

```text
amap -> reut -> uat -> cora -> cite -> eat -> usps
```

Allocate up to 6 hours per dataset. If a dataset reaches a strong local optimum
early, stop that dataset and move on; unused time does not need to be spent.

Rationale:

- AMAP has compact safe axes and is a good first sanity check for the workflow.
- Reuters/UAT have clear under-target gaps and need stability/negative-strength search.
- Cora/Citeseer need raw-branch cap and negative-strength refinement.
- EAT/USPS are more volatile and should be tuned with stability-aware scoring.
