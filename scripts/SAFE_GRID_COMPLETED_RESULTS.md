# Optional Safe Grid Completed Results

This memo is optional directional evidence for remote feedback tuning. Read it
together with `scripts/SAFE_FEEDBACK_TUNING_HANDOFF.md` if it exists. If this file
is absent on the remote machine, ignore it and run the standalone handoff flow.

Do not use any row here as a paper-table update by itself. Every selected
candidate still needs the exact 10-run confirmation required by the handoff.

## Snapshot

- Snapshot source: `experiment_output/final_scgcn_push/20260503_124358_866_safe_contract_grid/results.jsonl`
- Summary source: `experiment_output/final_scgcn_push/20260503_124358_866_safe_contract_grid/summary.md`
- Parsed rows at snapshot: 144
- Protocol in these rows: `runs=10`, `seed_start=0`, training seeds `0..9`
- Mode: `graph_mode=dual`, `fusion_mode=attn`, `enable_dcgl_negative_loss=True`
- AE graph: reused fixed `data/ae_graph/<dataset>_ae_graph.txt`
- Completed datasets in this snapshot: `reut`, `uat`, `amap`, `usps`
- No completed rows yet in this snapshot: `eat`, `cora`, `cite`

## Overall Readout

| Dataset | Rows | Pass rows | Best ACC candidate | ACC/NMI/ARI/F1 | Std ACC/NMI/ARI/F1 | Practical use |
| --- | ---: | ---: | --- | --- | --- | --- |
| reut | 12 | 0 | `safe_reut_joint_1336fc06` | 82.84/58.24/65.33/69.43 | 1.22/1.96/2.01/3.74 | no table candidate; use only as negative-strength clue |
| uat | 24 | 0 | `safe_uat_center_bf7abb91` | 55.73/27.30/22.42/55.33 | 1.21/0.63/1.72/2.33 | no table candidate; ARI can rise but F1 drops |
| amap | 79 | 3 | `safe_amap_joint_fa18e3c0` | 77.45/67.35/58.31/72.00 | 0.38/0.49/0.52/0.66 | strongest optional local base; confirm before table use |
| usps | 29 | 0 | `safe_usps_attn_dc994718` | 78.88/70.65/67.34/75.84 | 2.26/0.50/2.26/3.65 | far below current table; diagnose, do not chase blindly |

Pass means the candidate improved all four current Ours target metrics in the
grid script comparison. Only AMAP has pass rows in this snapshot.

## AMAP Completed Clues

AMAP is the only completed dataset with clear four-metric gains. Use these rows
as optional starting bases for feedback refinement, not as final table values
until exact confirmation.

| Candidate | Pass | ACC/NMI/ARI/F1 | Std ACC/NMI/ARI/F1 | Key params | Suggested next move |
| --- | --- | --- | --- | --- | --- |
| `safe_amap_joint_fa18e3c0` | YES | 77.45/67.35/58.31/72.00 | 0.38/0.49/0.52/0.66 | `fusion_balance=0.05`, `fusion_min_weight=0.0`, `lambda_inst=0.03`, `lambda_clu=0.035`, `dcgl_neg_tau=1.0`, `dcgl_neg_weight=0.8` | best all-metric pass by ACC; confirm once, then probe nearby |
| `safe_amap_joint_e1d5212b` | YES | 77.45/67.31/58.41/71.73 | 0.41/0.63/0.71/0.68 | `fusion_balance=0.05`, `fusion_min_weight=0.05`, `lambda_inst=0.07`, `lambda_clu=0.035`, `dcgl_neg_tau=0.5`, `dcgl_neg_weight=1.0` | best ARI among pass rows; useful if ARI is priority |
| `safe_amap_joint_d9720d86` | YES | 77.42/67.37/58.27/72.08 | 0.34/0.56/0.52/0.62 | `fusion_balance=0.05`, `fusion_min_weight=0.0`, `lambda_inst=0.07`, `lambda_clu=0.035`, `dcgl_neg_tau=1.0`, `dcgl_neg_weight=0.8` | better F1 than `fa18e3c0`; compare by confirmation |
| `safe_amap_joint_69e92626` | NO | 77.42/67.51/58.07/72.34 | 0.28/0.51/0.66/0.57 | `fusion_balance=0.05`, `fusion_min_weight=0.05`, `lambda_inst=0.0`, `lambda_clu=0.035`, `dcgl_neg_tau=1.0`, `dcgl_neg_weight=0.8` | best score-style row, but ARI is below current Ours; use only if F1/NMI probing |

AMAP direction:

- `fusion_balance=0.05` is consistently better than the current base region.
- `lambda_clu=0.035` should stay fixed initially.
- `fusion_min_weight=0.0` and `0.05` are both viable; compare by confirmation.
- `dcgl_neg_tau=1.0`, `dcgl_neg_weight=0.8` looks strong when paired with low
  `fusion_balance`.
- Do not widen AMAP immediately. First confirm `fa18e3c0` and `d9720d86`, then
  probe one step around `lambda_inst=0.03/0.07` and `fusion_min_weight=0.0/0.05`.

## Reuters Completed Clues

No Reuters row improves the current Ours line. The best rows show a tradeoff
between ACC/stability and F1.

| Candidate | Wins | ACC/NMI/ARI/F1 | Std ACC/NMI/ARI/F1 | Key params | Readout |
| --- | ---: | --- | --- | --- | --- |
| `safe_reut_joint_1336fc06` | 0 | 82.84/58.24/65.33/69.43 | 1.22/1.96/2.01/3.74 | `fusion_min_weight=0.10`, `dcgl_neg_tau=0.75`, `dcgl_neg_weight=0.4` | best ACC and lower ACC std; hurts NMI/ARI/F1 |
| `safe_reut_attn_a0c1afec` | 1 | 82.83/58.51/65.50/70.68 | 1.43/1.31/2.59/4.31 | `fusion_min_weight=0.20`, `dcgl_neg_tau=0.5`, `dcgl_neg_weight=0.6` | only top row with F1 above current Ours; other metrics still below |
| `safe_reut_dcgl_7ab603db` | 0 | 82.74/58.70/65.09/69.43 | 1.38/1.23/2.41/3.99 | `fusion_min_weight=0.15`, `dcgl_neg_tau=0.75`, `dcgl_neg_weight=0.6` | better NMI among stable rows, but not enough |

Reuters direction:

- Lowering `dcgl_neg_weight` to `0.4` with `dcgl_neg_tau=0.75` improves ACC
  stability but damages F1.
- Higher `fusion_min_weight=0.20` helps F1 but does not solve ACC/NMI/ARI.
- Next feedback probes should be local and paired: `fusion_min_weight=0.15/0.20`
  with `dcgl_neg_tau=0.5/0.75` and `dcgl_neg_weight=0.4/0.6`.
- Do not spend time on the current completed Reuters rows as table candidates.

## UAT Completed Clues

No UAT row passes. Several rows improve ARI or NMI while losing ACC and F1.

| Candidate | Wins | ACC/NMI/ARI/F1 | Std ACC/NMI/ARI/F1 | Key params | Readout |
| --- | ---: | --- | --- | --- | --- |
| `safe_uat_center_bf7abb91` | 1 | 55.73/27.30/22.42/55.33 | 1.21/0.63/1.72/2.33 | current center: `fusion_temp=1.9`, `fusion_balance=0.35`, `fusion_min_weight=0.20`, `lambda_clu=0.07`, `dcgl_neg_tau=0.5`, `dcgl_neg_weight=0.6` | best ACC and best score-style row; still below current Ours except ARI |
| `safe_uat_attn_e333b1f0` | 1 | 55.57/27.14/22.52/55.14 | 1.15/0.67/1.75/2.23 | `fusion_balance=0.45`, `fusion_min_weight=0.20`, `lambda_clu=0.075` | small ARI gain with tolerable std, but F1 down |
| `safe_uat_attn_830c4a55` | 2 | 55.52/27.48/23.49/53.84 | 1.78/1.50/2.53/2.95 | `fusion_balance=0.45`, `fusion_min_weight=0.25`, `lambda_clu=0.075` | better NMI/ARI, F1 collapses |
| `safe_uat_attn_3f5d5f47` | 2 | 54.89/27.81/25.60/51.30 | 1.93/1.89/3.22/3.41 | high floor/balance region | ARI approaches headline target, but ACC/F1 are not usable |

UAT direction:

- `fusion_min_weight=0.25` can raise ARI/NMI but damages F1 too much.
- Stay near `fusion_min_weight=0.20` first; use `0.25` only for ARI diagnosis.
- The completed rows do not show useful negative-strength exploration beyond the
  current `dcgl_neg_tau=0.5`, `dcgl_neg_weight=0.6` region. Next probes should
  test lower `dcgl_neg_weight=0.3/0.4/0.5` and higher `dcgl_neg_tau=0.75/1.0`.
- Prioritize candidates that recover F1 before chasing ARI.

## USPS Completed Clues

All completed USPS rows are far below the current Ours line. Treat this as a
diagnostic result, not a tuning success.

| Candidate | Wins | ACC/NMI/ARI/F1 | Std ACC/NMI/ARI/F1 | Key params | Readout |
| --- | ---: | --- | --- | --- | --- |
| `safe_usps_attn_dc994718` | 0 | 78.88/70.65/67.34/75.84 | 2.26/0.50/2.26/3.65 | `fusion_temp=2.0`, `fusion_balance=0.35`, `fusion_min_weight=0.15`, `dcgl_neg_tau=0.5`, `dcgl_neg_weight=0.6` | best ACC/NMI in completed rows, still far below table |
| `safe_usps_attn_e34cd726` | 0 | 78.87/70.45/67.71/75.18 | 2.28/0.61/1.09/3.99 | `fusion_temp=2.0`, `fusion_balance=0.45`, `fusion_min_weight=0.15`, `dcgl_neg_tau=0.5`, `dcgl_neg_weight=0.6` | best ARI in completed rows, F1 weak |
| `safe_usps_joint_d9cfe761` | 0 | 78.44/70.28/66.06/75.94 | 2.72/1.00/3.28/3.90 | `fusion_temp=1.8`, `fusion_balance=0.35`, `fusion_min_weight=0.15`, `dcgl_neg_tau=0.75`, `dcgl_neg_weight=0.8` | stronger negative settings do not fix the gap |
| `safe_usps_joint_8ef6ceae` | 0 | 78.20/70.41/65.49/76.02 | 2.48/1.03/3.49/3.31 | `fusion_temp=1.8`, `fusion_balance=0.35`, `fusion_min_weight=0.25`, `dcgl_neg_tau=0.75`, `dcgl_neg_weight=0.8` | best F1 in completed rows, still unusable |

USPS direction:

- Best completed rows keep `fusion_min_weight=0.15`; higher floors do not solve
  the main gap.
- `fusion_temp=2.0`, `fusion_balance=0.35/0.45`, `dcgl_neg_tau=0.5`,
  `dcgl_neg_weight=0.6` is the only reasonable local region from this snapshot.
- Completed higher negative-strength rows (`dcgl_neg_weight=0.8/1.0`,
  `dcgl_neg_tau=0.75/1.0`) do not improve enough.
- If USPS is revisited under the fixed AE contract, try reducing negative
  strength (`dcgl_neg_weight=0.3/0.4/0.5`) before any wider attention search.
- Do not allocate much remote time to USPS until the fixed AE graph condition and
  baseline reproduction are checked.

## How Remote Agents Should Use This Memo

1. Read the hard contract in `SAFE_FEEDBACK_TUNING_HANDOFF.md` first.
2. If tuning AMAP, start by confirming `safe_amap_joint_fa18e3c0` or
   `safe_amap_joint_d9720d86`, then run nearby one-parameter probes.
3. If tuning Reuters, UAT, or USPS, treat the rows above as negative feedback and
   adjust away from the weak regions described in each section.
4. If tuning EAT, Cora, or Citeseer, ignore this memo for that dataset and follow
   the standalone handoff procedure.
5. If this memo is absent, do not recreate the local grid. Run the feedback
   tuning procedure from the handoff document directly.
