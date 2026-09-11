# MaleCNS v1.0 adjacent-column motion prototype

This repository contains a deliberately small experiment around the adult
male *Drosophila* CNS connectome. It tests a real optic-column input sequence
through the annotated `L1 -> Mi1/Tm3 -> T4a-d` pathway and its selected
wide-field/descending continuation. It does not claim brain emulation, native
behavior, learning, or a general computational substrate.

## Current status

The runnable experiment now uses two adjacent right-eye optic columns,
`ME_R_col_01_07` and `ME_R_col_01_08`, with two conditions that reverse their
temporal order. This is the primary stimulus. It is not a left-eye versus
right-eye pulse. The materialized graph contains only the selected L1 seeds,
their queried Mi1/Tm3 targets, reachable T4a-d targets, selected HSE/HS/VS
projection targets, and their queried `descending_neuron` targets.

The model uses a transparent NumPy/SciPy sparse LIF integrator with delayed
synaptic events. Brian2 was initially selected, but the released Brian2 2.9.0
wheel fails at import under NumPy 2.4.6; the exact environment and complete
traceback are recorded in [`docs/diagnostics/brian2_numpy2.md`](docs/diagnostics/brian2_numpy2.md).
NumPy remains in the 2.x line and the runnable baseline does not modify the
third-party package. The optional reproduction environment is
`requirements-brian2-diagnostic.txt`.

The first run records forward order, reverse order, and a forward-order
T4-to-wide-field projection ablation. A zero downstream response is a result,
not evidence of biological absence: this minimal model has no photoreceptor
model, tonic network drive, receptor-specific glutamate model, or behavioral
decoder.

The observability pass also preserves every `0.1 ms` membrane-voltage sample
for all six Mi1/Tm3 neurons and all 43 T4 neurons, then compares forward and
reverse T4 traces neuron-by-neuron. It writes figures and raw diagnostics under
`results/first_experiment/figures/` without changing the experiment.

The next milestone is EXP-002: can a MaleCNS-derived T4 motion circuit
distinguish opposite directions? EXP-002 replaces the EXP-001 spike-gated
abstraction with a continuous graded local T4 model, retains MaleCNS synapse
counts and direct Mi1/Tm3/Mi4/Mi9/C3/CT1 inputs, and tests an inhibitory-input
ablation. Its compact public result record is
[`experiments/EXP-002-graded-t4-motion/record.json`](experiments/EXP-002-graded-t4-motion/record.json).

## Sources and exact files

* MaleCNS v1.0: <https://male-cns.janelia.org/>
* Official downloads: <https://male-cns.janelia.org/download/>
* Annotation: `https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather`
* Transmitter predictions: `https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-neurotransmitters-male-cns-v1.0.feather`
* Full weighted graph: `https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather`
* Official optic-column assignments: <https://github.com/flyconnectome/2025malecns/blob/main/supplemental_data/optic-column-type-assignments-v1.0.xlsx>
* Programmatic API: <https://neuprint.janelia.org>, dataset `male-cns:v1.0`

Downloaded data, caches, environments, tokens, `.env` files, and generated
results are ignored by Git. The MaleCNS source files are CC-BY; this repository
does not redistribute them.

## Biology tested

```text
adjacent L1 columns, opposite temporal orders
    -> Mi1/Tm3
    -> T4a/T4b/T4c/T4d
    -> HSE/HSN/HSS/HST/VS/VST1/VST2/VSm
    -> descending_neuron
```

The workbook supplies the optic-column grid and each column’s L1 body ID. The
query helper verifies that the two suffixes differ by one grid coordinate and
rejects a non-adjacent pair. The model labels the conditions by workbook order
(`01_07->01_08` and `01_08->01_07`) and does not invent a leftward/rightward
retinal mapping.

The positive-current event is an explicit L1 activity abstraction, not a
photoreceptor light-ON model. L1 is glutamatergic, and a body-level transmitter
label does not by itself specify every postsynaptic receptor sign. The strict
baseline maps acetylcholine to positive, GABA/glutamate to negative, and other
labels to unknown/zero. That sign choice is an explicit limitation.

## Install on Windows

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Download the two small official exports and the supplemental workbook locally:

```powershell
New-Item -ItemType Directory -Force data | Out-Null
Invoke-WebRequest https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather -OutFile data/body-annotations.feather
Invoke-WebRequest https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-neurotransmitters-male-cns-v1.0.feather -OutFile data/body-neurotransmitters.feather
Invoke-WebRequest https://raw.githubusercontent.com/flyconnectome/2025malecns/main/supplemental_data/optic-column-type-assignments-v1.0.xlsx -OutFile data/optic-column-type-assignments-v1.0.xlsx
```

## Reproduce the prototype

Materialize the two adjacent right-eye columns and their real queried path:

```powershell
.venv\Scripts\python.exe scripts/query_malecns_subgraph.py --eye right --suffixes 01_07 01_08
```

Run the three narrow conditions:

```powershell
.venv\Scripts\python.exe run_experiment.py
```

Materialize and run EXP-002 separately:

```powershell
.venv\Scripts\python.exe scripts/query_exp002_circuit.py
.venv\Scripts\python.exe run_exp002.py
```

The amplitude is an exposed sensitivity parameter, not a fitted biological
claim. Outputs are written to `results/first_experiment/` and ignored by Git:

* `metrics.json` / `metrics.csv` — condition-level L1, T4, and descending
  spike metrics plus max voltage by stage;
* `spikes_*.csv` — auditable spike events with body IDs, type, side, and stage;
* `voltage_traces_forward.csv` / `voltage_traces_reverse.csv` — raw numerical
  voltage traces for Mi1/Tm3 and T4;
* `t4_forward_minus_reverse.csv` / `.json` — one row per T4 neuron with maxima,
  pointwise order difference, timing, and spike counts;
* `observability_summary.json` — trace counts, subthreshold excursions, and
  top order-difference rows;
* `figures/pathway_subgraph.png`, `figures/spike_rasters_forward_reverse.png`,
  `figures/voltage_forward_mitm_t4.png`,
  `figures/voltage_reverse_mitm_t4.png`,
  `figures/t4_order_sensitivity_ranked.png`, and
  `figures/t4_heatmap_forward_reverse_difference.png`;
* `data/malecns_visual_subgraph/manifest.json` — exact selected IDs, grid
  order, stages, and query provenance.

## Annotation denominator

The raw annotation export is not a neuron count. The audit script is:

```powershell
.venv\Scripts\python.exe scripts/audit_malecns_annotations.py
```

The current Feather has 211,577 unique body IDs and no duplicate body IDs, but
also includes status/classification categories such as Glia, Unimportant,
Orphan, Assign, Anchor, and status-null records. The published 166,691 figure
is a curated proofread/annotated-neuron denominator; it is not reproduced by
equating raw rows, `status=Traced`, or the current neuPrint `Neuron` label with
one another. The exact audit and the explicit included/excluded set for this
small experiment are in [`docs/diagnostics/annotation_reconciliation.md`](docs/diagnostics/annotation_reconciliation.md).

## Model details

The baseline equations are:

```text
dv/dt = (v_rest - v + g + I_ext) / tau_m
dg/dt = -g / tau_syn
```

An incoming spike adds `w` to `g`, with
`w = synapse_count × transmitter_sign × 0.275 mV`. Events are delayed by the
published baseline delay and integrated with `dt=0.1 ms`. The coupling matrix
is post-by-pre and is built only from queried MaleCNS edges with a known strict
sign.

## Interpretation and scope

A convincing future result would require L1 activity, measurable Mi1/Tm3 and
T4 responses, a difference under temporal reversal, and a causal change under
the T4-to-projection ablation. The current run is intentionally narrow and
does not start FlyGym, full-body movement, framework abstractions, training, or
optimization. The next scientific step is receptor/photoreceptor calibration
and a larger neighboring-column sweep, not a behavioral claim.

## Git hygiene

The repository is initialized locally only. `.gitignore` excludes `.venv`, raw
MaleCNS downloads, Parquet/CSV data, generated results, caches, `.env` files,
and common token/key/certificate suffixes. Before committing, inspect both
`git status --short --ignored` and `git ls-files`, run the repository secret
scan, and stage only source/docs/tests/experiment-records. No neuPrint credential belongs in this
repository. No public remote is created or pushed by the prototype setup.
