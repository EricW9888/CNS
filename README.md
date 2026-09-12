# CNS — executable MaleCNS visual-circuit experiments

CNS is a small, reproducible research project for testing how visual-motion
signals propagate through circuits derived from the [MaleCNS v1.0
connectome](https://male-cns.janelia.org/). The current work follows local optic
column inputs through T4 motion neurons and selected downstream pathways. It is
not a whole-brain model and does not yet establish biologically calibrated
direction labels, DNp15 optic-flow selectivity, or steering behavior.

The organizing scientific question is:

> Can a MaleCNS-derived visual circuit distinguish opposite motion sequences,
> and can that information be carried into biologically interpretable
> downstream activity?

## Current results

| Experiment | Result |
|---|---|
| [EXP-001 strict LIF](reports/EXP-001-strict-lif-baseline.md) | L1 spikes and all six Mi1/Tm3 cells respond below threshold, but no signal reaches T4. Direction selectivity cannot be evaluated. |
| [EXP-002 graded local T4](reports/EXP-002-local-motion.md) | The frozen reduced model distinguishes opposite workbook-column orders in 43 T4 cells and across eight additional local materializations. The effect depends mainly on an imposed delayed inhibitory input channel; it is not a connectome-only or calibrated physiological result. |
| [EXP-003 embodied steering](reports/EXP-003-embodied-steering.md) | The temporary bridge changes yaw magnitude, but both stimulus orders and the zero-command control turn in the same direction. It is an engineering scaffold, not a biological steering result. |
| [EXP-003 downstream correction](reports/EXP-003-downstream-correction.md) | A restrictive target filter had excluded T4b/T4c paths. Correcting the materialization restores all four subtypes but still does not recover steering sign. |
| [EXP-003 bilateral readout](reports/EXP-003-bilateral-readout.md) | A minimal right-minus-left DNp15 readout remains same-signed for both mirrored conditions. Body simulation is therefore skipped. |
| [EXP-004 preflight failure](reports/EXP-004-preflight-failure.md) | A provisional recurrent HS/H2 implementation used incorrect electrical-coupling semantics and was unstable. Its outputs have no biological interpretation; the source is retained only to preserve the failed attempt. |

The concise cross-experiment state is maintained in [research
progress](reports/RESEARCH_PROGRESS.md). Machine-readable parameters and results
are under `experiments/`.

![EXP-002 local motion result](figures/EXP-002-direction-selectivity.png)

The figure shows a model-internal activity result. Activity is dimensionless,
the workbook axes are not mapped to canonical visual directions, and the
upstream inhibitory drive is partly imposed rather than fully propagated.

## Scientific boundaries

The project keeps four evidence layers separate:

- **MaleCNS structure:** neuron identity, directed chemical edges, synapse
  counts, sides, and annotations from the released dataset;
- **literature physiology:** only interactions and qualitative dynamics tied to
  cited primary sources;
- **model assumptions:** signs, normalization, time constants, rectification,
  stimulus encoding, and temporary readouts;
- **free parameters:** numerical values that are not measurements from the
  modeled cells.

Every report states which layer supports each part of an experiment. Failures
remain part of the experiment history instead of being replaced by later
models.

## Repository layout

```text
src/          model, numerical, materialization, and provenance modules
scripts/      data queries, experiment runners, verification, and benchmarks
experiments/  machine-readable experiment records and archived failed source
reports/      question/method/result/control/limitation summaries
docs/         architecture, environment, provenance, and diagnostics
figures/      selected compact public evidence
data/         tracked manifest only; downloaded/materialized data are ignored
results/      generated traces, figures, and videos are ignored
```

The current implementation and scaling boundary is described in
[implementation and performance](docs/implementation.md). Historical experiment
modules remain reproducible; future models can use validated sparse operators
and selected-state recording without inheriting the small-circuit Pandas/dense
paths.

## Environment

Python 3.11 is the tested version.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The exact Windows environment used for the 2026-09-12 verification is recorded
in `requirements-verified-windows.txt`. Brian2 is diagnostic-only; EXP-001 runs
with the NumPy/SciPy backend. FlyGym is optional and used only by the historical
EXP-003 body runner.

## Data

Downloaded MaleCNS files, query-derived graph bundles, caches, credentials, and
generated results are excluded from Git. Official source URLs, expected sizes,
SHA-256 hashes, and materialization contracts are tracked in
[`data/manifest.json`](data/manifest.json).

```powershell
Invoke-WebRequest https://male-cns.janelia.org/data/body-annotations.feather -OutFile data/body-annotations.feather
Invoke-WebRequest https://male-cns.janelia.org/data/body-neurotransmitters.feather -OutFile data/body-neurotransmitters.feather
Invoke-WebRequest https://male-cns.janelia.org/data/visual-neuron-columns.xlsx -OutFile data/visual-neuron-columns.xlsx
Invoke-WebRequest https://raw.githubusercontent.com/flyconnectome/2025malecns/main/supplemental_files/optic-column-type-assignments-v1.0.xlsx -OutFile data/optic-column-type-assignments-v1.0.xlsx
```

Verify the local evidence set with:

```powershell
.venv\Scripts\python.exe scripts\verify_local_data.py
```

No neuPrint credential belongs in this repository. Current materializers use
the public `male-cns:v1.0` endpoint.

## Reproduce experiments

EXP-001:

```powershell
.venv\Scripts\python.exe scripts\query_malecns_subgraph.py --eye right --suffixes 01_07 01_08
.venv\Scripts\python.exe run_experiment.py
```

EXP-002 and its frozen held-out validation:

```powershell
.venv\Scripts\python.exe scripts\query_exp002_circuit.py
.venv\Scripts\python.exe run_exp002.py
.venv\Scripts\python.exe scripts\run_exp002_validation.py
```

Historical EXP-003 commands are documented in their reports. Body runs require
the optional FlyGym environment. Generated CSVs, videos, caches, and graph
materializations remain ignored.

## Verify the repository

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m compileall -q src scripts tests run_experiment.py run_exp002.py
.venv\Scripts\python.exe -m pip check
.venv\Scripts\python.exe scripts\repository_policy_check.py
.venv\Scripts\python.exe scripts\verify_local_data.py --allow-missing
.venv\Scripts\python.exe scripts\benchmark_foundation.py
```

CI runs the fast checks without downloading MaleCNS or executing body
simulations. A separate workflow scans all reachable Git history with a
checksum-verified Gitleaks binary.

## Next scientific gate

EXP-004 restarts as a neural physiology experiment, without FlyGym. Controlled
HS/H2 inputs representing published optic-flow stimulus classes must first be
mapped onto identified cells. Any literature-added electrical coupling must use
specific supported pairs and a diffusive `other_state - self_state` current.
Interpretation is gated on effective-transition stability, zero-input decay,
determinism, explicit interaction provenance, and a quantitative external
physiology target fixed before evaluation.

## Sources

- [MaleCNS v1.0](https://male-cns.janelia.org/)
- [Official MaleCNS downloads and license](https://male-cns.janelia.org/download/)
- [MaleCNS supplemental optic-column assignments](https://github.com/flyconnectome/2025malecns)
- [Erginkaya et al. 2025 — H2/HS recurrent optic-flow network](https://doi.org/10.1038/s41593-025-01948-9)

## Release status

The canonical GitHub repository is private. It must remain private until the
owner selects an explicit license for this project's code. Upstream MaleCNS
data remains governed by its own license and is never copied into the tracked
tree. Git identity normalization is documented in the secondary
[history-normalization note](docs/history-normalization.md).
