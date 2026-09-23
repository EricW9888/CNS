# CNS — building an executable digital fruit fly

Building an executable digital fruit fly from the
[MaleCNS connectome](https://male-cns.janelia.org/).

CNS is an experimental project integrating mapped neural anatomy, biologically
constrained dynamics, sensory input and motor output into an executable system,
with embodiment and closed-loop behavior as the longer-term goal. It is built
bottom-up: each stage is tested against available anatomy and physiology so
failures can be localized before integration.

![EXP-008 observer: 3D world, measured compound-eye light and frozen neural activity](figures/EXP-008-observer.png)

Post-hoc EXP-008 observer: a prescribed 3D scene becomes measured compound-eye
facet signals, provisional central-local motion channels and identified
downstream activity through DNp15. The human camera is observer-only; this is
not a whole-eye retinotopy or behavior reconstruction.
[Panel guide and reproduction](docs/EXP-008-observer.md) ·
[Static overview](figures/EXP-008-observer-overview.png).

## Current capabilities

Today CNS contains visual-motion and optic-flow circuits extending into
selected descending pathways, with reproducible model results and provisional
interfaces. Local optic-column inputs are modeled through
[T4 motion neurons](https://elifesciences.org/articles/24394); EXP-005 also connects
explicit binocular stimuli through T4/T5, HS/H2 and central intermediates to DNp15.
EXP-008 supplies measured bilateral compound-eye samples from a controlled 3D
world through a provisional central-local readout. EXP-009/010 introduce a
broader measured-grid representation, while EXP-011 freezes the result that its
strict physical-light convergence criterion is not met.
EXP-006 extends that frozen chain to identified neck motor neurons and a
provisional open-loop head-torque interface.
It is not yet a whole-fly or whole-CNS simulation and does not establish
biologically calibrated direction labels,
reproduced [DNp15 binocular physiology](https://doi.org/10.1038/s41593-025-01948-9), or
biologically interpretable steering behavior.

## Current direction

[EXP-008](reports/EXP-008-compound-eye.md) establishes the measured compound-eye
sensory boundary. [EXP-009](reports/EXP-009-retinotopic-eye.md) introduces broader
local-direction sampling, but fails translation refinement.
[EXP-010](reports/EXP-010-retinotopic-refinement.md)'s fixed finer temporal
sampling improves the errors to 4.95% / 6.06%; the unchanged 5% criterion still
fails. The discrepancy localizes to blocker-intersecting acceptance cones,
consistent with occlusion / optical sampling sensitivity at a hard visibility
boundary. Numerical validation and individual measured-eye-to-MaleCNS
registration remain separate unresolved requirements.

[EXP-011](experiments/EXP-011-acceptance-convergence/record.json) tests numerical
convergence of that measured-grid sensory interface without changing the scene,
biology or downstream model. Its local relative convergence checks pass, but
the strict `1e-4` production/reference physical-light criterion fails in every
condition. The frozen negative numerical result does not establish failure of
the biological organism architecture; downstream LPi/HS/H2/DNp15 activity was
not evaluated.

[EXP-007](reports/EXP-007-CvNA2-motor-boundary.md)'s motor-calibration boundary
also remains: DNp15-to-CvNA recruitment and posture-conditioned calibration
await additional physiology, and EXP-006's torque interface is uncalibrated.
These results do not justify sensory-loop closure or steering claims;
EXP-004's failure to reproduce the
[published recurrent enhancement](https://doi.org/10.1038/s41593-025-01948-9) remains
part of the frozen evidence.

## Experimental development history

The experiments are validation checkpoints for constructing CNS. They record
what each stage establishes, its assumptions, and where propagation or
interpretation fails.

| Experiment | Result |
|---|---|
| [EXP-001 strict LIF](reports/EXP-001-strict-lif-baseline.md) | L1 spikes and all six Mi1/Tm3 cells respond below threshold, but no signal reaches T4. Direction selectivity cannot be evaluated. |
| [EXP-002 graded local T4](reports/EXP-002-local-motion.md) | The frozen reduced model distinguishes opposite workbook-column orders in 43 T4 cells and across eight additional local materializations. The effect depends mainly on an imposed delayed inhibitory input channel; it is not a connectome-only or calibrated physiological result. |
| [EXP-003 embodied steering](reports/EXP-003-embodied-steering.md) | The temporary bridge changes yaw magnitude, but both stimulus orders and the zero-command control turn in the same direction. It is an engineering scaffold, not a biological steering result. |
| [EXP-003 downstream correction](reports/EXP-003-downstream-correction.md) | A restrictive target filter had excluded T4b/T4c paths. Correcting the materialization restores all four subtypes but still does not recover steering sign. |
| [EXP-003 bilateral readout](reports/EXP-003-bilateral-readout.md) | A minimal right-minus-left DNp15 readout remains same-signed for both mirrored conditions. Body simulation is therefore skipped. |
| [EXP-004 preflight failure](reports/EXP-004-preflight-failure.md) | A provisional recurrent HS/H2 implementation used incorrect electrical-coupling semantics and was unstable. Its outputs have no biological interpretation; the source is retained only to preserve the failed attempt. |
| [EXP-004 controlled binocular physiology](reports/EXP-004-binocular-physiology.md) | A resolved 37-neuron / 360-edge circuit is stable and reduces DNp15 translation sensitivity relative to HS/H2. However, the feed-forward control is more selective and key calcium targets remain unmatched; the proposed recurrent enhancement is not reproduced. |
| [EXP-005 continuous sensory-to-DNp15 chain](reports/EXP-005-sensory-to-DNp15.md) | Explicit binocular luminance stimuli propagate through provisional ON/OFF motion channels, identified T4/T5 projections, HS/H2 and central intermediates to both DNp15 cells. Visual disconnection abolishes descending activity. The limited transformation comparison passes, but the recurrent-mechanism comparison still fails; early vision remains an approximation. |
| [EXP-006 neck-motor/open-loop physical boundary](reports/EXP-006-DNp15-neck-motor.md) | Identified DNp15-to-CvNA1/CvNA2 connections drive a provisional torque interface with small, opposite-signed head-azimuth changes and causal disconnection controls. This is not calibrated muscle physiology, whole-body yaw, walking steering or closed-loop behavior. |
| [EXP-007 CvNA2 motor-calibration boundary](reports/EXP-007-CvNA2-motor-boundary.md) | Primary CvNA2 activation curves provide a quantitative movement target with explicit digitization uncertainty. DNp15-to-CvNA recruitment and a posture-conditioned motor transform remain unidentified, so the data do not calibrate or replace EXP-006's torque bridge. |
| [EXP-008 measured compound-eye boundary](reports/EXP-008-compound-eye.md) | Controlled 3D scenes produce light samples at 857 left / 852 right measured facet directions. A provisional 65 L / 66 R central-local readout propagates to both DNp15 cells; individual facet-to-MaleCNS registration remains unresolved. |
| [EXP-009 broader local-direction representation](reports/EXP-009-retinotopic-eye.md) | A 1,500-neighborhood measured-grid representation introduces local direction channels, but translation refinement errors of 6.98% / 9.27% exceed the predeclared 5% criterion. It does not validate full-eye retinotopy. |
| [EXP-010 temporal refinement and spatial sensitivity](reports/EXP-010-retinotopic-refinement.md) | The fixed 1 ms-to-0.5 ms successor improves translation errors to 4.95% / 6.06%, without passing the unchanged 5% criterion. Error energy concentrates near blocker-intersecting acceptance cones; numerical sensitivity and biological registration remain distinct blockers, not downstream instability or biological failure. |
| [EXP-011 measured-grid numerical convergence](experiments/EXP-011-acceptance-convergence/record.json) | Under unchanged biological, scene and downstream assumptions, the local relative convergence checks pass but the strict physical-light production/reference criterion fails in all five conditions. This is a frozen negative numerical result; downstream activity was not evaluated, and it does not establish failure of the biological architecture. |

The concise cross-experiment state is maintained in [project
status](reports/RESEARCH_PROGRESS.md). Machine-readable parameters and results
are under `experiments/`.

![EXP-002 local motion result](figures/EXP-002-direction-selectivity.png)

The figure shows a model-internal activity result. Activity is dimensionless,
the workbook axes are not mapped to canonical visual directions, and the
upstream inhibitory drive is partly imposed rather than fully propagated.

## Evidence and assumptions

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
with the NumPy/SciPy backend. FlyGym is optional and used by the historical
EXP-003 body runner and EXP-006 open-loop neck test.

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

The [experiment registry](experiments/registry.json) lists canonical verification
commands; the [experiment workflow](docs/experiment-workflow.md) distinguishes
saved-input replay from production runs and historical invariant checks.

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

EXP-004 (direct controlled HS/H2 input, no body or T4/T5):

```powershell
.venv\Scripts\python.exe scripts\prepare_exp004_physiology.py --download-sources --check-specification
.venv\Scripts\python.exe scripts\run_exp004_physiology.py --check-record
```

The [report](reports/EXP-004-binocular-physiology.md) describes the release-aware
identity crosswalk, observation assumptions, physiological target and controls.

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

## Sources

- [MaleCNS v1.0](https://male-cns.janelia.org/)
- [Official MaleCNS downloads and license](https://male-cns.janelia.org/download/)
- [MaleCNS supplemental optic-column assignments](https://github.com/flyconnectome/2025malecns)
- [Erginkaya et al. 2025 — H2/HS recurrent optic-flow network](https://doi.org/10.1038/s41593-025-01948-9)

## Release status

Project code and original documentation are licensed under
[Apache-2.0](LICENSE). Upstream MaleCNS data retains its
[separate data license](https://male-cns.janelia.org/download/) and is not copied
into the tracked tree; third-party works retain their original terms.
Release checks cover reproducibility, provenance and secret scanning.
Licensing does not strengthen the scientific claims above.
