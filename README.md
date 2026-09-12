# CNS — audited MaleCNS motion-circuit research prototype

CNS is a small, transparent research project testing visual-motion and
visual-to-descending hypotheses against the adult male *Drosophila* MaleCNS
v1.0 connectome. It is not a brain emulator, validated biophysical simulator,
or demonstrated behavioral model.

## What the project currently establishes

The strongest positive result is narrow: a frozen phenomenological graded model
using MaleCNS local connectivity distinguishes opposite temporal orders of two
neighboring optic-column inputs. The response generalizes across sampled
workbook-grid locations. It does **not** yet establish a canonical visual
direction, biological T4 voltage, or behavior, because delayed inhibitory drive,
signs, time constants, coordinate inference, rectification, and normalization
remain explicit model assumptions.

![EXP-002 local order sensitivity](figures/EXP-002-direction-selectivity.png)

The complete audited status is:

| Work | Status | Result |
|---|---|---|
| EXP-001 strict LIF | **INCONCLUSIVE** | L1 spikes and Mi1/Tm3 responds subthreshold; nothing reaches T4. |
| EXP-001 observability | **SUPPORTED diagnostic** | Preserves the same failed run's full voltage evidence. |
| EXP-002 graded T4 | **SUPPORTED in model** | Opposite workbook orders produce different T4 activity; biological direction/mechanism remain unresolved. |
| EXP-003 embodiment | **SUPPORTED engineering scaffold** | Order changes yaw magnitude, but both conditions and zero-command control turn the same way. |
| EXP-003 downstream audit | **SUPPORTED anatomy correction** | A target-filter bug excluded T4b/c; restoring them does not fix steering sign. |
| EXP-003 bilateral readout | **UNSUPPORTED in minimal model** | DNp15 right-minus-left stays same-signed; body simulation is correctly skipped. |
| provisional EXP-004 | **INVALID_IMPLEMENTATION** | Incorrect electrical semantics and unstable recurrence invalidate all biological interpretation. |

See the [full claim ledger](reports/PROJECT_AUDIT_2026-09-12.md) and concise
[experiment reports](reports/).

## Scientific boundary

MaleCNS supplies neuron/body identities, annotations, sides, directed chemical
connections, synapse counts, and optic-column L1 assignments. It does not supply
the model's receptor signs, dynamics, gap junctions, world geometry, controller
actions, or parameter values.

Every report separates:

- MaleCNS structural facts;
- literature-supported physiology;
- model assumptions and free parameters;
- engineering or embodiment scaffolding;
- measured outputs and controls.

Workbook suffixes such as `01_07→01_08` are stimulus-space coordinates only.
They are not called leftward/rightward, front-to-back, or yaw direction without
an authoritative geometry mapping.

## Repository map

```text
src/          historical models plus reusable numerical/provenance checks
scripts/      materialization, experiment, and verification entry points
tests/        fast unit and scientific-invariant tests
experiments/  exact machine-readable records, including invalid history
reports/      audited human-readable scientific reports
docs/         architecture, provenance, environment, and technical diagnostics
figures/      small deliberately promoted public evidence
data/         ignored raw/materialized data plus tracked hash registry
results/      ignored bulk traces, figures, and videos
.github/      lightweight CI and full-history secret scan
```

Historical implementations remain intact when changing them would alter an old
result. Corrected science must normally become an explicit successor experiment.
The invalid provisional EXP-004 logic is quarantined under its experiment record
and is not active source code.

## Install

Python 3.11 is the audited version.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --disable-pip-version-check -r requirements.txt
```

The exact audited Windows environment is recorded in `requirements-audit.txt`.
FlyGym/MuJoCo are optional historical EXP-003 dependencies in
`requirements-exp003-flygym.txt`; the neural-only core does not require them.

## Obtain MaleCNS inputs

Raw data is not redistributed. MaleCNS v1.0 is licensed CC-BY. Download the
three registered source files:

```powershell
New-Item -ItemType Directory -Force data | Out-Null
Invoke-WebRequest https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather -OutFile data/body-annotations.feather
Invoke-WebRequest https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-neurotransmitters-male-cns-v1.0.feather -OutFile data/body-neurotransmitters.feather
Invoke-WebRequest https://raw.githubusercontent.com/flyconnectome/2025malecns/main/supplemental_data/optic-column-type-assignments-v1.0.xlsx -OutFile data/optic-column-type-assignments-v1.0.xlsx
```

`data/manifest.json` records source and materialization SHA-256 hashes. Verify
the local evidence set with:

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

EXP-002 and its held-out validation:

```powershell
.venv\Scripts\python.exe scripts\query_exp002_circuit.py
.venv\Scripts\python.exe run_exp002.py
.venv\Scripts\python.exe scripts\run_exp002_validation.py
```

Historical EXP-003 commands are documented in their reports. They require the
optional FlyGym environment for body runs. Generated CSVs, videos, caches, and
materializations remain ignored.

## Verify the repository

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m compileall -q src scripts tests run_experiment.py run_exp002.py
.venv\Scripts\python.exe -m pip check
.venv\Scripts\python.exe scripts\repository_policy_check.py
.venv\Scripts\python.exe scripts\verify_local_data.py --allow-missing
```

CI runs these fast checks without downloading MaleCNS or running expensive
connectome/body experiments. A separate workflow scans all reachable Git history
with checksum-verified Gitleaks.

## Next scientific gate

The recommended successor is neural-only EXP-004: drive anatomically identified
bilateral HS/H2 inputs with controlled published stimulus classes, then test
whether a stable recurrent/inhibitory network increases DNp15
rotation-versus-translation discrimination relative to upstream HS/H2.

Before interpretation it must have:

- a defensible MaleCNS-to-literature neuron identity crosswalk;
- only literature-supported electrical pairs, kept separate from chemical edges;
- diffusive coupling proportional to `other_state - self_state`;
- effective-transition/Jacobian stability and zero-input decay;
- quantitative external physiology targets fixed before evaluation;
- full, no-electrical, recurrent/inhibitory, and chemical-recurrence controls.

No FlyGym/body work resumes until that neural gate passes. A failed stable model
must be preserved as a negative model result, not tuned into success.

## Sources

- [MaleCNS v1.0](https://male-cns.janelia.org/)
- [Official MaleCNS downloads and license](https://male-cns.janelia.org/download/)
- [MaleCNS supplemental optic-column assignments](https://github.com/flyconnectome/2025malecns)
- [Erginkaya et al. 2025 — H2/HS recurrent optic-flow network](https://doi.org/10.1038/s41593-025-01948-9)

## Release status

The repository is suitable for private canonical hosting and review. It is not
ready to be made public until the owner selects an explicit code license and the
private remote's CI, history, and uploaded tree are verified. Upstream MaleCNS
data remains governed by its own CC-BY license.
