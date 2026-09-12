# CNS project audit — 2026-09-12

## Scope and method

This audit treats source code, reachable Git history, local materialized inputs,
machine-readable records, generated outputs, tests, and the complete available
development transcript as separate evidence streams. The transcript establishes
intent and chronology; it is not accepted as implementation truth. Claims below
are accepted only where code, records, and reruns agree.

Audit base: `fbe1f055a9372eabcf5d4163a66d3141e76a1012` on `main`.

The audit performed the following read-only checks before repository changes:

- inspected all eight reachable commits and every tracked file;
- inspected the three untracked provisional EXP-004 files and ignored outputs;
- reran EXP-001, EXP-002, EXP-003, the EXP-003 downstream audit, and the
  bilateral EXP-003 pre-body experiment from their local frozen bundles;
- reran the eight EXP-002 held-out bundles without changing their inputs;
- reran the annotation reconciliation against the local MaleCNS v1.0 export;
- reproduced the Brian2 import failure in a fresh Python 3.11 / NumPy 2.4.6 /
  Brian2 2.9.0 environment and checked the current NumPy 2.0.2 environment;
- evaluated provisional EXP-004's effective zero-input transition matrix and
  electrical-coupling semantics;
- ran the full test suite, bytecode compilation, dependency consistency check,
  reachable-history secret scan, and source-working-tree secret scan;
- reviewed the local Prophet repository only for lightweight project-management
  patterns: implementation truth, explicit limits, deterministic checks, and a
  clean public/private boundary.

## Reconstructed chronology

| Commit | Historical action | Audited interpretation |
|---|---|---|
| `33c0ddc` | Initialized EXP-001 | Strict spike-gated LIF baseline. Signal stops before T4. |
| `2126cd4` | Added observability | Diagnostic extension to EXP-001, not a new biological experiment. |
| `59afc68` | Implemented EXP-002 | Graded local T4 proxy with exogenous delayed inhibitory drive. |
| `82fc335` | Recorded EXP-002 | Narrow model-internal order sensitivity recorded. |
| `7a3c620` | Validated EXP-002 | Frozen spatial sweep and a structurally matched but dynamically inert control. |
| `aeccf1c` | Implemented EXP-003 | FlyGym demonstration using an explicit T4a-minus-T4d actuator bridge. |
| `cad078b` | Audited downstream graph | Corrected a real T4b/T4c target-filter omission; steering sign still failed. |
| `fbe1f05` | Tested bilateral readout | Homologous DNp15 right-minus-left signal stayed same-signed; body gate not passed. |
| uncommitted | Provisional EXP-004 | Invalid implementation; not a biological result. |

## Claim/status ledger

Statuses have project-wide meanings:

- **SUPPORTED** — the narrow claim is implemented, measured, and reproduced.
- **UNSUPPORTED** — available evidence contradicts the stated claim.
- **INCONCLUSIVE** — the test cannot answer the biological question as posed.
- **INVALID_IMPLEMENTATION** — numerical or semantic defects invalidate interpretation.
- **SUPERSEDED** — a later implementation replaces an engineering choice without
  erasing the historical result.

| ID | Claim | Status | What survives the audit |
|---|---|---|---|
| DIAG-001 | The raw annotation table has 211,577 unique bodies and is not the published 166,691-neuron denominator. | **SUPPORTED** | Counts and status partitions reproduce. The exact publication predicate remains unavailable; no row/status shortcut is asserted. |
| DIAG-002 | Brian2 2.9.0 import fails because its installed `fundamentalunits.py` reads `np.ndarray.ptp`. | **SUPPORTED**, environment-specific | Reproduced with NumPy 2.4.6. It imports with NumPy 2.0.2, so “Brian2 fails with NumPy 2” is too broad. |
| EXP-001-A | The 137-node/356-edge strict-LIF run produces L1 spikes and subthreshold Mi1/Tm3 activity. | **SUPPORTED** | Rerun is byte-identical; all six Mi1/Tm3 cells respond, maximum absolute excursion 2.2600059509 mV. |
| EXP-001-B | EXP-001 evaluates T4 direction selectivity. | **INCONCLUSIVE** | T4 remains exactly at -52 mV and never spikes; the signal never reaches T4. |
| EXP-001-C | Commit `2126cd4` is a second biological experiment. | **UNSUPPORTED** | It changes recording/analysis only and is part of EXP-001. |
| EXP-002-A | The frozen model is temporally order-sensitive at every one of 43 selected T4 cells. | **SUPPORTED in model** | Primary metrics reproduce exactly. The output is dimensionless graded activity, not biological voltage. |
| EXP-002-B | The primary workbook-axis order raises mean T4a peak and reversal raises mean T4b peak. | **SUPPORTED in model** | Reproduced for the primary pair. Canonical visual direction is unresolved. |
| EXP-002-C | The result emerges from MaleCNS connectivity alone. | **UNSUPPORTED** | MaleCNS provides IDs, directed edges, and synapse counts. Signs, time constants, rectification, per-post normalization, inferred scalar coordinates, and delayed inhibitory visual drive are model assumptions. |
| EXP-002-D | Direct inhibitory-input ablation identifies the biological mechanism. | **INCONCLUSIVE** | Ablation reduces mean peak order contrast by 82.3%, but the inhibited cells are driven by an exogenous delayed spatial channel rather than propagated upstream anatomy. The control tests that modeled channel, not a fully materialized biological pathway. |
| EXP-002-E | The matched irrelevant-edge control establishes mechanism specificity. | **UNSUPPORTED as a matched dynamical control** | It adds/removes T4-to-downstream edges that `_build_edges` assigns zero modeled sign. Its exact no-effect result is true by construction. |
| EXP-002-F | Order sensitivity generalizes over sampled workbook positions/axes. | **SUPPORTED in model** | All eight frozen materializations reproduce their recorded mean order contrasts and ablation reductions. Each pair has a separately selected local graph, and workbook axes are not calibrated visual directions. |
| EXP-003-A | Reversing order changes embodied final yaw magnitude. | **SUPPORTED as engineering output** | 0.151196 vs 0.139958 rad reproduces; ablation makes both 0.113105 rad. |
| EXP-003-B | The model produces opposite direction-dependent biological steering. | **UNSUPPORTED** | Both conditions and the zero-command control turn in the same positive direction. |
| EXP-003-C | Steering sign is derived neutrally from downstream MaleCNS activity. | **INCONCLUSIVE / assumption-dominated** | The bridge explicitly assigns opposite output signs to connectivity-weighted T4a and T4d path mass. Downstream propagation is instantaneous, positive, and normalized; transmitter signs are not used. |
| EXP-003-D | The body run is meaningfully closed-loop visual simulation. | **INCONCLUSIVE** | Yaw shifts two abstract pulse times by a free 100 ms/rad transform; no rendered retinal scene feeds the visual circuit. |
| EXP-003-E | T4b/T4c absence was biological. | **UNSUPPORTED** | A restrictive H/VS target filter caused the omission. |
| EXP-003-F | Corrected target selection restores all four subtype paths. | **SUPPORTED anatomically** | Coverage becomes 12/12 T4a, 12/12 T4b, 13/13 T4c, and 6/6 T4d in the selected two-hop graph. |
| EXP-003-G | Corrected downstream coverage fixes steering sign. | **UNSUPPORTED** | Both open-loop commands remain predominantly negative; order changes magnitude/timing only. |
| EXP-003-H | A homologous bilateral DNp15 difference recovers opposite signs. | **UNSUPPORTED in the minimal model** | Both conditions produce a positive right-minus-left command area (~0.2316 ms). This does not falsify the biological binocular network. |
| EXP-004-P | The provisional H2/HS network tests yaw-vs-translation biology. | **INVALID_IMPLEMENTATION** | Electrical coupling is positive recurrence, not a state difference; generic H2-to-all-HS pairs are added; the effective LPTC transition radius is 1.00125; zero-input perturbations grow; yaw/translation ratio is exactly 1.0. No biological interpretation is valid. |

## Provenance boundary by experiment

### MaleCNS structural facts

- neuron/body IDs, annotations, sides, types, directed chemical edges, and
  synapse counts in each materialized bundle;
- optic-column L1 assignments from the v1.0 supplemental workbook;
- exact bilateral DNp15 and DNa02 type/side matches in EXP-003-bilateral.

MaleCNS does not provide the model's receptor signs, membrane/graded dynamics,
electrical coupling, controller actions, or workbook-axis-to-world geometry.

### Literature-supported physiology

- graded early visual signaling and receptor-dependent L1 sign inversion;
- qualitatively distinct Mi1/Tm3 timing;
- excitatory and inhibitory T4 input classes and spatially offset timing motifs;
- HS/H2 recurrent inhibitory binocular processing and enhanced DNp15
  rotation-versus-translation selectivity;
- a specific contralateral H2–HSE electrical interaction, not generic
  all-to-all H2–HS coupling.

The existing EXP-002 numerical time constants and delay are not direct fits to
published measurements. The provisional EXP-004 does not use the published
quantitative response target and cannot validate the cited physiology.

### Model assumptions and free parameters

- EXP-001 transmitter-sign table and LIF operating point;
- EXP-002 scalar coordinate inference, per-post normalization, rectification,
  exogenous inhibitory drive, all time constants, and the 15 ms delay;
- EXP-003 instantaneous downstream readout, T4a-minus-T4d bridge, saturation,
  controller gain, baseline gait, and yaw-to-pulse-time transform;
- every provisional EXP-004 time constant, coupling gain, recurrence scaling,
  and unsupported electrical pairing.

## Reproduction evidence

| Check | Audit result |
|---|---|
| EXP-001 fresh output | Metrics byte-identical; T4 excursion/difference 0.0 mV. |
| EXP-002 fresh output | Scientific JSON payload identical after excluding output-directory strings. |
| Eight EXP-002 held-out bundles | Mean contrasts and ablation reductions reproduce to floating-point precision. |
| EXP-003 body run | Scientific payload identical; final yaw values reproduce exactly. |
| EXP-003 downstream audit | Scientific payload identical; graph coverage and body values reproduce exactly. |
| EXP-003 bilateral | Scientific payload identical; same-sign pre-body failure reproduces exactly. |
| Annotation audit | 211,577 rows/unique IDs; recorded partitions reproduce. |
| Brian2 diagnostic | Failure reproduces with NumPy 2.4.6; import succeeds with NumPy 2.0.2. |
| Existing tests | 18 passed. |
| Compilation | `python -m compileall` passed. |
| Dependency consistency | `pip check` reported no broken requirements. |
| Reachable Git history secret scan | Gitleaks 8.30.1: 8 commits, no leaks. |
| Source working-tree secret scan | No leaks. Raw data/results were excluded from this scan and separately verified ignored. |

## Numerical findings

The provisional EXP-004 zero-input LPTC block is

```text
A = (1 - dt/tau) I + (dt/tau) (W_chemical + g W_electrical)
```

with raw feedback spectral radius 1.25 and effective transition spectral radius
1.00125. Starting from a deterministic LPTC perturbation, the norm changes from
0.1850 after one step to 0.6045 at 100 ms, 46,198 at 1,000 ms, and
2.32e26 at 5,000 ms. This is unstable despite the small Euler step.

For the electrical term, equal states produce a current vector of all ones
before gain. A diffusive conductance must give zero for equal states and oppose
pairwise differences. The implementation therefore fails both semantic and
stability preflight.

## Architecture findings

1. Historical implementations are small and inspectable, but experiment
   orchestration, plotting, model code, and record generation are tightly mixed.
2. EXP-003 imports private helpers from EXP-002, coupling a frozen historical
   experiment to later online execution.
3. Normalization appears in multiple modules with different scientific meaning
   and no shared invariant checks.
4. Materialized manifests identify query rules but omit content hashes and some
   software/query timestamps, so exact local bundles cannot be verified from Git.
5. Result records contain exact numbers but there were no concise human-readable
   per-experiment reports or committed evidence figures.
6. The README is stale: it describes EXP-002 as the next milestone and says body
   work has not started despite three committed EXP-003 stages.
7. Dependency ranges permit future behavior drift; the audited environment was
   not recorded in a machine-readable file.
8. Tests cover local mechanics but not record schema, bundle manifests,
   determinism, normalization invariants, or recurrent/electrical semantics.
9. Provisional EXP-004 source lives in active `src/`/`scripts/` paths despite
   failing preflight; it should be archived as invalid evidence.
10. Reachable commits expose `Eric Wang <99614113+EricW9888@users.noreply.github.com>` as author and
    committer. No remote exists, so one deliberate pre-publication normalization
    can remove that identity without rewriting an already shared repository.

## Ranked changes

1. **Archive and label provisional EXP-004 as `INVALID_IMPLEMENTATION`.** Keep
   exact source and compact failure evidence, but remove it from active modules.
2. **Establish the public claim boundary.** Add this ledger, concise experiment
   reports, an accurate README, and only small evidence figures.
3. **Make input provenance verifiable.** Track URLs, versions, sizes, SHA-256
   hashes, materialization commands, and manifest contracts—not raw MaleCNS data.
4. **Add scientific invariant tests.** Cover deterministic execution,
   normalization, transmitter behavior, record/manifest validation, diffusive
   coupling semantics, and effective-transition stability utilities.
5. **Add lightweight CI and secret scanning.** Do not run network queries,
   FlyGym, or large simulations on every commit.
6. **Record a tested environment.** Keep broad install requirements, add a
   machine-readable audited environment snapshot, and document optional FlyGym
   dependencies separately.
7. **Normalize Git identity once before first push.** Use the authenticated
   GitHub account's no-reply address and reconcile historical hash references.
8. **Create and validate a private canonical GitHub repository.** Push only
   tracked public artifacts; verify remote visibility and tree contents.
9. **Do not optimize yet.** Current local experiments complete in seconds to
   roughly one minute. Scientific validity, not runtime, is the blocking issue.
10. **Restart EXP-004 only after an anatomy/identity crosswalk and quantitative
    physiology target are explicit.** Use controlled HS/H2 inputs, exact
    literature-supported pairings, diffusive electrical coupling, and stability
    gates before interpreting any response.

## Publication blockers at audit time

- no explicit code license selected by the repository owner;
- stale README and absent human-readable experiment reports;
- missing tracked data hashes/provenance contract;
- personal Gmail in reachable Git metadata;
- no CI/secret scan on a remote;
- provisional EXP-004 active despite invalid semantics;
- EXP-002 biological language needs narrowing around the imposed inhibitory
  channel and unresolved visual geometry.

These blockers concern public release. They do not prevent creation of a private
canonical remote after the repository is cleaned and scanned.
