# CNS experimental progress — 2026-09-14

## Current state

CNS has a reproducible, provisional continuous sensory-to-DNp15 chain and an
identified neck-motor extension with a causal open-loop physical head effect,
alongside local visual-motion results and informative negative results. CvNA2
activation now has an extracted quantitative movement target, but DNp15-to-CvNA
recruitment and posture-conditioned motor calibration remain unresolved.
Measured bilateral compound-eye sampling now connects a controlled 3D world to
the frozen sensory chain through a provisional central-local readout. The
broader local-direction representation does not yet pass translation
refinement, and individual measured-eye-to-MaleCNS registration is unresolved.
It does not yet have a biologically calibrated direction result, a validated
recurrent yaw-selective DNp15 transformation, calibrated head control, or
biologically interpretable embodied steering or closed-loop behavior.

## Experiment progression

- EXP-001 shows that the strict spike-gated baseline stops at subthreshold
  Mi1/Tm3 activity; T4 remains silent.
- EXP-002 distinguishes opposite neighboring-column orders in a frozen
  phenomenological T4 model across the primary pair and eight additional local
  materializations.
- The EXP-003 downstream correction identifies a restrictive target filter
  that excluded T4b/T4c paths. Restoring all four subtypes does not recover
  steering sign.
- EXP-003 and its bilateral successor both fail to generate opposite steering
  signs. These are retained negative model results.
- Annotation/body-denominator and Brian2/NumPy diagnostics retain their
  appropriately narrow technical conclusions.
- The provisional EXP-004 recurrent implementation failed numerical and
  semantic preflight. Its outputs are excluded from biological interpretation.
- The [controlled EXP-004 successor](EXP-004-binocular-physiology.md) resolves
  bIPS/PS321 and release-aware H2rn identities, and passes stability and
  reproducibility checks. DNp15 shows partial qualitative magnitude agreement,
  but the feed-forward control is more selective than the recurrent network;
  the experimentally motivated recurrent enhancement is not reproduced.
- [EXP-005](EXP-005-sensory-to-DNp15.md) connects explicit binocular luminance
  stimuli through provisional ON/OFF motion channels, identified MaleCNS T4/T5
  projections, LPi and HS/H2, and the frozen central circuit to both DNp15 cells.
  Static images produce zero activity and visual disconnection abolishes
  descending activity. The limited transformation comparison passes, but the
  feed-forward central control remains more selective. The first quantitative
  external discrepancy occurs at H2; successful propagation is not validation
  of early-vision transduction or the published recurrent mechanism.
- [EXP-006](EXP-006-DNp15-neck-motor.md) extends that frozen chain through four
  direct DNp15-to-CvNA1/CvNA2 connections, with verified MaleCNS/MANC identities,
  cervical-output sides and TH1/TH2 muscle correspondence. A literature-informed,
  uncalibrated torque approximation produces small, opposite-signed open-loop
  head-azimuth changes. Disconnection controls establish causality within the
  implementation, not calibrated muscle physiology, whole-body yaw or walking
  steering.
- [EXP-007](EXP-007-CvNA2-motor-boundary.md) extracts a quantitative CvNA2 movement
  target from primary activation curves: right-axon-standardized population
  mean yaw is positive and pitch negative, with explicit digitization envelopes
  rather than biological confidence intervals. The source contains 677
  validation trials from 11 flies. Identified DNp15-to-CvNA recruitment and
  CvNA2-specific posture-conditioned calibration remain unidentified; direct
  motor activation cannot supply the missing DN recruitment transfer.
  Saved EXP-006 motor controls replay exactly without changing the bridge or
  adding a body run.
- [EXP-008](EXP-008-compound-eye.md) establishes a measured compound-eye sensory
  boundary: 857 left / 852 right facet directions sample controlled 3D scenes.
  A provisional 65 L / 66 R central-local motion readout propagates to both
  DNp15 cells. The observer camera never supplies neural input; this is not
  individual facet-to-MaleCNS registration or full-eye retinotopy.
- [EXP-009](EXP-009-retinotopic-eye.md) introduces a broader 1,500-neighborhood
  measured-grid local-direction representation. Translation refinement errors
  are 6.98% / 9.27%, above the predeclared 5% criterion. Successful propagation
  does not validate that interface.
- [EXP-010](EXP-010-retinotopic-refinement.md) preserves the fixed 1 ms-to-0.5 ms
  successor result: +X improves to 4.95%, while −X remains 6.06%. The unchanged
  5% criterion fails, so the refined candidate is not validated downstream.
  In the original EXP-009 discrepancy, 99.4% / 86.3% of local error energy is
  in neighborhoods whose acceptance cones can intersect the foreground blocker.
  Blocker removal nearly eliminates the absolute hotspots; higher optical
  quadrature reduces them, while a common lens origin does not. These diagnostics
  support occlusion / optical sampling sensitivity at a hard visibility boundary,
  not a biological failure or downstream-network instability.
  Same-specimen male optic-lobe tables establish released body/column assignments
  and Mi1–T4 correspondences, but do not establish individual measured-eye-to-
  MaleCNS registration. That biological blocker is independent of refinement.

## Reproducibility foundation

The repository contains machine-readable records and human-readable reports for
each experiment, compact figures for EXP-002 and EXP-004 through EXP-010,
SHA-256 provenance for 94 source/materialization artifacts, and graph contracts
for 24 local bundles.
Fast tests cover model invariants, exact batch/online equivalence, event
composition, materialization integrity, sparse projection semantics, diffusive
coupling, stability helpers, result comparisons, and repository policy.
The [registry](../experiments/registry.json) provides canonical verification
entrypoints, and [machine evidence](../experiments/evidence.json) ties selected
results to record fields and tests. The [experiment workflow](../docs/experiment-workflow.md)
defines predeclaration, local validation, controls, replay and freezing without
a workflow engine.

Downloaded data, local graph bundles, generated traces, videos, caches,
environments, and credentials remain outside Git. The tracked manifest provides
the source URLs, hashes, and graph counts needed to verify local evidence.

## Implementation progress

The EXP-002 numerical core now compiles graph metadata once, uses indexed arrays
for coordinate inference and edge construction, shares one state-update kernel
between batch and online execution, and constructs output tables without
per-sample Python dictionaries. Exact historical traces and result tables are
unchanged.

A separate sparse backend provides explicit body-ID axes, CSR projections, and
selected-state recording for successor experiments. Historical EXP-003 dense
readouts remain unchanged because they define recorded experiments.

Representative 2026-09-12 measurements on Windows/Python 3.11.9:

| Workload | Previous | Current | Observation |
|---|---:|---:|---|
| Four EXP-002 numerical runs under cProfile | 3.912 s | 0.314 s | 12.5x faster; exact output tables |
| Four EXP-002 coordinate-inference calls | 2.428 s | 0.004 s | repeated DataFrame scans removed |
| Full EXP-002 run including six CSVs and two figures | 10.192 s | 7.831 s | rendering and serialization now dominate |
| 50,000-node/500,000-edge synthetic accumulation | 0.0546 s scatter | 0.00903 s CSR | 6.04x faster; max difference 5.33e-15 |
| Corrected EXP-003 downstream propagation | 293,728 B dense | 20,248 B CSR | 14.5x less operator storage |

Wall-clock rendering measurements vary across runs and are not treated as
scientific benchmarks. See [implementation and
performance](../docs/implementation.md) for methods and remaining limits.

## Current direction

EXP-005 through EXP-010 are frozen evidence. EXP-008 establishes the measured
3D-world-to-compound-eye boundary; EXP-009/010 leave broader local-direction
sampling numerically unvalidated for translation. The sensory frontier is
resolving optical sampling sensitivity at the blocker boundary and, separately,
individual measured-eye-to-MaleCNS registration. The earlier same-specimen
optic-lobe release supplies body/column evidence, not the missing coordinate
registration chain. No further candidate was chosen to cross the 5% threshold.
EXP-004's H2/H2rn/DNp15 mismatches and failure to reproduce recurrent enhancement
remain unresolved; upstream integration does not remove those limitations.

EXP-007 establishes why EXP-006's motor interface cannot yet be calibrated.
Motor calibration awaits identified DNp15-to-CvNA recruitment measurements,
CvNA2-specific activation/firing calibration, and the authors' CvNA2 trial/fit
subset with starting poses and held-out assignments. The movement subset can
constrain posture dependence but cannot resolve DN recruitment by itself.
The provisional open-loop physical test has proceeded; calibrated head control,
sensory-loop closure and biological steering claims remain unsupported.

## Repository state

The tracked tree contains no downloaded MaleCNS data, generated bulk results,
environments, caches, or credentials. CI runs tests, compilation, dependency
checks, data-manifest validation, repository policy, and a full-history secret
scan.

Project code and original documentation now have an explicit
[Apache-2.0 license](../LICENSE); upstream data retains its separate terms.
Scientific uncertainty is documented in the experiment reports rather than
treated as a reason to hide negative results.

See the [architecture](../docs/architecture.md), [reproducibility
boundary](../docs/reproducibility.md), [implementation
notes](../docs/implementation.md), and [references](../docs/references.md).
