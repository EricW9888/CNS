# CNS experimental progress — 2026-09-13

## Current state

CNS has one reproducible model-internal visual-motion result and several
informative negative results. It does not yet have a biologically calibrated
direction result, a validated yaw-selective DNp15 transformation, or a
defensible embodied steering result.

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

## Reproducibility foundation

The repository contains machine-readable records and human-readable reports for
each experiment, compact EXP-002 and EXP-004 figures, SHA-256 provenance for 62
source/materialization artifacts, and graph contracts for 22 local bundles.
Fast tests cover model invariants, exact batch/online equivalence, event
composition, materialization integrity, sparse projection semantics, diffusive
coupling, stability helpers, result comparisons, and repository policy.

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

EXP-004's resolved neural implementation has stable, deterministic dynamics,
fixed physiological comparisons and complete mechanistic controls. The next
scientific requirement is an independently constrained explanation for its
H2/H2rn/DNp15 mismatches and the increased common-mode response introduced by
the full network. The current result is preserved without retuning. It does
not justify reconnecting unresolved T4 workbook geometry or proceeding to body
simulation.

Body simulation remains out of scope until that neural gate succeeds.

## Repository state

The canonical GitHub repository remains private. The tracked tree contains no
downloaded MaleCNS data, generated bulk results, environments, caches, or
credentials. CI runs tests, compilation, dependency checks, data-manifest
validation, repository policy, and a full-history secret scan.

Project code and original documentation now have an explicit
[Apache-2.0 license](../LICENSE); upstream data retains its separate terms.
The repository remains private pending an explicit publication decision.
Scientific uncertainty is documented in the experiment reports rather than
treated as a reason to hide negative results.

See the [architecture](../docs/architecture.md), [reproducibility
boundary](../docs/reproducibility.md), [implementation
notes](../docs/implementation.md), and [references](../docs/references.md).
