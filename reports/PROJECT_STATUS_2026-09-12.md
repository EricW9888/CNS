# CNS project status — 2026-09-12

## Bottom line

CNS has one reproducible model-internal visual-motion result and several useful
negative/audit results. It does not yet have a biologically calibrated direction
result, validated yaw-selective DNp15 transformation, or defensible embodied
steering result.

## Evidence that survives

- EXP-001 correctly documents failure of strict spike-gated propagation before
  T4.
- EXP-002 reproducibly distinguishes opposite temporal orders in a frozen
  phenomenological T4 model across the primary pair and eight held-out local
  materializations.
- The EXP-003 downstream audit correctly identifies and fixes a target filter
  that excluded T4b/T4c paths.
- EXP-003 and its bilateral successor reproducibly fail to generate opposite
  steering signs. Those failures are retained.
- Annotation/body denominator and Brian2/NumPy 2.4.6 diagnostics reproduce with
  their narrowed scope.
- Provisional EXP-004 is invalid, archived, and excluded from active code.

## Current foundation

- six machine-readable records and six human-readable experiment reports;
- a complete cross-experiment claim ledger;
- one promoted EXP-002 evidence figure;
- SHA-256 registry for 49 source/materialization artifacts;
- graph-contract verification across 21 local bundles;
- reusable diffusive-coupling and effective-transition stability helpers;
- 29 fast deterministic tests, compilation, dependency, repository-policy, and
  secret-scan checks;
- lightweight GitHub CI that does not download data or run body simulations.

## Performance snapshot

Measured on the audited Windows/Python 3.11.9 environment:

| Task | Wall time |
|---|---:|
| 29-test suite | 1.36 s |
| EXP-001 with full observability/figures | 5.33 s |
| EXP-002 primary with traces/figures | 6.68 s |

Earlier independent audit runs completed EXP-003 body integration in about
28 seconds and the two-graph downstream comparison in about 52 seconds. These
times do not justify GPU/CUDA work. Scientific validity and data-flow clarity
are the current bottlenecks; no performance optimization was made.

## Public/private boundary

Tracked: code, tests, query logic, assumptions, records, reports, provenance
hashes, compact evidence, CI, and scientifically relevant failures.

Ignored/local: 1.7+ GB of upstream data, materialized Parquet bundles, bulk CSV
traces, videos, caches, environments, malformed duplicate output directories,
and the full AI transcript. None is a hidden dependency: exact public download,
query, and verification instructions are tracked.

## Next decision gate

Restart EXP-004 as a neural physiology experiment only. Use controlled HS/H2
inputs and published response classes before reconnecting unresolved T4 geometry.
Interpret nothing until exact neuron/circuit provenance, diffusive electrical
pairings, effective-transition stability, zero-input decay, deterministic output,
and quantitative external targets pass. Body work remains gated off.

The private canonical repository is verified at
[`EricW9888/CNS`](https://github.com/EricW9888/CNS): visibility is private,
local and server `main` matched at the transition checkpoint, the uploaded tree
contained no excluded artifacts, and both test CI and full-history secret scan
passed. Git author/committer history uses the authenticated account's no-reply
address; the old-to-canonical hash map is retained in the
[normalization ledger](../docs/history-normalization.md).

Public release remains blocked only by owner selection of an explicit code
license. Scientific uncertainty is not a release blocker when it is stated
honestly.

See the [claim ledger](PROJECT_AUDIT_2026-09-12.md),
[architecture](../docs/architecture.md), [reproducibility boundary](../docs/reproducibility.md),
and [references](../docs/references.md).
