# Implementation and performance

This note records the implementation choices tested on 2026-09-12. Scientific
results are documented in the experiment reports; the measurements here concern
correctness, representation, and execution cost.

## Subsystems examined

| Subsystem | Implemented boundary | Alternatives considered | Verification |
|---|---|---|---|
| MaleCNS ingestion | Unique body IDs, finite positive aggregate weights, explicit endpoint and schema checks | silently keeping the first duplicate; query-specific joins | malformed-input tests and exact reconstruction of the 137-, 187-, and 405-node local bundles |
| Stimulus compilation | additive event composition with clipped time windows | last-event-wins assignment | overlap, negative-window, invalid-duration, and order tests |
| EXP-001 LIF | CSR coupling, delayed event ring, selected voltage recording | dense coupling; all-node recording | discrete Euler invariant tests and byte-identical experiment CSVs |
| EXP-002 graded model | one shared batch/online kernel; indexed NumPy arrays; ring-buffered delay | repeated Pandas filtering; duplicated online implementation; precomputed dense operators | exact batch/online sample equality and byte-identical experiment traces/tables |
| Downstream projection | CSR operators with explicit source/target body-ID axes for successors | historical dense matrices; edge-list scatter accumulation | synthetic and real-bundle dense/sparse equivalence |
| Electrical coupling | sparse symmetric diffusive operator | positive recurrent adjacency input | equal-state zero current, sign reversal, conservation, and dissipative-energy tests |
| Comparison/provenance | exact neuron/time-axis contracts and fail-fast graph manifests | silent inner joins/intersections and duplicate dropping | missing-axis, duplicate, hash, endpoint, stage-count, and edge-count tests |
| Output recording | preallocated selected-state arrays | whole-network long-form Pandas traces | selection/shape tests and memory estimates |

## Correctness changes

- Events that fall partly or entirely before the simulation window no longer
  use negative NumPy slices, and overlapping events sum instead of overwriting
  one another.
- Duplicate node/transmitter identities and duplicate aggregate query edges are
  rejected instead of selecting an arbitrary first row.
- Missing T4 neurons or time samples now fail forward/reverse comparison instead
  of disappearing through an intersection or inner join.
- Nonfinite parameters, nonpositive time constants, invalid thresholds,
  nonpositive synapse counts, and graph endpoints outside the node table are
  rejected at their boundaries.
- Electrical coupling is represented by a graph Laplacian current whose paired
  terms are proportional to `other_state - self_state`.
- EXP-002 batch and online execution now call the same numerical kernel, removing
  two implementations that could drift.

None of these changes alters the default EXP-001 or frozen EXP-002 numerical
outputs. EXP-001's scientific JSON payload and all numerical CSVs reproduce
exactly. EXP-002's four full traces and both result tables are byte-identical.

## Measured performance

Measurements used Python 3.11.9 on the project Windows environment. cProfile
includes instrumentation overhead; microbenchmarks report the median of seven
warm runs. They are representative comparisons, not hardware-independent
performance claims.

| Workload | Before | After |
|---|---:|---:|
| Four EXP-002 `run_graded` calls under cProfile | 3.912 s | 0.314 s |
| Four coordinate-inference calls | 2.428 s | 0.004 s |
| Full EXP-002 runner, including six CSVs and two figures | 10.192 s | 7.831 s |
| 187 nodes / 958 edges / 1,501 steps: edge scatter vs CSR | 0.00336 s | 0.00279 s |
| 50,000 nodes / 500,000 edges / 20 steps: edge scatter vs CSR | 0.0546 s | 0.00903 s |
| Corrected EXP-003 operator storage: dense vs CSR | 293,728 B | 20,248 B |

The synthetic CSR comparisons differ from scatter accumulation by at most
5.33e-15, consistent with floating-point summation order. The real corrected
EXP-003 dense/CSR output differs by at most 3.33e-16.

Plotting and CSV conversion now dominate the EXP-002 entry point. They are
analysis outputs, not the simulation kernel, so low-level optimization there is
not currently justified.

Run the benchmark locally with:

```powershell
.venv\Scripts\python.exe scripts\benchmark_foundation.py
```

## Whole-CNS path

A 166,691-neuron, 1,501-step float32 trace would occupy about 1.00 GB before
labels or DataFrame overhead; 500 selected probes require about 3.00 MB. The
successor boundary therefore keeps state in contiguous arrays, uses CSR
operators, and records named probes.

Remaining limitations are explicit:

- historical EXP-003 propagation still uses dense small-circuit matrices and
  Python table assembly;
- EXP-002 still uses `np.add.at` in its per-step edge accumulation because CSR
  gives only a modest gain at 187 nodes;
- data ingestion still uses Pandas/Parquet and may need chunked Arrow or
  database-side projection for whole-connectome materialization;
- the state loop is single-threaded CPU Python around vectorized kernels;
- no GPU backend, automatic differentiation, or distributed execution exists;
- numerical efficiency does not resolve the biological assumptions in EXP-002
  or the missing validated EXP-004 physiology.

The next backend decision should be based on a stable, scientifically specified
EXP-004 kernel and a representative large materialization. A GPU port before
that point would optimize an unsettled model.
