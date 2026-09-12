# Architecture and scientific boundaries

## Data flow

```text
official MaleCNS files + public neuPrint
        |
        v
validated query/materialization -> ignored Parquet graph bundles
        |
        v
compiled numerical state        -> selected recordings and summaries
        |                                      |
        +--> machine records                   +--> ignored bulk outputs
        +--> human reports                     +--> selected public evidence
```

### Anatomy and materialization

`src/malecns_io.py` loads graph tables and conservative body-level transmitter
predictions. `src/materialization.py` validates aggregate chemical edge rows
and joins complete, unique node metadata. Query scripts select bounded MaleCNS
v1.0 subgraphs. Duplicate bodies or aggregate edge pairs, missing requested
annotations, invalid endpoints, and nonpositive/nonfinite synapse counts fail
explicitly.

The source of truth for structural claims is the ignored materialized bundle
verified against `data/manifest.json`, not a copied edge list in prose.

### Recorded model implementations

- `src/lif_model.py`: EXP-001 strict spike-gated LIF baseline.
- `src/graded_model.py`: EXP-002 phenomenological graded local T4 model.
- `src/exp003.py`, `src/exp003_corrected.py`, and
  `src/exp003_bilateral.py`: explicit downstream and embodiment scaffolds.

The EXP-002 batch and online paths now share `GradedCircuitKernel`; exact trace
regressions ensure the refactor does not alter the frozen experiment. EXP-003's
small dense readouts remain in place because changing their representation
would obscure historical reproduction.

### Successor-model foundation

`src/sparse_backend.py` compiles edge tables once into CSR projections with
explicit source and target body-ID axes. It also records only requested states,
avoiding an all-neuron time-series allocation. `src/numerics.py` provides
normalization, validated diffusive electrical coupling, effective Euler
transitions, spectral radius, and zero-input decay checks.

These utilities do not define biological dynamics by themselves. A successor
experiment must still specify equations, units, connection provenance, and
free parameters.

### Records and evidence

- `experiments/*/record.json`: exact historical machine state;
- `reports/*.md`: question, method, result, controls, and limitations;
- `figures/`: selected compact evidence;
- `results/`: ignored generated output;
- `data/`: ignored source/materialized data except the tracked registry.

## Rules for successor experiments

1. Declare structural facts, literature physiology, model assumptions, and free
   parameters separately.
2. Freeze stimulus selection and parameters before held-out validation.
3. Gate recurrent interpretation on effective-transition stability and explicit
   zero-input decay.
4. Keep electrical conductances separate from chemical edges and test their
   state-difference semantics.
5. Compare against quantitative external physiology where available; a nonzero
   output alone is not validation.
6. Treat ablations as mechanistic only when they remove an active, biologically
   relevant component and are compared with a genuinely effective control.
7. Give changed science a new experiment ID; do not silently repair history.

## Scaling direction

Tables are an inspection and interchange format, not the inner-loop state
representation. Successor models should compile immutable body-ID mappings and
sparse operators once, update contiguous arrays, record selected probes, and
convert to labeled tables only at analysis boundaries.

Current CPU/SciPy performance is sufficient for these circuits. Whole-CNS work
will require batched sparse state updates and bounded recording, but not
necessarily a GPU. Backend specialization should follow equation validation and
representative profiling rather than precede them.
