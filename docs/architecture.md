# Architecture and scientific boundaries

## Current layers

```text
official MaleCNS files + public neuPrint
        |
        v
query/materialization scripts -> ignored nodes/edges/manifest bundles
        |
        v
historical model modules -> experiment runners -> ignored bulk outputs
        |                                      |
        +--> machine records                   +--> selected public figure
        +--> human reports
```

### Anatomy/materialization

`src/malecns_io.py` loads graph tables and conservative body-level transmitter
predictions. Query scripts select bounded MaleCNS v1.0 subgraphs. The source of
truth for structural claims is the ignored materialized bundle verified against
`data/manifest.json`, not a copied edge list in prose.

### Historical model implementations

- `src/lif_model.py`: EXP-001 strict spike-gated LIF baseline.
- `src/graded_model.py`: EXP-002 phenomenological graded local T4 model.
- `src/exp003.py`, `src/exp003_corrected.py`, and
  `src/exp003_bilateral.py`: explicit downstream and embodiment scaffolds.

These modules remain inspectable and frozen where changing them would alter a
recorded experiment. Their limitations are part of their scientific identity.
Later code currently imports some private EXP-002 helpers; that coupling is
accepted for historical reproducibility but should not be copied into a new
model family.

### Reusable foundation

`src/numerics.py` defines model-independent normalization, diffusive electrical
coupling, effective Euler transitions, spectral radius, and zero-input decay
checks. `src/provenance.py` validates hashes and materialization contracts.
These utilities do not change any historical experiment.

### Records and evidence

- `experiments/*/record.json`: exact historical machine state;
- `reports/*.md`: audited scientific interpretation;
- `reports/PROJECT_AUDIT_2026-09-12.md`: cross-experiment claim ledger;
- `figures/`: only small deliberately promoted public evidence;
- `results/`: ignored bulk output;
- `data/`: ignored upstream/materialized data except its public registry.

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
   relevant component and are compared with a genuinely effective matched
   control.
7. Give changed science a new experiment ID; do not silently repair history.

## Scaling direction

The existing graph boundary—columnar tables transformed into explicit sparse or
dense operators—is compatible with larger MaleCNS materializations. Before
whole-CNS work, model equations and units must be validated on small circuits,
operators must avoid dense all-pairs allocation, and profiling must identify the
actual bottleneck. GPU/CUDA specialization is not justified by current runtimes
or scientific maturity.
