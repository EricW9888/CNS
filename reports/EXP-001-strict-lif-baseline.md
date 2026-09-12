# EXP-001 — strict-LIF adjacent-column baseline

**Status: INCONCLUSIVE for T4 direction selectivity; implementation result SUPPORTED.**

## Question

Can opposite temporal orders of two adjacent MaleCNS optic-column L1 inputs
produce an order-dependent T4 response in a strict spike-gated LIF network?

## Anatomy and protocol

MaleCNS v1.0 supplies the selected right-eye L1 IDs, directed chemical edges,
cell annotations, and synapse counts. The materialized graph has 137 nodes and
356 edges: 2 L1, 6 Mi1/Tm3, 43 T4, 11 selected wide-field cells, and 75
descending cells. Workbook-adjacent columns `01_07` and `01_08` receive 10 ms,
20 mV events at 10 and 20 ms; the reverse condition swaps their order.

The model is a deterministic sparse LIF system:

```text
dv/dt = (v_rest - v + g + I_ext) / tau_m
dg/dt = -g / tau_syn
```

Presynaptic spikes add delayed conductance proportional to MaleCNS synapse
count. Body-level transmitter labels map acetylcholine positive, GABA and
glutamate negative, and unknown labels to zero. That table is a model
assumption, not receptor-resolved physiology.

## Result

- each driven L1 emits one spike;
- all six Mi1/Tm3 cells have measurable subthreshold responses;
- maximum absolute Mi1/Tm3 excursion from rest: 2.2600059509 mV;
- every T4 cell remains exactly at -52 mV and emits no spike;
- forward-minus-reverse T4 voltage difference: 0.0 mV.

Commit `2126cd4` added full traces and figures without changing the model or
creating a second experiment. The fresh audit rerun produced byte-identical
metrics.

## Interpretation

The strict spike gate stops the signal at subthreshold Mi1/Tm3 cells, so T4
direction selectivity is not tested. This is an informative propagation and
operating-point failure, not evidence that biological T4 cells lack direction
selectivity. The T4-to-wide-field ablation is inert because T4 is already silent.

## Reproduce

```powershell
.venv\Scripts\python.exe scripts\query_malecns_subgraph.py --eye right --suffixes 01_07 01_08
.venv\Scripts\python.exe run_experiment.py
```

Exact machine state: `experiments/EXP-001-adjacent-column-motion/record.json`.

References: [project reference ledger](../docs/references.md).
