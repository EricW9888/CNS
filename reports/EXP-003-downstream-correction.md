# EXP-003 — downstream materialization correction

**Outcome: T4b/T4c paths were missing because of a target-filter error;
restoring them does not recover steering sign.**

## Question

Why did the original EXP-003 graph contain downstream projections from selected
T4a and T4d cells but none from T4b or T4c?

## Materialization correction

The original query admitted only HSE/HSN/HSS/HST/VS/VST1/VST2/VSm as direct
T4 targets. That boundary excluded valid visual targets. The omission was not
caused by subtype naming, side selection, query depth, or a biological absence.

The corrected two-hop selection admits MaleCNS targets whose superclass is
`visual_projection` or `visual_centrifugal`, plus LPi types, then retains
their connections to `descending_neuron`. The graph grows from 137 nodes and
356 edges to 405 nodes and 1,566 edges.

| Type | Selected T4 | Direct target | Two-hop DN path |
|---|---:|---:|---:|
| T4a | 12 | 12 | 12 |
| T4b | 12 | 12 | 12 |
| T4c | 13 | 13 | 13 |
| T4d | 6 | 6 | 6 |

Added target classes include LPi, LPLC, LLPC, LPC, LPT, VS/VSm, OLVC, Nod,
VCH/V1, and vCal cells. The machine record contains the exact type list and
counts.

## Frozen comparison

The original EXP-003 model and bridge were not retuned. The corrected graph's
forward/reverse command areas are -0.629978 and -0.629920 ms. Both remain
negative. Final yaw is positive in both conditions (0.153141 and 0.156693 rad),
and the yaw order contrast shrinks from 0.011238 to 0.003552 rad. Projection
ablation removes the remaining contrast.

The correction restores subtype coverage but does not rescue steering sign.
The frozen bridge still uses only T4a-minus-T4d path weights; it does not assign
new output signs to T4b or T4c.

## Limitations

This is a graph-materialization correction and a dimensionless activity-proxy
comparison, not evidence of canonical left/right or front/back steering. The
workbook-to-visual direction mapping remains unresolved, and additional
LPi-mediated integration was not added.

## Reproduce

```powershell
.venv\Scripts\python.exe scripts\query_exp003_corrected_downstream.py
.venv\Scripts\python.exe scripts\run_exp003_downstream_correction.py
```

Exact machine state:
`experiments/EXP-003-downstream-correction/record.json`.

References: [project references](../docs/references.md).
