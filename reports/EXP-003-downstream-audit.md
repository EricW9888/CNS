# EXP-003 downstream materialization audit

**Status: SUPPORTED anatomical correction; steering hypothesis remains UNSUPPORTED.**

## Question

Why did the original EXP-003 graph contain downstream projections from selected
T4a and T4d cells but none from T4b or T4c?

## Finding

The original query admitted only HSE/HSN/HSS/HST/VS/VST1/VST2/VSm as direct
T4 targets. The omission was a target-filter bug, not a subtype naming, side,
query-depth, or biological-absence result.

The corrected two-hop selection admits MaleCNS targets whose superclass is
`visual_projection` or `visual_centrifugal`, plus LPi types, then retains their
connections to `descending_neuron`. The graph grows from 137 nodes/356 edges to
405 nodes/1,566 edges. Direct-target and two-hop descending coverage becomes:

| Type | Selected T4 | Direct target | Two-hop DN path |
|---|---:|---:|---:|
| T4a | 12 | 12 | 12 |
| T4b | 12 | 12 | 12 |
| T4c | 13 | 13 | 13 |
| T4d | 6 | 6 | 6 |

Added target classes include LPi, LPLC, LLPC, LPC, LPT, VS/VSm, OLVC, Nod,
VCH/V1, and vCal cells. The exact list and counts are in the machine record.

## Frozen comparison

The original EXP-003 model and bridge were not retuned. The corrected graph's
forward/reverse command areas are -0.629978 and -0.629920 ms. Both stay
negative. Final yaw remains positive in both conditions (0.153141 and 0.156693
rad), and the yaw order contrast shrinks from 0.011238 to 0.003552 rad.
Projection ablation removes the remaining contrast.

The anatomical correction survives. It does not rescue steering sign, and the
frozen bridge still uses only T4a-minus-T4d path weights rather than assigning
new roles to T4b/T4c.

## Reproduce

```powershell
.venv\Scripts\python.exe scripts\query_exp003_corrected_downstream.py
.venv\Scripts\python.exe scripts\run_exp003_downstream_audit.py
```

Exact machine state: `experiments/EXP-003-downstream-audit/record.json`.
