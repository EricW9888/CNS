# EXP-003 — bilateral DNp15 readout

**Status: UNSUPPORTED in the minimal model; biological binocular hypothesis INCONCLUSIVE.**

## Question

Does a homologous right-minus-left DNp15 population difference turn the two
mirrored binocular stimulus orders into opposite-signed pre-body commands?

## Circuit and readout

Left and right optic columns with matching workbook suffixes are materialized
independently through frozen EXP-002 and the corrected two-hop downstream rule.
The primary readout uses exact MaleCNS type/side homologs:

- DNp15 left body 12069 and right body 11215;
- secondary-only DNa02 left body 523769 and right body 10360.

The command is `DNp15_R - DNp15_L`; no subtype preferred-direction rule or
direct stimulus lookup is added. The sides are anatomically corresponding, but
their local materializations are not numerically symmetric (197 vs 187 visual
nodes; 374 vs 405 downstream nodes).

## Result

| Stimulus-space condition | Command area (ms) | Sign |
|---|---:|---|
| left forward / right reverse | 0.231559 | positive |
| left reverse / right forward | 0.231585 | positive |

The maximum pointwise difference between commands is 0.000928. Disconnecting
both T4-to-target pathways makes both commands exactly zero. The pre-body sign
gate fails, so embodiment is correctly skipped.

Two independent unilateral proxies followed by subtraction do not reproduce
the known nonlinear binocular transformation. This does not falsify biology:
the model omits HS/H2 cross-hemisphere electrical and recurrent inhibitory
interactions and still lacks calibrated workbook-to-visual geometry.

## Reproduce

```powershell
.venv\Scripts\python.exe scripts\query_exp003_bilateral.py
.venv\Scripts\python.exe scripts\run_exp003_bilateral.py
```

Exact machine state: `experiments/EXP-003-bilateral/record.json`.
