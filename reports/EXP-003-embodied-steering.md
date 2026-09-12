# EXP-003 — embodied steering scaffold

**Outcome: direction-dependent magnitude in an engineering scaffold, without
opposite or biologically interpretable steering.**

## Question

Can frozen EXP-002 activity feed a thin downstream readout and FlyGym controller
such that reversing the two-column order changes body yaw?

## Implementation boundary

The visual stage reuses frozen EXP-002. A two-hop MaleCNS graph is reduced to
instantaneous, nonnegative, per-post-normalized matrices. The temporary bridge
computes descending path mass reached from T4a minus path mass reached from T4d,
normalizes by a free activity reference of 0.01, clips to [-1, 1], and applies
opposite ±0.35 changes to FlyGym's left/right turning-controller actions around
a baseline of 1.0.

This is explicit actuator scaffolding. It assigns T4a and T4d opposite output
roles and does not use downstream transmitter signs or dynamics. The visual
“closed loop” shifts two pulse times by a free 100 ms/rad function of simulated
yaw; it is not rendered retinal optic flow.

## Result

An independent rerun exactly reproduces:

| Condition | Final yaw (rad) |
|---|---:|
| full forward | 0.151196 |
| full reverse | 0.139958 |
| path ablated forward | 0.113105 |
| path ablated reverse | 0.113105 |

The path-dependent order contrast is 0.011238 rad and disappears when all
T4-to-wide-field projection weights are removed. However, both conditions and
the zero-command control rotate in the same positive direction. The result
therefore establishes only that the artificial bridge can perturb an embodied
controller differently for the two frozen signals. It does not establish
opposite biological steering or a calibrated visual-to-yaw mapping.

## Reproduce

```powershell
.venv\Scripts\python.exe scripts\run_exp003.py
```

Exact machine state: `experiments/EXP-003-embodied-steering/record.json`.

References: [project references](../docs/references.md).
