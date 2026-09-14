# Experiment workflow

Keep one active scientific frontier:
`build/experiment → result → critique → next boundary`.
Infrastructure follows concrete repeated friction, not speculative generality.

The default lifecycle is:
`question → evidence/preflight → frozen specification → local validation → integrated run → controls → convergence/replay → record/report/figure → freeze`.

## Before execution

State one question and identify the frozen predecessor. Separate measured
facts, literature constraints, assumptions, free parameters and unknowns.
Declare success/failure criteria, numerical tolerances, controls and a valid
negative result before examining final DNp15 or body outputs.
The optional [successor template](../experiments/successor-template.json)
contains empty contracts only: fill them with evidence, not generated conclusions.
It is not a canonical record or a runnable model.

Validate the new boundary locally before integration. Frozen upstream stages
remain unchanged; never select upstream parameters by final downstream output.
Use measured biology where available and predeclared invariants otherwise.
Keep stage-level observations. Stop at an evidence boundary or name a narrow
replaceable approximation explicitly; do not silently invent missing physiology.

## Run and preserve

Run cheap preflights and the minimum falsification conditions before expensive
execution. Record deterministic state/seeds. Distinguish production from
verification: replay saved sensory light when it is sufficient, rather than
rerendering or requerying the world.

Write exploratory output under ignored `results/` or local data paths. Inspect
the scientific diff before record generation. A canonical record may be created
only after its declared generation gate; runners must refuse to overwrite it.
A failed scientific criterion is a result, not permission to move a threshold.
Resolve acceptance explicitly as positive, negative or incomplete; distinguish
implementation/materialization failure from model/biological disagreement.
Changes after freeze require a successor or an explicit documented correction
that states which earlier conclusions change and which remain intact.

Promote only the specification, compact record, concise report and selected
recorded-data figure/media. Bulk traces, caches and downloads stay ignored.
Figures are derived views, not another source of truth. Public visualizations
are observer-only and cannot feed the model.

## Verify and freeze

The [registry](../experiments/registry.json) lists every experiment and its
canonical check. [Machine evidence](../experiments/evidence.json) ties selected
current headline statements, including false/incomplete ones, to exact fields
and tests. Update both with the new result, without rewriting predecessors.

```powershell
.venv\Scripts\python.exe scripts\verify_experiment.py EXP-010-retinotopic-refinement
# Partial-install/CI integrity only; explicitly does not claim model replay:
.venv\Scripts\python.exe scripts\verify_experiment.py --all --integrity-only
```

Default verification checks source hashes and frozen artifacts, runs declared
targeted tests, delegates exact record replay where supported, then rechecks
artifacts. EXP-008/009/010 consume saved light; small original-neighborhood
diagnostic rays in EXP-010 are still regenerated. Earlier experiments without
an exact-record runner use explicitly labeled historical artifact/invariant
verification: this does not claim a fresh body replay.
The registry's frozen commit pins the current version of its record/figure;
original execution commits remain in historical records.

Freeze only after controls/acceptance and convergence/stability have resolved,
required replay/provenance passes, report limitations match the record, the
selected figure is inspected, and repository/security checks pass. Negative
results can freeze; unsuccessful implementation outputs cannot become biology.
Commit the coherent scientific result and require a clean tree before push.
No opportunistic cleanup belongs in an experiment commit. A downstream stage
may consume frozen evidence but must not silently modify it.
