# Contributing

This is an experimental scientific repository. Changes should preserve a clear
boundary between MaleCNS structure, literature-derived physiology, modeling
assumptions, free parameters, and engineering scaffolding.

Before opening a change:

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m compileall -q src scripts tests run_experiment.py run_exp002.py
.venv\Scripts\python.exe scripts\repository_policy_check.py
.venv\Scripts\python.exe scripts\verify_local_data.py --allow-missing
```

Scientific changes should receive a new experiment ID when they alter dynamics,
parameters, connectivity selection, stimulus semantics, or interpretation.
Never rewrite an earlier failed result into a success. Keep raw MaleCNS data,
credentials, environments, caches, and bulk generated outputs outside Git.

Before public release, the repository owner must choose and add an explicit code
license. MaleCNS upstream data remains governed by its own CC-BY license.
