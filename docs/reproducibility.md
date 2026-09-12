# Reproducibility and evidence boundary

## Three artifact classes

1. **Tracked records and reports** preserve exact parameter/result summaries,
   interpretation, failures, and compact public figures.
2. **Ignored materializations** contain query-derived Parquet graphs. Their
   source URLs, query commands, sizes, and SHA-256 hashes are tracked in
   `data/manifest.json`.
3. **Ignored generated results** contain bulk CSV traces, videos, caches, and
   intermediate figures. They are reproducible outputs, not hidden inputs.

No experiment requires an API token stored in the repository. Materialization
uses the public MaleCNS neuPrint endpoint. If access policy changes, credentials
must be supplied outside Git; no neuPrint credential may be committed.

## Local verification

```powershell
.venv\Scripts\python.exe scripts\verify_local_data.py
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m compileall -q src scripts tests run_experiment.py run_exp002.py
.venv\Scripts\python.exe -m pip check
```

The data verifier hashes all registered local files and validates every local
graph with adjacent `manifest.json`, `nodes.parquet`, and `edges.parquet` files:
unique node IDs, valid endpoints, nonnegative numeric synapse counts, exact
node/edge counts, and exact stage counts.

## Determinism

The current NumPy experiment paths are deterministic. EXP-003 passes seed zero
to FlyGym/MuJoCo. The 2026-09-12 audit reproduced EXP-001 byte-for-byte and
reproduced the scientific JSON payloads of EXP-002 and all EXP-003 variants
exactly after excluding output-directory strings.

The committed tests prefer invariants over arbitrary full-output snapshots.
Exact historical metrics remain in experiment records and are independently
checked during audits, not on every CI commit because CI intentionally excludes
raw MaleCNS data and expensive simulations.

## Frozen history versus successors

Historical model modules are not silently corrected when doing so would change
an existing experiment. A scientifically changed model becomes a successor
experiment. Shared infrastructure may be added around historical code, but the
records state which commit and assumptions produced each result.

The archived provisional EXP-004 is not importable active code and must not be
used as a successor baseline.
