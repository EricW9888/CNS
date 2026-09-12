# Local MaleCNS data

Raw MaleCNS files and materialized graph bundles are intentionally ignored.
They are upstream data, generated artifacts, or both; none is redistributed by
this repository. MaleCNS v1.0 is licensed CC-BY by Janelia.

`manifest.json` records the exact source URLs, local file sizes, SHA-256 hashes,
and hashes of graph bundles used for the recorded experiments. Download the
three small source files with the commands in the project README, then run the
materialization commands documented in each experiment report.

Verify all locally present registered files and graph contracts with:

```powershell
.venv\Scripts\python.exe scripts\verify_local_data.py
```

Use `--allow-missing` when checking a partial installation. Verification does
not contact neuPrint and never reads credentials.
