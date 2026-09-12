# Git history normalization

Before the repository had any remote, its 14 existing commits were rewritten
once to replace a personal author/committer address with the authenticated
GitHub account's private no-reply address. Historical file contents containing
that address were normalized in the same operation. No model, result, data,
timestamp, commit message, or file content unrelated to that address changed.

This ledger keeps old local notes and transcript references reconstructable.
The canonical hashes are the right-hand values.

| Legacy local commit | Canonical commit | Subject |
|---|---|---|
| `33c0ddc` | `2292f84` | Initialize narrow MaleCNS motion prototype |
| `2126cd4` | `b222654` | Add motion-pathway observability diagnostics |
| `59afc68` | `5dc96b4` | Add graded EXP-002 local motion circuit |
| `82fc335` | `16006a7` | Record EXP-002 direction-selectivity result |
| `7a3c620` | `004ba7f` | Validate frozen EXP-002 spatial direction selectivity |
| `aeccf1c` | `64e29b1` | Start EXP-003 embodied MaleCNS steering |
| `cad078b` | `529d42f` | Audit and extend EXP-003 downstream pathway |
| `fbe1f05` | `af26f03` | Test bilateral EXP-003 optic-flow readout |
| `b6d8adb` | `c33cf4b` | Audit experiment claims and preserve scientific record |
| `b8a5e16` | `93c70ff` | Archive invalid provisional EXP-004 implementation |
| `2c2be8d` | `f64b0ae` | Add numerical and provenance invariant checks |
| `5fa2be7` | `bd7cfa1` | Record verifiable data and environment provenance |
| `9e23f5c` | `686de34` | Establish audited project structure and lightweight CI |
| `cc30f11` | `646e9fe` | Add current project status and reference ledger |

The pre-normalization repository is retained only as a local recovery bundle
outside the working tree. It must not be uploaded or treated as the canonical
history.
