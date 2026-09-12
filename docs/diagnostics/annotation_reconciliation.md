# MaleCNS annotation-body reconciliation

Recorded 2026-09-10. Source: local official
`body-annotations-male-cns-v1.0-minconf-0.5.feather` plus a read-only query to
the public `male-cns:v1.0` neuPrint dataset.

## Counts observed

| Object/schema view | Count | Interpretation |
|---|---:|---|
| Raw Feather rows | 211,577 | Annotation/body records |
| Unique raw `bodyId` values | 211,577 | No duplicate IDs in this file |
| Raw `status=Traced` | 165,122 | A curation status, not the publication denominator |
| neuPrint `Neuron` label | 176,422 | Current database materialization |
| Published count | 166,691 | Curated, proofread/annotated neurons, including sensory axons |

The raw row count is therefore 44,886 above the published count. The current
neuPrint `Neuron` label is 9,731 above it. These are different schema views and
neither difference is safely closed by deleting arbitrary rows.

## Raw status partition

The raw body table partitions as follows:

| `status` | Rows | Treatment here |
|---|---:|---|
| `Traced` | 165,122 | Not automatically promoted to the paper denominator |
| `Orphan` | 15,925 | Not selected by the path query |
| `Glia` | 11,864 | Excluded from the neuronal pathway |
| `Unimportant` | 10,751 | Excluded from the neuronal pathway |
| null | 5,472 | Not automatically biological neurons |
| `Assign` | 1,832 | Intermediate assignment status; not final here |
| `Anchor` | 611 | Anchor records; not automatically counted as ordinary neurons |

These categories sum to 211,577. Relevant null/quality fields are:

* `superclass` null: 44,877
* `class` null: 185,064
* `type` null: 47,071
* `instance` null: 50,071

The raw file has 2,605 `status=Traced` records with null `type`; conversely,
1,985 status-null records have a non-null `type`. This is why a type/status
intersection also cannot be silently declared the published neuron set.

## What this experiment includes

The simulation does not load all raw rows. The query helper includes only IDs
returned by these explicit stages:

1. the two selected right-eye L1 IDs (`94444`, `65625`);
2. their queried `Mi1`/`Tm3` postsynaptic targets (6 nodes);
3. their queried `T4a`/`T4b`/`T4c`/`T4d` postsynaptic targets (43 nodes);
4. queried `HSE`/`HSN`/`HSS`/`HST`/`VS`/`VST1`/`VST2`/`VSm` targets (11 nodes);
5. queried `superclass="descending_neuron"` targets (75 nodes).

The final materialized graph is 137 nodes and 356 directed edges. All other
raw annotation/body records—including glia, unimportant, orphan, assignment,
anchor, status-null, unrelated neurons, and bodies outside the queried path—
are excluded from the simulation. The manifest records the exact IDs and query
stages.

## Conclusion

The evidence supports saying “211,577 raw body-annotation records” and
“166,691 published curated neurons,” not treating those as interchangeable.
The current export/API does not expose a single verified predicate in this
reconciliation that reproduces 166,691 exactly. That unresolved release/schema drift is
recorded rather than hidden behind a made-up filter.
