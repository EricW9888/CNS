# MaleCNS v1.0 prototype: technical note

> Historical EXP-001 note. For the audited cross-experiment state, see
> `reports/PROJECT_AUDIT_2026-09-12.md` and the reports under `reports/`.

Date: 2026-09-10

## Narrow question

Can two adjacent optic-column L1 inputs, delivered in opposite temporal
orders, produce a measurable difference through the MaleCNS
`L1 -> Mi1/Tm3 -> T4a-d` pathway and its small visual-to-descending extension?

This is a connectivity-and-dynamics test, not a claim of brain emulation or
behavior.

## Data and provenance

The official download page supplies body annotations, body-level transmitter
predictions, the full weighted graph, and related synapse tables. The official
supplemental workbook supplies left/right optic-column assignments, including
L1 IDs. The first run uses the workbook plus public neuPrint queries against
`male-cns:v1.0` so the 1.1 GB full graph is not needed during initial
validation. Each materialized edge retains its real pre-body, post-body, and
synapse count.

## Stimulus and pathway

The selected columns are `ME_R_col_01_07` and `ME_R_col_01_08`, which differ by
one coordinate in the workbook grid. The two primary conditions are:

```text
01_07 at 10 ms, then 01_08 at 20 ms
01_08 at 10 ms, then 01_07 at 20 ms
```

Each event drives the selected column’s L1 body ID. The query stages are:

```text
L1 -> Mi1/Tm3 -> T4a/T4b/T4c/T4d
   -> HSE/HSN/HSS/HST/VS/VST1/VST2/VSm
   -> descending_neuron
```

The script rejects non-adjacent suffixes. The names preserve workbook order;
they do not claim a calibrated leftward/rightward retinal direction. A whole-
eye pulse is not the primary experiment and is not required for this run.

The positive input current is an explicit L1 activity abstraction, not a
photoreceptor light-ON model. L1 is glutamatergic and the exact sign of an
individual postsynaptic response is receptor/circuit dependent. The strict
baseline therefore maps acetylcholine to positive, GABA/glutamate to negative,
and other transmitter labels to unknown/zero. This is a stated model
assumption, not a claim that a body-level label resolves all receptor effects.

## Annotation reconciliation

`body-annotations-male-cns-v1.0-minconf-0.5.feather` has 211,577 unique body
IDs, with no duplicate body IDs. It is a body-annotation table, not a biological
neuron table. Its status partition is:

| Raw status | Rows | Treatment in this prototype |
|---|---:|---|
| Traced | 165,122 | Not globally equated with the paper count; only path-selected IDs are used |
| Orphan | 15,925 | Not selected by the curated L1-to-output queries |
| Glia | 11,864 | Excluded from this neuronal pathway |
| Unimportant | 10,751 | Excluded from this neuronal pathway |
| status null | 5,472 | Excluded unless explicitly returned as a path node |
| Assign | 1,832 | Not treated as final curated neurons |
| Anchor | 611 | Not treated as a global neuron denominator |

The table also has 44,877 null `superclass`, 47,071 null `type`, and 50,071
null `instance` values. Thus neither row count, unique ID count, status alone,
nor type presence is a safe substitute for the published denominator.

The publication’s 166,691 figure is the curated “identified, proofread and
annotated” neuron count (including sensory axons). This audit does not invent a
predicate that maps the current v1.0 body table to that historical denominator.
As a separate schema check, the current public neuPrint `Neuron` label returns
176,422 nodes. That is another materialized release/schema view, not proof that
176,422 equals the publication count. For this experiment, the included set is
explicit and auditable: 2 selected L1 IDs, 6 reachable Mi1/Tm3 nodes, 43
reachable T4 nodes, 11 selected wide-field projection nodes, and 75 queried
descending-neuron targets, for 137 nodes total. Every other raw annotation/body
record is excluded from the simulation.

The reproducible field-level audit is:

```powershell
.venv\Scripts\python.exe scripts/audit_malecns_annotations.py
```

## Dynamics and Brian2 diagnostic

The intended baseline parameters are from the published Shiu et al. style
whole-brain LIF model where applicable: rest/reset `-52 mV`, threshold `-45
mV`, membrane time constant `20 ms`, synaptic time constant `5 ms`, refractory
period `2.2 ms`, delay `1.8 ms`, and `0.275 mV` per effective synapse.

Brian2 2.9.0 was tested in the project Python 3.11.9 environment with NumPy
2.4.6. Import fails before model construction because the installed file
`brian2/units/fundamentalunits.py`, class `Quantity`, line 1661, evaluates
`np.ndarray.ptp`. NumPy 2 removed that ndarray method. The complete traceback,
version record, and upstream comparison are in
[`docs/diagnostics/brian2_numpy2.md`](docs/diagnostics/brian2_numpy2.md).

Brian2 2.7 release notes document NumPy-2 compatibility, while current upstream
development source guards the optional `ptp` attribute. The released wheel and
the documented behavior disagree in this environment. NumPy is therefore not
downgraded. The runnable experiment uses the same equations in a small
NumPy/SciPy sparse integrator, with real edge weights and explicit delayed
events.

## Controls and limitations

The forward-order T4-to-wide-field ablation removes only real queried T4 to
HSE/HS/VS edges. A meaningful positive future result would require L1 spikes,
Mi1/Tm3 and T4 responses, a difference under order reversal, and a change under
that ablation. The current model has no tonic drive, receptor-specific glutamate
kinetics, photoreceptor model, or motor/behavioral decoder; a null downstream
response is therefore an honest limitation of this narrow baseline.

FlyGym, full-body movement, framework abstractions, training, and optimization
are deliberately out of scope until this pathway-level test is biologically
calibrated and reproducible.

## Observability pass

The experiment now captures the pre-reset membrane voltage at every existing
integration step for the six Mi1/Tm3 nodes and 43 T4 nodes in both temporal
orders. It does not alter the integration loop, stimulus, node set, signs, or
parameters. The analysis writes raw long-form voltage CSVs, a T4
forward-minus-reverse CSV/JSON table, a pathway graph, grouped spike rasters,
Mi1/Tm3 and T4 voltage plots, a ranked T4 difference plot, and a three-panel
forward/reverse/difference T4 heatmap under
`results/first_experiment/figures/`.

This makes the current null diagnosable: if Mi1/Tm3 move below rest but do not
spike, no synaptic event is emitted into T4 in this LIF implementation. T4 can
therefore remain exactly at rest, yielding no order-dependent T4 voltage
difference; that is distinct from having tested and rejected temporal-order
sensitivity at an active T4 operating point.
