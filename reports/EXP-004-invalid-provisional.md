# Provisional EXP-004 — invalid implementation audit

**Status: INVALID_IMPLEMENTATION. This is not a biological result.**

## Intended question

The provisional code attempted to test whether a bilateral T4-derived HS/H2
network transformed upstream activity into stronger DNp15 discrimination of
yaw-like asymmetric flow than symmetric translation-like flow.

## Materialized structure

The ignored local bundle contains 175 nodes and 609 MaleCNS chemical edges:
92 T4 cells, 71 selected visual/intermediate cells, bilateral HSE/HSN/HSS/H2
(8 cells), and bilateral DNp15/DNa02 (4 cells). That materialization is an
auditable structural artifact, not validation of the dynamics or of a mapping
between unidentified intermediates and the named circuit in the physiology
paper.

## Fatal preflight failures

The code updates LPTC state toward

```text
T4 input + intermediate input + W_chemical x + g W_gap x
```

The last term is an additional positive recurrent synapse. Electrical
conductance should instead depend on paired state differences, producing zero
current at equal states and currents that reduce unequal states.

Observed preflight values:

| Check | Result |
|---|---:|
| nonzero entries in generic gap matrix | 12 |
| equal-state current norm before gain | 2.828427 |
| raw chemical-plus-gap feedback spectral radius | 1.250000 |
| effective Euler transition spectral radius | 1.001250 |
| perturbation norm after first step | 0.185021 |
| perturbation norm after 1,000 ms, zero input | 46,198.44 |
| perturbation norm after 5,000 ms, zero input | 2.32e26 |

The provisional gap matrix also connects contralateral H2 generically to HSE,
HSN, and HSS. The cited literature supports specific pairings and does not
justify this all-HS construction.

## Uninterpretable output

The generated DNp15 mean absolute bilateral-difference area is 7.884850 for
both yaw-like and translation-like stimulus groups (ratio effectively 1.0).
Because the network is unstable and electrical semantics are wrong, this is
neither positive nor negative biological evidence.

## Preservation and successor

The three previously uncommitted source files are retained under
`experiments/EXP-004-invalid-provisional/archive/` with their logic unchanged.
Generated traces remain ignored. A successor must start from controlled HS/H2
inputs and pass diffusive-coupling, provenance, effective-Jacobian stability,
zero-input decay, determinism, and quantitative physiology gates before any
biological interpretation.
