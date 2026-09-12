# EXP-004 — provisional recurrent-model preflight failure

**Outcome: the implementation fails electrical-coupling and stability checks;
its generated responses are not a biological result.**

## Intended question

The provisional code attempted to test whether a bilateral T4-derived HS/H2
network transformed upstream activity into stronger DNp15 discrimination of
yaw-like asymmetric flow than symmetric translation-like flow.

## Materialized structure

The ignored local bundle contains 175 nodes and 609 MaleCNS chemical edges:
92 T4 cells, 71 selected visual/intermediate cells, bilateral HSE/HSN/HSS/H2
(8 cells), and bilateral DNp15/DNa02 (4 cells). This remains a structural
materialization only; it does not validate the dynamics or identify the
unnamed intermediates with the cells in the physiology paper.

## Preflight failures

The implementation updates LPTC state toward

```text
T4 input + intermediate input + W_chemical x + g W_gap x
```

The last term is an additional positive recurrent synapse. Electrical
conductance should instead depend on paired state differences, producing zero
current at equal states and currents that reduce unequal states.

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
Because the network is unstable and the electrical semantics are wrong, this is
neither positive nor negative biological evidence.

## Preservation and successor

The three source files are retained under
`experiments/EXP-004-preflight-failure/archive/` with the failed numerical
logic unchanged. They are not active modules or successor code. Generated
traces remain ignored.

A new implementation must start from controlled HS/H2 inputs and pass
interaction-provenance, diffusive-coupling, effective-transition stability,
zero-input decay, determinism, and quantitative physiology gates before any
biological interpretation.

References: [project references](../docs/references.md).
