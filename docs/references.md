# Scientific and data references

## MaleCNS

- Berg et al. (2025), *Sexual dimorphism in the complete connectome of the
  Drosophila male central nervous system*. DOI:
  [10.1101/2025.10.09.680999](https://doi.org/10.1101/2025.10.09.680999).
- [MaleCNS v1.0 portal](https://male-cns.janelia.org/) and
  [official download/license page](https://male-cns.janelia.org/download/).
- [Official supplemental data repository](https://github.com/flyconnectome/2025malecns),
  including the optic-column assignment workbook.

## Early visual motion and EXP-002 physiology

- Yang et al. (2016), *Subcellular imaging of voltage and calcium signals
  reveals neural processing in vivo*. The project record uses the associated
  Mi1/Tm3 timing evidence via
  [PMC4243710](https://pmc.ncbi.nlm.nih.gov/articles/PMC4243710/).
- Shinomiya et al. (2019), receptor/sign context for the L1 ON pathway,
  [eLife 49373](https://elifesciences.org/articles/49373).
- Takemura et al. (2017), *The comprehensive connectome of a neural substrate
  for ON motion detection in Drosophila*,
  [eLife 24394](https://elifesciences.org/articles/24394).
- Gruntman et al. (2018), *Simple integration of fast excitation and offset,
  delayed inhibition computes directional selectivity in Drosophila*,
  [Nature Neuroscience / PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5967973/).

These sources motivate qualitative rules. They do not directly determine the
exact EXP-002 time constants, delay, inferred coordinates, or normalization.

## Bilateral HS/H2 and DNp15

- Farrow, Haag and Borst (2006), *Nonlinear, binocular interactions underlying flow
  field selectivity of a motion-sensitive neuron*,
  [PMID 16964250](https://pubmed.ncbi.nlm.nih.gov/16964250/). This blowfly study
  supports a specific contralateral H2–HSE electrical interaction, not generic
  H2-to-all-HS coupling. Species transfer requires separate justification.
- Pokusaeva et al. (2024), *Bilateral interactions of optic-flow sensitive
  neurons coordinate course control in flies*,
  [Nature Communications 15:8830](https://doi.org/10.1038/s41467-024-53173-w).
- Erginkaya et al. (2025), *A competitive disinhibitory network for robust optic
  flow processing in Drosophila*,
  [Nature Neuroscience 28:1241–1255](https://doi.org/10.1038/s41593-025-01948-9),
  with [public analysis code](https://github.com/ChiappeLab/Erginkaya_et_al_2025).

The 2025 work identifies HS/H2 inputs, recurrent GABAergic intermediates, and a
DNp15 transformation that is more selective for rotational versus translational
optic flow. Its quantitative data/code should define the external target for a
new EXP-004; it should not be used as a hidden fit after seeing model output.
