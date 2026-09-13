"""Resolve identities, extract external targets, and export the EXP-004 anatomy.

This preparatory path does not import or execute the neural model. Downloaded
source tables and queried chemical edges stay under ignored data/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import urllib.request

import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.query_malecns_subgraph import DATASET, NEUPRINT_URL, query
from src.materialization import canonical_edge_frame, materialize_nodes
from src.malecns_io import load_annotations, load_transmitters
from src.provenance import sha256_file, validate_materialization

PAPER = "https://doi.org/10.1038/s41593-025-01948-9"
SUPPLEMENT = "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41593-025-01948-9/MediaObjects/41593_2025_1948_MOESM1_ESM.pdf"
COUPLING_PAPER = "https://doi.org/10.1038/s41467-024-53173-w"
SOURCE_BASE = "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41593-025-01948-9/MediaObjects/"
SOURCE_URLS = {
    "source_fig1.xlsx": SOURCE_BASE + "41593_2025_1948_MOESM6_ESM.xlsx",
    "source_fig3.xlsx": SOURCE_BASE + "41593_2025_1948_MOESM8_ESM.xlsx",
    "source_fig4.xlsx": SOURCE_BASE + "41593_2025_1948_MOESM9_ESM.xlsx",
    "supplement.pdf": SUPPLEMENT,
    "flywire_annotations.tsv": "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/8587524c1748ce5ef2080822a2fc890fc03bf597/supplemental_files/Supplemental_file1_neuron_annotations.tsv",
    "flywire_annotations_v630.tsv": "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/df6bb136f5b3d91c3992df4e8de2642329e2a384/supplemental_files/Supplemental_file1_annotations.tsv",
    "flywire_columns_v630.md": "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/df6bb136f5b3d91c3992df4e8de2642329e2a384/supplemental_files/Supplemental_files_columns.md",
    "flywire_columns_v783.md": "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/8587524c1748ce5ef2080822a2fc890fc03bf597/supplemental_files/README.md",
    "PaperFiguresImaging.m": "https://raw.githubusercontent.com/ChiappeLab/Erginkaya_et_al_2025/b1b59653a9758335cfff28a81d42010ea6ff082a/scripts/matlab/PaperFiguresImaging.m",
}
# Published Supplementary Table 2, entries 26, 28-31. These exact FlyWire
# roots, not anatomical similarity or shared connectivity, establish the
# crosswalk to released FlyWire type labels and MaleCNS flywireType.
H2RN_ROOTS = [
    "720575940621865991", "720575940622082451", "720575940624756836",
    "720575940617566769", "720575940642076576",
]
SPEC_PATH = ROOT / "experiments/EXP-004-binocular-physiology/specification.json"


def source_statistics(path: Path, sheet: str) -> dict:
    """Read the first DI block without treating blanks/p-values as samples."""
    frame = pd.read_excel(path, sheet_name=sheet, header=None)
    if frame.iloc[0, 0] != "bFtoB-Yaw Discrimination index":
        raise ValueError("unexpected physiological target header")
    result = {}
    for row in frame.iloc[1:].itertuples(index=False, name=None):
        if pd.isna(row[0]) or str(row[0]).startswith("p"):
            break
        values = np.asarray([float(x) for x in row[1:] if pd.notna(x)])
        if len(values) < 2 or not np.isfinite(values).all():
            raise ValueError("invalid ROI measurements")
        half = float(t.ppf(.975, len(values)-1) * values.std(ddof=1) / np.sqrt(len(values)))
        result[str(row[0])] = {
            "n_rois": len(values), "mean_di": float(values.mean()),
            "descriptive_roi_t95_interval": [float(values.mean()-half), float(values.mean()+half)],
            "source_sheet": sheet, "source_excel_row": len(result)+2,
        }
    return result


def resolve_h2rn_roots(old: pd.DataFrame, current: pd.DataFrame) -> list[dict]:
    """Release 630 root -> identical anchor supervoxel -> released 783 type.

    Both table schemas document the anchor-supervoxel identifier. This is
    an exact released identifier join, not a soma-distance, morphology,
    connectivity or desired-response match. Missing/ambiguous joins stop.
    """
    roots = old[old.root_id.astype(str).isin(H2RN_ROOTS)]
    if len(roots) != len(H2RN_ROOTS) or roots.root_id.duplicated().any():
        raise ValueError("published H2rn roots do not resolve uniquely in release 630")
    matched = roots[["root_id", "supervoxel_id"]].merge(
        current[["root_id", "supervoxel_id", "cell_type"]], on="supervoxel_id",
        how="left", suffixes=("_630", "_783"), validate="one_to_one")
    if matched[["root_id_783", "cell_type"]].isna().any().any() or matched.root_id_783.duplicated().any():
        raise ValueError("missing/ambiguous released H2rn anchor crosswalk")
    return matched.sort_values("root_id_630").to_dict(orient="records")


def resolve_populations(annotations: pd.DataFrame, h2rn_crosswalk: list[dict]) -> pd.DataFrame:
    h2rn_types = {row["cell_type"] for row in h2rn_crosswalk}
    # bIPS: supplement entry 21 hb1932497911; MaleCNS PS321 instances
    # additionally identify the contralateral hb1963869636 counterpart.
    known = annotations[annotations.type.isin(["HSE", "HSN", "HSS", "H2", "DNp15", "PS321"])]
    ulpt = annotations[annotations.hemibrainType.fillna("").isin([f"PS{i:03}" for i in range(70,75)])]
    h2rn = annotations[annotations.flywireType.fillna("").isin(h2rn_types)]
    nodes = pd.concat([known, ulpt, h2rn]).copy()
    if nodes.bodyId.duplicated().any() or nodes.somaSide.isna().any():
        raise ValueError("ambiguous population identity/side")
    nodes["population"] = "HS"
    nodes.loc[nodes.type.eq("H2"), "population"] = "H2"
    nodes.loc[nodes.type.eq("PS321"), "population"] = "bIPS"
    nodes.loc[nodes.bodyId.isin(ulpt.bodyId), "population"] = "uLPTCrn"
    nodes.loc[nodes.bodyId.isin(h2rn.bodyId), "population"] = "H2rn"
    nodes.loc[nodes.type.eq("DNp15"), "population"] = "DNp15"
    for population in ["HS", "H2", "bIPS", "uLPTCrn", "H2rn", "DNp15"]:
        selected = nodes[nodes.population.eq(population)]
        if set(selected.somaSide) != {"L", "R"}:
            raise ValueError(f"missing bilateral {population}")
    return nodes.sort_values("bodyId").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/exp004_physiology")
    parser.add_argument("--output", type=Path, default=ROOT / "data/malecns_exp004_physiology")
    parser.add_argument("--check-specification", action="store_true", help="verify rather than overwrite the frozen specification")
    parser.add_argument("--download-sources", action="store_true", help="download missing primary-source files into ignored data/")
    args = parser.parse_args()
    if args.download_sources:
        args.source.mkdir(parents=True, exist_ok=True)
        for name, url in SOURCE_URLS.items():
            destination = args.source / name
            if not destination.exists():
                with urllib.request.urlopen(url, timeout=120) as response:
                    destination.write_bytes(response.read())
    for name, expected_bytes in [("flywire_annotations.tsv", 31718505), ("flywire_annotations_v630.tsv", 21714061)]:
        if (args.source / name).stat().st_size != expected_bytes:
            raise ValueError(f"incomplete source table: {name}; expected pinned Git blob size {expected_bytes}")
    annotations = load_annotations(ROOT / "data/body-annotations.feather")
    transmitters = load_transmitters(ROOT / "data/body-neurotransmitters.feather")
    flywire = pd.read_csv(args.source / "flywire_annotations.tsv", sep="\t", dtype=str)
    flywire_old = pd.read_csv(args.source / "flywire_annotations_v630.tsv", sep="\t", dtype=str)
    h2rn_crosswalk = resolve_h2rn_roots(flywire_old, flywire)
    resolved = resolve_populations(annotations, h2rn_crosswalk)
    ids = resolved.bodyId.astype(int).tolist()
    bips = resolved[resolved.population.eq("bIPS")]
    if set(bips.bodyId) != {11919, 12072}:
        raise ValueError("bIPS identity does not match the primary hemibrain crosswalk")
    for body_id, hemibrain_id in [(11919, "1963869636"), (12072, "1932497911")]:
        if hemibrain_id not in str(bips.set_index("bodyId").loc[body_id, "instance"]):
            raise ValueError("bIPS instance lacks the paper's hemibrain identity")
    # Independent unrestricted neighborhood, not a target-type-filtered
    # traversal. Retain even partners excluded from the modeled induced graph.
    neighborhood = []
    for endpoint in ["a", "b"]:
        neighborhood.extend(query(
            f"MATCH (a:Neuron)-[c:ConnectsTo]->(b:Neuron) WHERE {endpoint}.bodyId IN [11919,12072] "
            "RETURN a.bodyId AS pre_body,b.bodyId AS post_body,c.weight AS synapse_count,"
            "a.type AS pre_type,b.type AS post_type ORDER BY pre_body,post_body"))
    neighborhood_frame = pd.DataFrame(neighborhood)
    if neighborhood_frame.groupby(["pre_body", "post_body"]).synapse_count.nunique().max() != 1:
        raise ValueError("unrestricted bIPS queries disagree")
    neighborhood_frame = neighborhood_frame.drop_duplicates(["pre_body", "post_body"]).sort_values(["pre_body", "post_body"])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "bips_neighborhood.json").write_text(neighborhood_frame.to_json(orient="records", indent=2)+"\n", encoding="utf8")
    cypher = (
        f"MATCH (a:Neuron)-[c:ConnectsTo]->(b:Neuron) WHERE a.bodyId IN {ids} AND b.bodyId IN {ids} "
        "RETURN a.bodyId AS pre_body,b.bodyId AS post_body,c.weight AS synapse_count "
        "ORDER BY pre_body,post_body"
    )
    edges = canonical_edge_frame(query(cypher))
    retained_neighborhood = neighborhood_frame[neighborhood_frame.pre_body.isin(ids) & neighborhood_frame.post_body.isin(ids)]
    checked = edges.merge(retained_neighborhood[["pre_body", "post_body", "synapse_count"]],
                          on=["pre_body", "post_body"], how="outer", suffixes=("", "_unrestricted"), indicator=True)
    relevant = checked[checked.pre_body.isin([11919,12072]) | checked.post_body.isin([11919,12072])]
    if not relevant._merge.eq("both").all() or not np.array_equal(relevant.synapse_count,relevant.synapse_count_unrestricted):
        raise ValueError("induced bIPS edges disagree with the unrestricted query")
    upstream_ids = resolved[resolved.population.isin(["HS", "H2"])].bodyId.astype(int).tolist()
    roi_rows = query(f"MATCH (n:Neuron) WHERE n.bodyId IN {upstream_ids} RETURN n.bodyId AS bodyId,n.roiInfo AS roiInfo ORDER BY bodyId")
    roi_validation = []
    for row in roi_rows:
        roi = json.loads(row["roiInfo"])
        side = resolved.set_index("bodyId").loc[row["bodyId"], "somaSide"]
        lop = {s: int(roi.get(f"LOP({s})", {}).get("post", 0)) for s in ["L", "R"]}
        if lop[side] <= lop[{"L": "R", "R": "L"}[side]] or lop[side] == 0:
            raise ValueError("upstream eye does not match dominant LOP postsynaptic side")
        roi_validation.append({"bodyId": row["bodyId"], "controlled_input_eye": side, "LOP_post": lop})
    if len(roi_validation) != len(upstream_ids):
        raise ValueError("missing upstream eye verification")
    population = resolved.set_index("bodyId").population.to_dict()
    nodes = materialize_nodes(ids, stages=population, annotations=annotations, transmitters=transmitters)
    nodes = nodes.sort_values("bodyId").reset_index(drop=True)
    # Retain official correspondence fields, including uncertainty/many-to-one
    # class matching. Never assign an individual physiological ROI to a body.
    nodes = nodes.merge(resolved[["bodyId", "hemibrainType", "flywireType"]], on="bodyId", validate="one_to_one")
    for row in nodes.itertuples(index=False):
        expected = "acetylcholine" if row.stage in {"HS", "H2", "DNp15"} else "gaba"
        if row.consensus_nt != expected:
            raise ValueError(f"unresolved transmitter/sign for {row.bodyId}")
    manifest = {
        "dataset": DATASET, "release": "MaleCNS v1.0", "neuprint_endpoint": NEUPRINT_URL,
        "node_count": len(nodes), "edge_count": len(edges),
        "stage_counts": nodes.stage.value_counts().sort_index().to_dict(),
        "query": cypher, "chemical_graph_only": True,
        "selection": "induced graph of explicitly crosswalked populations; all positive chemical counts; no synapse threshold, extra hop, electrical edge, T4/T5 or DNa02",
    }
    validate_materialization(manifest, nodes, edges)
    args.output.mkdir(parents=True, exist_ok=True)
    nodes.to_parquet(args.output / "nodes.parquet", index=False)
    edges.to_parquet(args.output / "edges.parquet", index=False)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf8")
    anatomy_files = [{"path": str(path.relative_to(ROOT)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
                     for path in [args.output / name for name in ["manifest.json", "nodes.parquet", "edges.parquet", "bips_neighborhood.json"]]]
    target = source_statistics(args.source / "source_fig3.xlsx", "3e")
    bips_ablation_target = source_statistics(args.source / "source_fig4.xlsx", "4f")
    fig1 = source_statistics(args.source / "source_fig1.xlsx", "1h")
    for population in ["HS", "DNp15"]:
        for field in ["n_rois", "mean_di", "descriptive_roi_t95_interval"]:
            if target[population][field] != fig1[population][field]:
                raise ValueError("physiological source figures disagree")
    sources = [{"path": str((args.source / name).relative_to(ROOT)).replace("\\", "/"), "url": url,
                "bytes": (args.source/name).stat().st_size, "sha256": sha256_file(args.source/name), "redistributed": False}
               for name, url in SOURCE_URLS.items()]
    gap_pairs = []
    for side, opposite in [("L", "R"), ("R", "L")]:
        def body(kind, cell_side):
            match = nodes[nodes.type.eq(kind) & nodes.somaSide.eq(cell_side)]
            if len(match) != 1:
                raise ValueError("electrical pairing is not individually resolved")
            return int(match.iloc[0].bodyId)
        gap_pairs.extend([
            {"first": body("HSE", side), "second": body("H2", opposite),
             "provenance": COUPLING_PAPER, "evidence": "Drosophila Fig. 7/dye-coupling text: HSE with contralateral H2; classic direct physiology in blowfly DOI 10.1038/nn1769"},
            {"first": body("HSN", side), "second": body("DNp15", side),
             "provenance": COUPLING_PAPER, "evidence": "Drosophila Fig. 7/dye-coupling text: HSN with DNp15; direct two-node conductance is a modeling approximation, not measured conductance"},
        ])
    spec = {
        "experiment_id": "EXP-004-binocular-physiology", "stage": "controlled HS/H2 input; neural only",
        "external_target": {"paper": PAPER, "source_fig1": "1h", "source_fig3": "3e",
            "di_formula": "(OF_translation - OF_yaw) / (abs(OF_translation) + abs(OF_yaw))",
            "response_window_ms": [3500, 4000], "zero_denominator": "undefined (JSON null), not zero selectivity",
            "observed": target, "published_hs_dnp15_rank_sum_p": 0.00076248,
            "bIPS_GABA_receptor_control": bips_ablation_target,
            "published_bips_rdl_rank_sum_p": 0.00051284,
            "comparison": "report per-body signed DI then population mean against the published ROI means. The primary acceptance uses the separate observation_contract magnitude, polarity, matched-upstream and recurrent-mechanism requirements; signed DI alone cannot pass.",
            "quantitative_check": "report error relative to source ROI means and descriptive ROI t95 intervals; these are not independent-fly confidence intervals or calibrated model acceptance thresholds",
            "measurement_limit": "female-fly z-scored calcium versus male anatomy/dimensionless state proxy; no voltage/calcium scale or noise model fit"},
        "observation_contract": {
            "primary_transform": "C_proxy(t) = x(t) - mean_pre_stimulus(x); baseline is exactly zero here. A shared unit positive linear gain, no fitted class gains, convolution, clipping, z-scoring or condition normalization. This is an explicit uncalibrated late-response calcium proxy, not a prediction of fluorescence or membrane voltage.",
            "justification_and_limit": "Published OF/DI use baseline-referenced calcium late in a 2-s motion epoch. A common positive gain cancels in DI and translation/yaw ratios. The identity transform is a modeling assumption: no measured GCaMP transfer function is available for these modeled states. Quantitative absolute amplitudes, onset kinetics and noise are therefore not validated.",
            "primary_window_ms": [3500, 4000],
            "sample_convention": "pre-update samples; [3500,4000) ms, analogue of author code strict 3.5<t<4 s on 12.2-Hz calcium frames",
            "signed_DI": "preserve published signed OF and DI conventions, including negative responses; never select a modeled neuron's preferred direction post hoc",
            "magnitude_check": "also report (abs(OF_translation)-abs(OF_yaw))/(abs(OF_translation)+abs(OF_yaw)); signed DI alone cannot pass",
            "matched_upstream": "for each DNp15 side, use its ipsilateral three HS cells plus contralateral-dendritic-eye H2. RMS each unit's late OF contrasts before pooling, so opposing translation signs cannot cancel. Use the same observation transform and window for input, intermediate and DN nodes.",
            "common_mode": "max(abs(C_FF),abs(C_BB))/max(abs(C_yaw_LF),abs(C_yaw_RF)); upstream numerator/denominator use RMS over matched units before maxima, not signed population averages",
            "acceptance": "both DNp15 bodies must have positive declared preferred yaw response and OF_yaw; abs(OF_translation)/abs(OF_yaw) AND common-mode/yaw ratio must be strictly below their matched upstream ratios. DN mean magnitude DI must also be more rotational than both HS and H2 population magnitude DI. All differences must exceed numerical epsilon 1e-10. A superposition residual records nonlinearity but does not by itself prove a recurrent mechanism; controls determine necessity. Report yaw retention versus feed-forward control without an uncalibrated absolute biological amplitude cutoff.",
            "numerical_epsilon": 1e-10,
            "timestep_convergence": "all conditions and every control: rerun dt=0.25 ms, compare raw states at original sample times (max absolute discrepancy <=0.01 unit drive) and late-response/DI discrepancy <=1e-6; no dynamics/parameter fitting",
            "result_scope": "passing these contracts establishes only a qualitative surrogate neural transformation; descriptive ROI interval agreement is reported, never used as a fitted threshold or proof of calibrated physiology",
            "recurrent_mechanism_contract": "in addition to the limited magnitude/polarity gate, DNp15 must show a nonzero bilateral-superposition residual above 1e-10 and lower translation/yaw and common-mode/yaw ratios in the full model than in the fixed-weight feedforward_chemical_only control on both sides. Otherwise a feed-forward structure can account for the selectivity and recurrent enhancement is not reproduced.",
        },
        "parameters": {"dt_ms": .5, "tau_ms": 20., "chemical_gain": .5, "gap_conductance": .1, "input_amplitude": 1.,
                       "onset_ms": 2000., "offset_ms": 4000., "duration_ms": 6000.},
        "parameter_basis": "one declared default set, not measured or fitted: 20-ms low-pass, 0.5 chemical gain bounds normalized absolute recurrent mass below leak; 0.1 conductance relative to leak; unit signed controlled upstream drive",
        "dynamics_contract": "tau dx/dt=-x+Wchem max(x,0)+Gdiff x+u(t); Wchem post-by-pre: chemical_gain*presynaptic_sign*MaleCNS_count/full_retained_incoming_count; normalize full graph once before ablations. Gap pairs each contribute g*(other_state-self_state) in both directions. Signed dimensionless point-neuron states, zero baseline, no tonic drive/no compartmental or presynaptic gating; explicit Euler dt=0.5 ms and independent dt-halving verification.",
        "stimuli": {
            "yaw_L_F": [1, -1], "yaw_R_F": [-1, 1],
            "translation_F": [1, 1], "translation_B": [-1, -1],
            "left_F": [1, 0], "left_B": [-1, 0], "right_F": [0, 1], "right_B": [0, -1], "zero": [0, 0]},
        "stimulus_encoding": "vector is [left-eye, right-eye] horizontal motion: F=published front-to-back, B=back-to-front; controlled HS drive = ipsilateral vector value; H2 drive = its negative, using MaleCNS LOP postsynaptic side. No workbook indices. Independent 2-s baseline/2-s motion/2-s washout, rather than the paper's paired 13-s trial. Negative input preserves nonpreferred subthreshold inhibition; no fitted binocular input term.",
        "anatomy": manifest,
        "anatomy_files": anatomy_files,
        "upstream_input_eye_verification": roi_validation,
        "bips_connectivity_verification": {"unrestricted_neighborhood_edges": len(neighborhood_frame), "retained_induced_edges_checked": len(retained_neighborhood), "target_type_filter": None, "synapse_threshold": None, "selection_consistency_passed": True},
        "nodes": nodes.replace({np.nan: None}).to_dict(orient="records"),
        "identity_provenance": {"HS_H2_DNp15": "exact MaleCNS types and instances", "bIPS": f"{SUPPLEMENT} Table 2 entries 21/23: hemibrain 1932497911/1963869636 + exact MaleCNS PS321 hemibrain-linked instances", "uLPTCrn": f"{SUPPLEMENT} Table 2 PS070-PS074 + MaleCNS hemibrainType; class-level correspondence only; no individual ROI/body matching", "H2rn": {"paper_roots": H2RN_ROOTS, "source": "published Table 2 root630 -> v1.1.0 release630 exact root -> identical documented anchor supervoxel in pinned release783 annotation table -> current cell_type -> exact MaleCNS flywireType", "released_crosswalk": h2rn_crosswalk, "resolved_types": sorted({r["cell_type"] for r in h2rn_crosswalk}), "level": "identified source neurons establish the class crosswalk; MaleCNS homologues are not individual female-fly ROIs"}},
        "electrical_pairs": gap_pairs,
        "electrical_omissions": "HS/bIPS dye coupling and within-HS coupling omitted because the cited material does not establish all direct pair/side combinations. No generic class-wide gaps. Homologous L/R physiology transfer and equal state scales are explicit assumptions.",
        "chemical_signs": {"acetylcholine": 1, "gaba": -1, "basis": f"MaleCNS consensus transmitter predictions; excitatory cholinergic and inhibitory GABA-A functional interpretation from {PAPER}; not measured edge-specific receptor conductance"},
        "analysis_orientation": "fixed before evaluation: per soma/dendritic side HS, bIPS, uLPTCrn and DNp15 yaw uses ipsilateral F/contralateral B; H2 yaw uses opposite order. Translation is F-minus-B except H2 and H2rn use B-minus-F, following authors' Figure 3e code. Source-table side labels are not treated as MaleCNS body correspondences. Preserve both raw signed OF and magnitude-only DI; H2rn/pooled intermediate ROI correspondence remains class-level.",
        "controls": {
            "full": "all retained chemical edges plus four separately specified electrical pairs",
            "no_electrical": "disable only literature-added electrical operator; chemical graph unchanged. Model intervention, not ShakB: the published Flp-shakB intervention largely preserves HS-to-descending coupling, whereas this control removes HSN-to-DNp15 too.",
            "no_bIPS_chemical_output": "disable all bIPS chemical output, analogous to TNT; electrical pairings unchanged",
            "no_bIPS_GABA_input": "disable all retained GABAergic chemical input to bIPS, idealized Rdl/fast-GABA-receptor perturbation",
            "no_DNp15_GABA_input": "disable all retained GABAergic chemical input to DNp15, idealized Rdl/fast-GABA-receptor perturbation",
            "no_chemical_feedback": "disable intermediate-to-input and intermediate-to-intermediate chemical recurrence; preserve every chemical DN input and literature-added electrical pair",
            "feedforward_chemical_only": "retain only HS/H2-to-intermediate, HS/H2-to-DNp15 and intermediate-to-DNp15 edges; remove electrical interactions. Surviving chemical weights/denominators unchanged. Stage-DAG structural control, not a biological manipulation.",
            "normalization": "normalize full chemical graph once; preserve weights/denominators of surviving edges under ablation"},
        "sources": sources,
        "freeze": "created by anatomy/target-only preparation before first neural evaluation; runner verifies this exact specification, source hashes and anatomy counts",
    }
    if args.check_specification:
        if json.loads(SPEC_PATH.read_text(encoding="utf8")) != spec:
            raise ValueError("frozen specification changed")
    else:
        if SPEC_PATH.exists():
            raise FileExistsError("refusing to overwrite specification; use --check-specification")
        SPEC_PATH.parent.mkdir(parents=True, exist_ok=True)
        SPEC_PATH.write_text(json.dumps(spec, indent=2, allow_nan=False)+"\n", encoding="utf8")
    print(json.dumps({"nodes": len(nodes), "edges": len(edges), "stage_counts": manifest["stage_counts"], "observed": target}, indent=2))


if __name__ == "__main__":
    main()
