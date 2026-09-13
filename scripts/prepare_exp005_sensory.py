"""Freeze the sensory interface and query its chemical anatomy, without neural evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.query_malecns_subgraph import DATASET, query
from src.malecns_io import load_annotations, load_transmitters
from src.materialization import canonical_edge_frame, materialize_nodes
from src.provenance import sha256_file, validate_materialization

SPEC = ROOT / "experiments/EXP-005-sensory-to-DNp15/specification.json"
FOLDER = ROOT / "data/malecns_exp005_sensory"
CENTRAL = ROOT / "experiments/EXP-004-binocular-physiology/specification.json"
LPI_IDS = [10066, 10308, 10436, 11041, 11709, 13310]
TYPES = [f"T{number}{subtype}" for number in [4, 5] for subtype in "abcd"]


def verify_input_eyes(nodes):
    """Validate eye assignment by sensory dendritic ROI, not soma alone.

    This additional anatomical check does not alter the frozen input encoding.
    It records scalar ROI counts rather than downloading complete synapse data.
    """
    rows = []
    for stage, region in [("T4", "ME"), ("T5", "LO")]:
        ids = nodes[nodes.stage.eq(stage)].bodyId.tolist()
        for start in range(0, len(ids), 500):
            statement = (f"MATCH (n:Neuron) WHERE n.bodyId IN {ids[start:start+500]} "
                         "WITH n, apoc.convert.fromJsonMap(n.roiInfo) AS r "
                         f"RETURN n.bodyId AS bodyId, r['{region}(L)'].post AS L_post, "
                         f"r['{region}(R)'].post AS R_post, r['LOP(L)'].pre AS L_LOP_pre, r['LOP(R)'].pre AS R_LOP_pre ORDER BY bodyId")
            for row in query(statement):
                row.update({"stage": stage, "input_region": region})
                rows.append(row)
    metadata = nodes.set_index("bodyId")
    if len(rows) != nodes.stage.isin(["T4", "T5"]).sum() or len({r["bodyId"] for r in rows}) != len(rows):
        raise ValueError("missing/duplicate sensory eye verification")
    for row in rows:
        row["L_post"], row["R_post"] = int(row["L_post"] or 0), int(row["R_post"] or 0)
        row["L_LOP_pre"], row["R_LOP_pre"] = int(row["L_LOP_pre"] or 0), int(row["R_LOP_pre"] or 0)
        row["assigned_eye"] = metadata.loc[row["bodyId"], "somaSide"]
        row["basis"] = "sensory_dendritic_ROI"
        ipsi = row[row["assigned_eye"]+"_post"]
        contra = row[("R" if row["assigned_eye"] == "L" else "L")+"_post"]
        if ipsi == contra == 0:
            # A typed traced cell may have incomplete dendritic reconstruction.
            # Retain it with same-eye LOP output evidence; never invent a RF.
            row["basis"] = "LOP_output_only_dendritic_ROI_absent"
            ipsi = row[row["assigned_eye"]+"_LOP_pre"]
            contra = row[("R" if row["assigned_eye"] == "L" else "L")+"_LOP_pre"]
        if ipsi <= contra:
            raise ValueError(f"soma-based sensory eye disagrees with dominant dendritic ROI: {row}")
    return {"dataset": DATASET, "method": "Dominant postsynaptic ME side for T4 and LO side for T5; if both dendritic ROIs absent, require dominant LOP presynaptic side. Every selected cell checked; output-only evidence is marked, not workbook RF geometry", "verified_bodies": len(rows),
            "dendritic_ROI_absent_body_ids": [r["bodyId"] for r in rows if r["basis"] != "sensory_dendritic_ROI"], "rows": sorted(rows, key=lambda r: r["bodyId"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-specification", action="store_true")
    args = parser.parse_args()
    if SPEC.exists() and not args.check_specification:
        raise FileExistsError("specification already frozen")
    central = json.loads(CENTRAL.read_text(encoding="utf8"))
    input_ids = sorted(n["bodyId"] for n in central["nodes"] if n["stage"] in {"HS", "H2"})
    targets = sorted(input_ids + LPI_IDS)
    cypher = ("MATCH (a:Neuron)-[c:ConnectsTo]->(b:Neuron) "
              f"WHERE b.bodyId IN {targets} AND (a.type IN {TYPES} OR a.bodyId IN {LPI_IDS}) "
              "RETURN a.bodyId AS pre_body, b.bodyId AS post_body, c.weight AS synapse_count "
              "ORDER BY pre_body, post_body")
    edges = canonical_edge_frame(query(cypher))
    annotations = load_annotations(ROOT / "data/body-annotations.feather")
    stages = {i: "LPi" for i in LPI_IDS}
    stages.update({n["bodyId"]: n["stage"] for n in central["nodes"] if n["bodyId"] in input_ids})
    motion_ids = sorted(set(edges.pre_body) - set(LPI_IDS))
    annotation_types = annotations.set_index("bodyId").type
    stages.update({i: "T4" if annotation_types.loc[i].startswith("T4") else "T5" for i in motion_ids})
    nodes = materialize_nodes(set(edges.pre_body) | set(targets), stages=stages,
                              annotations=annotations, transmitters=load_transmitters(ROOT / "data/body-neurotransmitters.feather"))
    nodes = nodes.sort_values("bodyId").reset_index(drop=True)
    if not nodes.somaSide.isin(["L", "R"]).all():
        raise ValueError("motion-input side unresolved")
    expected = nodes.stage.map({"T4": "acetylcholine", "T5": "acetylcholine", "HS": "acetylcholine", "H2": "acetylcholine", "LPi": "gaba"})
    if not nodes.consensus_nt.eq(expected).all():
        raise ValueError("unexpected consensus transmitter; no silent sign substitution")
    if set(nodes[nodes.stage.eq("LPi")].type) != {"LPi12", "LPi21"}:
        raise ValueError("horizontal LPi identity changed")
    manifest = {"dataset": DATASET, "release": "MaleCNS v1.0", "node_count": len(nodes), "edge_count": len(edges),
                "stage_counts": nodes.stage.value_counts().sort_index().to_dict(), "query": cypher,
                "selection": "All positive T4/T5 or six identified horizontal LPi inputs to eight frozen HS/H2 and those LPi; no count threshold. Upstream boundary, not an induced whole-visual graph. Central induced graph reused unchanged.",
                "chemical_graph_only": True}
    validate_materialization(manifest, nodes, edges)
    FOLDER.mkdir(parents=True, exist_ok=True)
    nodes.to_parquet(FOLDER / "nodes.parquet", index=False)
    edges.to_parquet(FOLDER / "edges.parquet", index=False)
    (FOLDER / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf8", newline="\n")
    files = [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size, "sha256": sha256_file(p)}
             for p in [FOLDER / n for n in ["manifest.json", "nodes.parquet", "edges.parquet"]]]
    groups = nodes[nodes.stage.isin(["T4", "T5"])].groupby(["type", "somaSide"]).bodyId.count()
    spec = {
        "experiment_id": "EXP-005-sensory-to-DNp15", "parent_commit": "15547c9",
        "central_specification": {"path": CENTRAL.relative_to(ROOT).as_posix(), "sha256": sha256_file(CENTRAL)},
        "central_parameters": central["parameters"], "external_target": central["external_target"],
        "observation_contract": central["observation_contract"], "stimuli": central["stimuli"],
        "sensory_parameters": {"dt_ms": .5, "pixel_degrees": 2.25, "side_pixels": 16, "wavelength_degrees": 18.,
                               "speed_degrees_s": 45., "mean_luminance": .5, "contrast": .5, "highpass_tau_ms": 50.,
                               "delay_tau_ms": 15., "emission_tau_ms": 10., "LPi_tau_ms": 20., "chemical_gain": .5,
                               "onset_ms": 2000., "offset_ms": 4000., "duration_ms": 6000.},
        "stimulus_interface": "Explicit two-eye 16x16 luminance patches: sinusoid in x, stationary for 2 s, translated for 2 s, stationary at final phase for 2 s. x positive is published stimulus-space front-to-back on each eye, not workbook/body geometry. Grating wavelength/speed/contrast are fixed engineering defaults, NOT the experimental spherical starfield or a fitted visual calibration. Periodic spatial boundary and uniform laminar flow; no parallax, gaze, photoreceptor reconstruction or behavioral controller.",
        "motion_interface": "Two-quadrant Reichardt approximation applied to luminance: high-pass (I minus adapting low-pass), ON/OFF half-wave split, 15-ms delayed adjacent spatial products, subtract reversed arm, spatial mean then positive/negative split, 10-ms nonnegative emission filter. All cells sharing eye and T4/T5 subtype share a channel for this spatially uniform stimulus. This physiological input assignment is provisional, not emergent direction selectivity from MaleCNS microcircuit or an individual receptive-field reconstruction. No workbook indices used.",
        "channel_assignment": {"T4": "ON", "T5": "OFF", "a": "+x / stimulus-space front-to-back", "b": "-x / back-to-front", "c": "+y / upward", "d": "-y / downward"},
        "physiology_sources": {
            "ON_OFF_and_subtype_directions": "https://doi.org/10.1038/nature12320",
            "Reichardt_filter_times": "https://doi.org/10.1038/s41593-025-01948-9; Methods, Agent simulations, high-pass 50 ms and delay 15 ms. Do not import the paper's manually fitted downstream weights.",
            "T4_T5_LPi_LPTC_path": "https://doi.org/10.1016/j.cub.2022.06.061",
            "multilevel_opponency": "https://doi.org/10.1038/s41593-023-01443-z; measured vertical LPi glutamate/GluCl conductance is NOT assigned to horizontal LPi12/21. MaleCNS selected horizontal cells are consensus GABA.",
            "central_interactions": "Frozen EXP-004 specification; its four pair-specific electrical interactions, identity crosswalk and source measurements unchanged."},
        "numerical_assumptions": "Motion filters are explicit causal low-pass updates; shared T4/T5 channels are exact storage compression of the declared uniform-input assignment. New visual chemical counts normalized once per target by ALL retained visual input counts, gain 0.5, ACh positive/GABA negative. LPi rectified emission, signed state, tau 20 ms. External drive enters only the eight original HS/H2 nodes. Existing central chemical normalization, gap operator, leak and parameters are untouched; no central-to-visual feedback is reconstructed.",
        "free_parameter_basis": "No fit or search: new pattern parameters, 10-ms T4/T5 emission time and 20-ms LPi time are explicit defaults; 0.5 visual gain independently provides a contraction bound. 50/15-ms motion filters follow the paper's algorithm, not measured universal cell constants. Raw luminance-product amplitudes are not normalized or rescaled to recover DNp15.",
        "validation_contract": {
            "static_and_uniform_flicker": "no directed motion output",
            "sensory_symmetries": "reversal swaps a/b; eye exchange swaps banks; contrast inversion swaps ON/OFF; vertical movement tests c/d; causal prefixes and deterministic execution",
            "anatomical_invariants": "exact count preservation; all identified central cells/operators unchanged; compressed projection equals explicit per-body sparse projection; no post-ablation renormalization",
            "preflight": "global LPi and central effective Euler contraction below one, perturbed zero-input decay below 1e-8 at 2 s, diffusive central coupling semantics; each control at both timesteps",
            "timestep_convergence": "dt 0.5 versus 0.25 ms; stage trace maximum absolute discrepancy / global maximum stage amplitude <= 0.05 (zero stages use 1e-12 floor); central late signed/magnitude DI discrepancy <= 0.02, undefined indices must remain undefined. These sensory transient tolerances replace only EXP-005 numerical convergence, never EXP-004 biology/analysis contracts.",
            "biology": "Report motion-channel polarity/direction, HS/H2 front/back preferences, intermediate and DNp15 responses. Reuse frozen EXP-004 observation/window/sign/magnitude/matched-upstream requirements and source ROI comparisons, including causal recurrent controls. New gratings and uncalibrated states permit only qualitative/descriptive comparison, not a calibrated calcium prediction. A failed biological comparison does not block preserving a stable observable chain."},
        "controls": {
            "full": "entire new sensory path plus frozen central network",
            "visual_disconnected": "zero only visual external HS/H2 drive; central network unchanged",
            "no_LPi_output": "remove all LPi chemical output including LPi recurrence; preserve direct T4/T5 and survivor weights",
            "no_T4": "zero ON/T4 emissions, leave all chemical weights and OFF branch unchanged",
            "no_T5": "zero OFF/T5 emissions, leave all chemical weights and ON branch unchanged",
            "no_electrical": "reuse frozen central model intervention, not a ShakB replica",
            "no_bIPS_GABA_input": "reuse frozen central GABA-to-bIPS intervention",
            "feedforward_chemical_only": "reuse frozen central stage-DAG/electrical-off control"},
        "anatomy": manifest, "anatomy_files": files,
        "motion_coverage": {f"{kind}_{side}": int(n) for (kind, side), n in groups.items()},
        "LPi_identity": nodes[nodes.stage.eq("LPi")][["bodyId", "type", "somaSide", "consensus_nt"]].to_dict("records"),
        "unknowns": ["Individual T4/T5 receptive fields and workbook-to-visual geometry", "Photoreceptor/lamina/medulla dynamics omitted by provisional motion interface", "Horizontal LPi receptor-specific strengths and electrical interactions beyond frozen supported pairs", "Female calcium to male dimensionless-state observation transform", "Common input scale and ignored visual feedback"],
        "freeze": "Anatomy and assumptions created before any EXP-005 central evaluation; no parameters chosen from tested DNp15 responses."}
    if args.check_specification:
        if spec != json.loads(SPEC.read_text(encoding="utf8")):
            raise ValueError("materialization or specification differs from frozen version")
    else:
        SPEC.parent.mkdir(parents=True, exist_ok=True)
        SPEC.write_text(json.dumps(spec, indent=2, allow_nan=False)+"\n", encoding="utf8", newline="\n")
    eye_evidence = verify_input_eyes(nodes)
    eye_path = FOLDER / "input_eye_verification.json"
    eye_path.write_text(json.dumps(eye_evidence, indent=2)+"\n", encoding="utf8", newline="\n")
    registered_files = files + [{"path": eye_path.relative_to(ROOT).as_posix(), "bytes": eye_path.stat().st_size, "sha256": sha256_file(eye_path)}]
    registry_path = ROOT / "data/manifest.json"
    registry_text = registry_path.read_text(encoding="utf8")
    registry = json.loads(registry_text)
    entry = {"name": "EXP-005 sensory-to-HS/H2", "files": registered_files}
    existing = [m for m in registry["materializations"] if m["name"] == entry["name"]]
    previous = {"name": entry["name"], "files": files}
    if existing and existing not in [[entry], [previous]]:
        raise ValueError("registered materialization changed")
    def block(value):
        return "\n".join("    "+line for line in json.dumps(value, indent=2).splitlines())
    if existing == [previous]:
        registry_text = registry_text.replace(block(previous), block(entry))
    elif not existing:
        end = registry_text.rfind("\n  ]\n}")
        if end < 0:
            raise ValueError("unexpected registry layout")
        registry_text = registry_text[:end]+",\n"+block(entry)+registry_text[end:]
    registry_path.write_text(registry_text, encoding="utf8", newline="\n")
    print(json.dumps({"anatomy": manifest, "motion_coverage": spec["motion_coverage"], "verified_input_eyes": len(eye_evidence["rows"])}, indent=2))


if __name__ == "__main__":
    main()
