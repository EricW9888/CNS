"""Freeze the direct DNp15-to-identified-neck-MN experiment before evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import urllib.request

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.query_malecns_subgraph import DATASET, NEUPRINT_URL, query
from src.malecns_io import load_annotations, load_transmitters
from src.materialization import canonical_edge_frame, materialize_nodes
from src.provenance import sha256_file, validate_materialization

SPEC = ROOT / "experiments/EXP-006-DNp15-neck-motor/specification.json"
FOLDER = ROOT / "data/malecns_exp006_neck"
DN = [11215, 12069]
# Primary Gorko Supplementary Table 2, joined by official MaleCNS mancBodyid.
MOTOR = {801678: (20860, "CvNA1", "TH1", "L"),
         903152: (10407, "CvNA1", "TH1", "R"),
         813696: (17653, "CvNA2", "TH2", "L"),
         804343: (14549, "CvNA2", "TH2", "R")}
PAPER = "https://doi.org/10.1038/s41586-024-07222-5"


def manc_query(statement):
    request = urllib.request.Request(NEUPRINT_URL,
        data=json.dumps({"cypher": statement, "dataset": "manc:v1.2.1"}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        body = json.load(response)
    if "error" in body:
        raise RuntimeError(body["error"])
    return [dict(zip(body["columns"], r)) for r in body["data"]]


def file_entry(path):
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def store_or_verify_bundle(folder, nodes, edges, documents, *, check=False):
    """A re-query checks frozen products without overwriting them on failure."""
    if check:
        pd.testing.assert_frame_equal(nodes, pd.read_parquet(folder / "nodes.parquet"))
        pd.testing.assert_frame_equal(edges, pd.read_parquet(folder / "edges.parquet"))
        for filename, value in documents:
            if json.loads((folder / filename).read_text(encoding="utf8")) != value:
                raise ValueError("re-queried anatomical evidence differs from frozen bundle")
    else:
        folder.mkdir(parents=True, exist_ok=True)
        nodes.to_parquet(folder / "nodes.parquet", index=False)
        edges.to_parquet(folder / "edges.parquet", index=False)
        for filename, value in documents:
            (folder / filename).write_text(json.dumps(value, indent=2)+"\n", encoding="utf8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-specification", action="store_true")
    args = parser.parse_args()
    if SPEC.exists() and not args.check_specification:
        raise FileExistsError("specification already frozen; no post-output tuning")
    # Unrestricted discovery is retained even though only independently identified
    # CvNA1/CvNA2 pairs are dynamically modeled. No target-class absence inference.
    discovery_query = ("MATCH (a:Neuron)-[c:ConnectsTo]->(b:Neuron) "
        f"WHERE a.bodyId IN {DN} RETURN a.bodyId AS pre_body,b.bodyId AS post_body,"
        "b.type AS type,b.instance AS instance,b.superclass AS superclass,"
        "b.somaSide AS somaSide,c.weight AS synapse_count ORDER BY pre_body,post_body")
    discovery = query(discovery_query)
    ids = sorted(DN + list(MOTOR))
    induced_query = ("MATCH (a:Neuron)-[c:ConnectsTo]->(b:Neuron) "
        f"WHERE a.bodyId IN {ids} AND b.bodyId IN {ids} RETURN a.bodyId AS pre_body,"
        "b.bodyId AS post_body,c.weight AS synapse_count ORDER BY pre_body,post_body")
    edges = canonical_edge_frame(query(induced_query))
    # Anatomical all-source denominator, not normalization over the selected DN alone.
    incoming_query = ("MATCH (a:Neuron)-[c:ConnectsTo]->(b:Neuron) "
        f"WHERE b.bodyId IN {sorted(MOTOR)} RETURN b.bodyId AS bodyId,"
        "sum(c.weight) AS incoming_count,count(a) AS source_count ORDER BY bodyId")
    incoming = query(incoming_query)
    if {r["bodyId"] for r in incoming} != set(MOTOR):
        raise ValueError("incomplete unrestricted motor input totals")
    manc_statement = (f"MATCH (n:Neuron) WHERE n.bodyId IN {sorted(x[0] for x in MOTOR.values())} "
        "RETURN n.bodyId AS bodyId,n.type AS type,n.instance AS instance,"
        "n.somaSide AS somaSide ORDER BY bodyId")
    manc = {r["bodyId"]: r for r in manc_query(manc_statement)}
    annotations = load_annotations(ROOT / "data/body-annotations.feather")
    metadata = annotations.set_index("bodyId")
    identity = []
    for body, (manc_id, name, muscle, axon_side) in sorted(MOTOR.items()):
        annotation = metadata.loc[body]
        source = manc[manc_id]
        if int(annotation.mancBodyid) != manc_id or source["type"] != annotation.mancType:
            raise ValueError("primary-ID/official-MaleCNS/MANC correspondence failed")
        if source["instance"].split("_")[-1] != axon_side:
            raise ValueError("MANC nerve-output laterality differs from frozen correspondence")
        identity.append({"bodyId": body, "type": annotation.type, "somaSide": annotation.somaSide,
            "manc_body_id": manc_id, "manc_type": annotation.mancType,
            "manc_instance": source["instance"], "axon_side": axon_side,
            "paper_cell": name, "muscle": muscle,
            "identity_source": PAPER+"; Supplementary Table 2; official MaleCNS mancBodyid",
            "muscle_source": PAPER+"; Figure 4h; CvNA2 also Supplementary Figure 10d",
            "laterality_source": "manc:v1.2.1 explicit CvN_L/R instance; NOT soma side"})
    stages = {i: "DNp15" for i in DN} | {i: "neck_motor" for i in MOTOR}
    nodes = materialize_nodes(ids, stages=stages, annotations=annotations,
        transmitters=load_transmitters(ROOT / "data/body-neurotransmitters.feather"))
    nodes = nodes.sort_values("bodyId").reset_index(drop=True)
    # Retain the complete induced graph. Its weak unclear-transmitter MN edges
    # are explicitly excluded from dynamics, not relabeled as biological absence.
    motor_edges = edges[edges.pre_body.isin(MOTOR)]
    if (not edges.post_body.isin(MOTOR).all() or
        set(map(tuple, motor_edges[["pre_body", "post_body", "synapse_count"]].to_numpy())) !=
            {(801678, 813696, 1), (804343, 903152, 1)}):
        raise ValueError("additional induced connections require a new mechanistic specification")
    if not nodes[nodes.stage.eq("DNp15")].consensus_nt.eq("acetylcholine").all():
        raise ValueError("DN presynaptic transmitter unsupported")
    manifest = {"dataset": DATASET, "release": "MaleCNS v1.0", "node_count": len(nodes),
        "edge_count": len(edges), "stage_counts": nodes.stage.value_counts().to_dict(),
        "query": induced_query, "discovery_query": discovery_query,
        "incoming_query": incoming_query, "manc_query": manc_statement,
        "selection": "Full induced chemical graph of two DNp15 and four primary-ID-crosswalked CvNA1/CvNA2 neck MNs. Unrestricted DN targets and unrestricted motor input totals retained separately. No intermediate is required for these direct connections.",
        "chemical_graph_only": True}
    validate_materialization(manifest, nodes, edges)
    documents = [("manifest.json", manifest), ("discovery.json", discovery),
                 ("incoming_totals.json", incoming), ("manc_identity.json", list(manc.values()))]
    store_or_verify_bundle(FOLDER, nodes, edges, documents, check=args.check_specification)
    anatomy_files = [file_entry(FOLDER / n) for n in ["manifest.json", "nodes.parquet",
        "edges.parquet", "discovery.json", "incoming_totals.json", "manc_identity.json"]]
    source_files = [file_entry(ROOT / "data/exp006_motor" / n) for n in ["gorko2024.pdf", "gorko2024_supplement.pdf"]]
    baseline_files = [file_entry(ROOT / p) for p in [
        "experiments/EXP-005-sensory-to-DNp15/specification.json",
        "experiments/EXP-005-sensory-to-DNp15/record.json", "src/sensory_chain.py",
        "scripts/run_exp005_sensory.py", "src/binocular_physiology.py"]]
    if args.check_specification:
        # Historical raw source-byte provenance is not regenerated on another OS;
        # canonical source integrity is independently checked by the EXP-005 tests.
        baseline_files = json.loads(SPEC.read_text(encoding="utf8"))["baseline_files"]
    spec = {"experiment_id": "EXP-006-DNp15-neck-motor", "parent_commit": "4e36468",
        "anatomy": manifest, "anatomy_files": anatomy_files, "source_files": source_files,
        "baseline_files": baseline_files, "identity": identity, "incoming_totals": incoming,
        "parameters": {"dt_ms": .5, "tau_ms": 20., "chemical_gain": .5,
                       "torque_native_per_activity": 1., "paper_body_pitch_degrees": 40.},
        "parameter_provenance": "Motor tau 20 ms and chemical gain 0.5 are inherited point-neuron engineering defaults, not motor recordings. Torque coefficient 1 native unit per dimensionless activity (=1 nN m/activity with FlyGym g-mm-s units) is an uncalibrated fixed unit-scale assumption, not fitted, peak-normalized or selected from DN output.",
        "dynamics": "tau dm/dt=-m + gain * counts/all-source incoming counts @ max(DNp15,0). Exact exponential update under held DN sample. Pre-update samples, zero initial state. No motor recurrence, tonic drive, recruitment threshold or inhibition inferred from MN transmitter uncertainty. Unmodeled inputs are zero.",
        "structural_edges_not_dynamically_modeled": motor_edges.to_dict("records"),
        "exclusion_reason": "The two 1-synapse motor-source edges remain in the induced anatomy but have no assigned effective action: official consensus transmitter is unclear. This is a direct DN-recruitment approximation, not a complete neck recurrent model.",
        "motor_bridge": "Only CvNA2/TH2 activity emits torque: k*(max(m_axon_R,0)-max(m_axon_L,0))*cos(40deg). Positive sign is the positive yaw component of right-axon-standardized CvNA2 mean activation in Gorko Supplementary Figure 10i,j, not DN side or stimulus order. Mirrored left effect is a symmetry assumption supported by bilateral opposing yaw MNs in Methods. Magnitude, linear recruitment, moment arm and posture independence within this test are provisional. CvNA1 is observed but not actuated. No trial-specific output scaling, gain tuning or controller.",
        "coordinate_contract": "Gorko right-handed lab X/Y/Z=roll/pitch/yaw, positive azimuth front-to-left; body pitched 40deg upward relative to lab X. Native FlyGym body X forward/Y left/Z up at spawn0. Paper lab yaw axis in native body coordinates is [sin40,0,cos40]. Retain only its native Z component; exclude source pitch/roll and native X contribution. Model XML joint_Head_roll is the physical native Z hinge; joint_Head_yaw is native X, verified by xaxis and finite qpos perturbation. This reduced physical azimuth is not the full experimental quaternion trajectory.",
        "body_parameters": {"flygym": "1.2.1", "mujoco": "3.2.7", "timestep_s": .0001,
            "control": "motor", "actuated_joint": "joint_Head_roll", "arena": "Tethered",
            "joint_stiffness": .05, "joint_damping": .06, "neck_stiffness": 10.,
            "non_actuated_joint_stiffness": 1., "non_actuated_joint_damping": 1.,
            "actuator_forcerange": 65., "spawn_orientation": [0., 0., 0.],
            "gravity_mm_s2": [0., 0., -9810.], "seed": 0, "duration_s": 6.},
        "stimuli": ["yaw_L_F", "yaw_R_F"],
        "controls": ["full", "DN_disconnected", "DN_L_disconnected", "DN_R_disconnected", "motor_disconnected"],
        "external_target": {"source": PAPER, "location": "Supplementary Figure 10, Methods, Extended Data Figure 2c",
            "samples": "677 trials, 11 flies; 300-ms unilateral optogenetic activation",
            "qualitative": "Right-axon-standardized CvNA2 activation has a positive population-mean yaw component and negative pitch component; individual trajectories depend on starting pose. Source magnitude is NOT a calibrated prediction for dimensionless visual DN drive.",
            "quantitative_status": "No source numerical axes/speeds extracted. Published plot inspected, not digitized or fitted. The pinned source figure_4/4kl MAT is 1,638,799,259 bytes; bounded header requests timed out. No CvN7-specific calibration substituted for CvNA2.",
            "source_code_commit": "8700dc2a74d4796025938f773498ac91b48240a1"},
        "predeclared_checks": {"motor_dt_relative_error_max": .02, "body_dt_relative_error_max": .02,
            "zero_input_decay_ratio_max": 1e-8, "disconnected_motor_and_torque_exact_zero": True,
            "unit_axon_R_positive_axon_L_negative_torque": True,
            "physical_Z_positive_torque_increases_head_azimuth_vs_zero_command": True,
            "no_expected_sign_assigned_to_visual_conditions": True,
            "claim_boundary": "Anatomically identified muscle drive plus uncalibrated open-loop head-torque tendency. No calibrated physiology, pose-targeted reflex, whole-body steering, gait or closed-loop behavior."}}
    if args.check_specification:
        if json.loads(SPEC.read_text(encoding="utf8")) != spec:
            raise ValueError("frozen anatomical/parameter specification changed")
    else:
        SPEC.parent.mkdir(parents=True, exist_ok=True)
        SPEC.write_text(json.dumps(spec, indent=2, allow_nan=False)+"\n", encoding="utf8", newline="\n")
    # Extend only the final materialization list; leave all previous provenance intact.
    entry = {"name": "EXP-006 identified DNp15 neck motor path", "files": anatomy_files+source_files}
    path = ROOT / "data/manifest.json"
    text = path.read_text(encoding="utf8")
    registry = json.loads(text)
    old = [m for m in registry["materializations"] if m["name"] == entry["name"]]
    if old and old != [entry]:
        raise ValueError("registered provenance differs")
    if not old:
        end = text.rfind("\n  ]\n}")
        if end < 0:
            raise ValueError("unexpected registry layout")
        block = "\n".join("    "+line for line in json.dumps(entry, indent=2).splitlines())
        path.write_text(text[:end]+",\n"+block+text[end:], encoding="utf8", newline="\n")
    print(json.dumps({"anatomy": manifest, "identity": identity, "incoming_totals": incoming}, indent=2))


if __name__ == "__main__":
    main()
