"""Pin source retinotopy arrays and predeclare EXP-009 before neural evaluation.

Author R implementations are downloaded only as local provenance, never copied
into CNS implementation. Optional extraction dependency is the existing rdata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from urllib.request import urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_exp008_eye import BASE, COMMIT, evidence_sha256
from scripts.run_exp008_eye import load_frozen_eye
from src.provenance import sha256_file

FOLDER = ROOT/"data/exp009_retinotopy"
SPEC = ROOT/"experiments/EXP-009-retinotopic-eye/specification.json"
SOURCES = ["data/eyemap.RData", "data/T4_RF_pred.RData", "data/med_ixy.RData",
           "proc_eyemap.R", "proc_T4.R", "Fig_4.R", "ED_fig_7.R", "eyemap_func.R"]


def frozen_entry_digest(entry, *, root=ROOT):
    """Explicit legacy newline policy, never generic JSON reserialization.

    EXP-008 generated its Windows record with CRLF, while its Git attribute
    stores LF. Normalize only that named legacy record and Python source.
    Every other scientific record, specification and array remains exact-byte.
    """
    path = root/entry["path"]
    method = entry.get("sha256_method", "existing_evidence_policy")
    if method == "utf8_with_lf_newlines":
        if entry["path"] != "experiments/EXP-008-compound-eye/record.json":
            raise ValueError("legacy newline normalization outside named predecessor record")
        return hashlib.sha256(path.read_text(encoding="utf8").encode("utf8")).hexdigest()
    if method != "existing_evidence_policy":
        raise ValueError("unknown frozen fingerprint method")
    return evidence_sha256(path)


def extract_reference(eye, field, medulla, geometry):
    mapping = np.asarray(eye["eyemap"])
    if (mapping.shape != (778, 2) or not np.isfinite(mapping).all()
            or not np.array_equal(mapping, np.round(mapping))):
        raise ValueError("unexpected primary eye-map indices")
    mapping = mapping.astype(np.int64)
    if any(len(np.unique(mapping[:, i])) != len(mapping) for i in (0, 1)):
        raise ValueError("primary eye map must be one-to-one within its source datasets")
    right = np.flatnonzero(~geometry.left)
    if np.any((mapping[:, 0] < 1) | (mapping[:, 0] > 779)):
        raise ValueError("eye-map medulla index outside source columns")
    if np.any((mapping[:, 1] < 1) | (mapping[:, 1] > len(right))):
        raise ValueError("eye-map lens outside measured right eye")
    ids = right[mapping[:, 1]-1]
    axes = np.asarray(eye["ucl_rot_sm"])
    if not np.array_equal(axes, geometry.directions[ids]):
        raise ValueError("source registered axes disagree with literal lens IDs")
    predicted = field["RF_lens_T4_pred"]
    if not np.array_equal(np.asarray(predicted.coords["dim_0"], dtype=np.int64), mapping[:, 1]):
        raise ValueError("direction field row names disagree with primary registration")
    endpoints = np.asarray(predicted).reshape(778, 4, 3)
    # proc_T4.R lines 426-437 specify a,b,c,d order and antiparallelism;
    # don't relabel by the misleading name of an intermediate R variable.
    if not (np.allclose(endpoints[:, 0]+endpoints[:, 1], 2*axes, atol=1e-12)
            and np.allclose(endpoints[:, 2]+endpoints[:, 3], 2*axes, atol=1e-12)):
        raise ValueError("unexpected primary antiparallel field construction")
    med = np.asarray(medulla["med_ixy"])
    if not np.array_equal(np.sort(med[:, 0]), np.arange(1, 780)):
        raise ValueError("source medulla addresses must cover 779 indexed columns")
    med = med[np.argsort(med[:, 0])]
    if not np.array_equal(med[mapping[:, 0]-1, 0], mapping[:, 0]):
        raise ValueError("medulla column indices inconsistent")
    registered_grid = np.asarray(eye["lens_ixy"])
    if not np.array_equal(registered_grid[mapping[:, 1]-1, 0], mapping[:, 1]):
        # Source lens_ixy is in traversal rather than lens order.
        registered_grid = registered_grid[np.argsort(registered_grid[:, 0])]
    if not np.array_equal(registered_grid[mapping[:, 1]-1, 1:], med[mapping[:, 0]-1, 1:]):
        raise ValueError("source medulla/lens hex addresses do not match")
    address_shift = registered_grid[mapping[:, 1]-1, 1:]-geometry.grid[ids]
    if not np.array_equal(address_shift, np.broadcast_to([1, 0], address_shift.shape)):
        raise ValueError("source registered-grid origin differs from verified measured-grid translation")
    return {"facet_ids": ids, "viewing_axes": axes, "pd_endpoints": endpoints,
            "FAFB_Mi1_column_indices": mapping[:, 0], "source_right_lens_ids_1based": mapping[:, 1],
            "source_registered_grid": med[mapping[:, 0]-1, 1:].astype(np.int64)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    eye_spec, geometry = load_frozen_eye()
    FOLDER.mkdir(parents=True, exist_ok=True)
    entries = []
    for source in SOURCES:
        path = FOLDER/Path(source).name
        if args.download and not path.exists():
            with urlopen(BASE+source, timeout=30) as response:
                path.write_bytes(response.read())
        entries.append({"path": path.relative_to(ROOT).as_posix(), "url": BASE+source,
                        "bytes": path.stat().st_size, "sha256": sha256_file(path), "redistributed": False})
    import rdata
    reference = extract_reference(*(rdata.read_rda(FOLDER/Path(f).name) for f in SOURCES[:3]), geometry)
    target = FOLDER/"reference.npz"
    if target.exists():
        with np.load(target, allow_pickle=False) as old:
            if set(old.files) != set(reference) or any(not np.array_equal(old[k], v) for k, v in reference.items()):
                raise ValueError("refusing to replace differing primary reference arrays")
    else:
        np.savez_compressed(target, **reference)
    entries.append({"path": target.relative_to(ROOT).as_posix(), "bytes": target.stat().st_size,
                    "sha256": sha256_file(target), "redistributed": False})
    # Independently inspect the available MaleCNS schema; no inferred body map.
    annotation = pd.read_feather(ROOT/"data/body-annotations.feather")
    motion = annotation[annotation.type.str.fullmatch("T[45][abcd]", na=False)]
    workbook = pd.read_excel(ROOT/"data/optic-column-type-assignments-v1.0.xlsx")
    identity_evidence = {"motion_annotation_count": len(motion),
                         "motion_assignedOlHex1_nonmissing": int(motion.assignedOlHex1.notna().sum()),
                         "motion_assignedOlHex2_nonmissing": int(motion.assignedOlHex2.notna().sum()),
                         "workbook_rows": len(workbook), "workbook_eye_prefixes": sorted(workbook.column.str[:4].unique()),
                         "workbook_columns": list(workbook.columns),
                         "interpretation": "available workbook is right-eye L1/R7/R8 organization, not a T4/T5 address table; no female-measured-eye to MaleCNS body registration supplied"}
    frozen = ["experiments/EXP-008-compound-eye/specification.json", "experiments/EXP-008-compound-eye/record.json",
              "src/compound_eye.py", "scripts/run_exp008_eye.py", "scripts/prepare_exp008_eye.py",
              "data/body-annotations.feather", "data/optic-column-type-assignments-v1.0.xlsx"]
    frozen_entries = []
    for f in frozen:
        entry = {"path": f}
        if f == "experiments/EXP-008-compound-eye/record.json":
            entry["sha256_method"] = "utf8_with_lf_newlines"
        entry["sha256"] = frozen_entry_digest(entry)
        # Preserve the compact existing order: path, SHA, optional method.
        frozen_entries.append({"path": f, "sha256": entry["sha256"],
                               **({"sha256_method": entry["sha256_method"]} if "sha256_method" in entry else {})})
    specification = {
        "experiment_id": "EXP-009-retinotopic-eye", "baseline_commit": "af2cb05",
        "question": "Does a broader measured-grid local-direction representation replace the central-local approximation while preserving the frozen sensory-to-descending chain?",
        "source_commit": COMMIT, "primary_doi": "10.1038/s41586-025-09276-5",
        "source_files": entries, "frozen_files": frozen_entries,
        "anatomical_mapping": "literal author right-lens -> female FAFB Mi1 column index map, 778 registered axes; NOT MaleCNS body IDs",
        "source_registered_grid_minus_EXP008_grid": [1, 0],
        "MaleCNS_identity_evidence": identity_evidence,
        "direction_basis": "measured complete six-neighbor grid: b=unit_tangent(mean(slots 2,3)-mean(slots 0,5)); d=unit_tangent(slot4-slot1); a=-b,c=-d. Zero-based slots have primary p,q offsets (1,0),(1,1),(0,1),(-1,0),(-1,-1),(0,-1). No azimuth/elevation cardinal projection or symmetry enforcement.",
        "direction_provenance": "Zhao Fig.4C compares T4b with local +h; ED Fig.7 compares T4d with -v; proc_T4.R constructs antiparallel a/b and c/d. Right field is anatomy-predicted and registered to the measured female eye. Applying the same measured-grid rule to the left eye and T5 is an explicit class-level provisional extension, not measured physiology.",
        "photoreceptor_boundary": "facet-axis scalar light treated as retinotopic visual-axis proxy; R1-R6 rhabdomere axes and neural superposition are NOT reconstructed; no photoreceptor, lamina or medulla body added",
        "upstream_anatomy_source": "https://doi.org/10.7554/eLife.24394; retinotopic ON/OFF pathway constraints do not uniquely specify the missing individual cross-dataset sensory map",
        "optics_scene_and_pose": "reuse exact frozen EXP-008 world and bilateral sensor geometry/5-degree optics/poses; recorded scalar-light replay validated against frozen trace digests and direct ray sampling; no RGB camera, semantics, flow or requested direction enters neural kernels",
        "parameters": eye_spec["early_vision"]["parameters"], "sensor_dt_ms": eye_spec["sensor_dt_ms"],
        "conditions": eye_spec["pose_conditions"],
        "local_dynamics": "a'= (I-a)/50ms; q=(max(I-a,0),max(a-I,0)); d'=(q-d)/15ms; C_j=d_neighbor_j*q_center-q_neighbor_j*d_center; r_b=(C0+C5-C2-C3)/4; r_d=(C1-C4)/2; e'= (max((-r_b,+r_b,-r_d,+r_d),0)-e)/10ms. Explicit Euler dt=0.5ms; output before update; adaptation starts at first light sample.",
        "adapter": "equal measured-neighborhood mean AFTER local rectification/emission within each eye, ON/OFF and a/b/c/d; all local traces retained at sensor-coincident times. Lossy shared-population adapter required by unchanged EXP-005; NOT individual MaleCNS retinotopy, wide-field RF reconstruction or connectivity-derived spatial weights.",
        "local_storage_dt_ms": 2., "storage": "float64 local emissions at 2ms; full shared channels and downstream at 0.5ms; no scientific interpolation",
        "acceptance": {
            "geometry": "exact source registered viewing axes and medulla/lens grid addresses; preserve all measured lens positions/asymmetries; complete same-eye reciprocal neighborhoods only",
            "source_orientation": "every comparable right-eye grid direction in same tangent hemiplane as source T4 prediction (positive dot product); report p50/p95/max angles, no fitted orientation correction. This is a deliberately weak sanity invariant, not quantitative functional-field equivalence.",
            "local": "static and uniform flicker null; contrast inversion exchanges ON/OFF; known scalar traveling patterns reverse a/b or c/d preference; exact causal prefixes/determinism; independent scalar reference equals vector kernel",
            "coverage": "all complete measured neighborhoods (no output-selected aperture); report omitted edge facets and absent anatomy",
            "numerical": "convex sensory filters; frozen LPi/central contraction and zero-input decay at full/half dt, same surviving operators in disconnections",
            "timestep_stage_relative_error": 0.05,
            "spatial_sampling": "discrete measured graph has no continuous resolution parameter; report two p-parity subset pooling sensitivities descriptively, do not invent finer facets or force agreement",
            "downstream": "evaluate only after local geometry/source/numerical gates; nonzero ON/OFF, HS/H2 and both DNs for moving conditions; causal disconnections exactly null; no DN-based fitting or gain rescaling",
            "biology": "preserve frozen EXP-004/005 discrepancies; unequal rotation/translation retinal speeds and uncalibrated activity prohibit a new yaw-selectivity or calcium-agreement claim"},
        "controls": ["full", "eye_to_retinotopy_disconnected", "retinotopy_to_T4_T5_disconnected", "visual_to_central_disconnected"],
        "freeze": "sensory choices fixed from source anatomy and local invariants before EXP-009 downstream output; no fitting/search; baseline model parameters unchanged"}
    text = json.dumps(specification, indent=2, allow_nan=False)+"\n"
    SPEC.parent.mkdir(parents=True, exist_ok=True)
    if SPEC.exists() and SPEC.read_text(encoding="utf8") != text:
        raise ValueError("predeclared EXP-009 specification differs; refusing overwrite")
    SPEC.write_text(text, encoding="utf8", newline="\n")
    print("Primary registration verified; EXP-009 specification frozen before downstream evaluation.")
    print(json.dumps(identity_evidence, indent=2))


if __name__ == "__main__":
    main()
