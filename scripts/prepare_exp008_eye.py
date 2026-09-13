"""Extract factual arrays from pinned primary eye data; no author R code is copied.

Optional extraction dependency: rdata==1.1.0. Downloads/derived arrays stay ignored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.compound_eye import EyeGeometry
from src.provenance import sha256_file
from src.sensory_chain import SensoryParameters
from src.compound_eye import Scene

COMMIT = "99d2a43123db636cedb55af9ff31a59657e7d17e"
BASE = f"https://raw.githubusercontent.com/reiserlab/eyemap_T4/{COMMIT}/"
FOLDER = ROOT / "data/exp008_eye"
SOURCE_PATHS = ["data/microCT/20240701.RData", "data/microCT/20240701_nb.RData",
                "data/microCT/20240701_position/lens.csv", "proc_uCT.R", "Fig_3_uCT.R", "LICENSE"]
SPEC = ROOT / "experiments/EXP-008-compound-eye/specification.json"


def evidence_sha256(path):
    """Canonical LF only for Python source; preserve exact scientific data bytes."""
    path = Path(path)
    if path.suffix == ".py":
        return hashlib.sha256(path.read_text(encoding="utf8").encode("utf8")).hexdigest()
    return sha256_file(path)


def freeze_specification(manifest):
    baselines = ["experiments/EXP-005-sensory-to-DNp15/specification.json",
                 "experiments/EXP-005-sensory-to-DNp15/record.json",
                 "experiments/EXP-006-DNp15-neck-motor/specification.json",
                 "experiments/EXP-006-DNp15-neck-motor/record.json",
                 "experiments/EXP-007-CvNA2-motor-boundary/specification.json",
                 "experiments/EXP-007-CvNA2-motor-boundary/record.json",
                 "src/sensory_chain.py", "src/binocular_physiology.py", "src/neck_motor.py", "src/motor_boundary.py"]
    spec = {
        "experiment_id": "EXP-008-compound-eye", "baseline_commit": "97e4c47",
        "question": "Can a measured bilateral compound-eye sensor sample a 3D world and drive the frozen sensory-to-DNp15 projection?",
        "eye_source": manifest,
        "source_fingerprint": "SHA256 of UTF8 Python source with canonical LF; all data/specification/record hashes are exact bytes",
        "frozen_files": [{"path": f, "sha256": evidence_sha256(ROOT/f)} for f in baselines],
        "scene": Scene().__dict__,
        "optics": {"fwhm_degrees": 5., "radial": 5, "azimuthal": 24,
                   "kernel": "normalized exp(-theta^2/(2 sigma^2)) on sphere, FWHM=2 sqrt(2 ln2) sigma, truncated at 3 sigma",
                   "provenance": "10.1242/jeb.074732, Simulation of fly motion vision: Gaussian acceptance model and 5-degree default; not a measured per-facet MaleCNS optical transfer"},
        "photoreceptor": "instantaneous bounded achromatic light proxy, linear radiance integral; no spectral channels, photon noise, saturation, retinal movement or neural superposition",
        "sensor_dt_ms": 2., "sensor_refinement_dt_ms": 1.,
        "pose_conditions": {"static": [0., 0.], "yaw_positive_Z": [45., 0.],
                            "yaw_negative_Z": [-45., 0.], "translation_positive_X": [0., 1.],
                            "translation_negative_X": [0., -1.]},
        "pose_units": "[prescribed physical Z rotation deg/s, world X translation mm/s]; 2 s still, 2 s pose change, 2 s final-pose hold; open loop",
        "early_vision": {"azimuth_range": [60., 100.], "elevation_limit": 15.,
                         "interface": "central-lateral adjacent neighbor-to-center pairs -> facet-specific adaptation and ON/OFF rectification -> pairwise two-quadrant products -> pooled shared eye/type channels; all peripheral facets recorded but unassigned",
                         "direction_constraint": "central a/b rearward/frontward and c/d dorsal/ventral approximation from 10.1038/nature12320; not a peripheral preferred-direction reconstruction",
                         "geometry_constraint": "10.1038/s41586-025-09276-5 shows non-cardinal peripheral T4 fields; no individual female eye/MaleCNS sensory-body correspondence inferred",
                         "filter_provenance": "reuse EXP-005 50/15/10 ms filters; independent neighbor sampler replaces its cyclic image grid",
                         "sampling": "causal zero-order hold of sensor samples; no interpolation from future frames",
                         "parameters": SensoryParameters().__dict__},
        "readout_geometry_correction": {"initial_specification_sha256": "f3b9e4972cabbf97b2dcee26ba8881c28c0b83d9ba998b94a0dda2d95022f860",
                                        "finding": "diametric flank prototype skipped center, spanning 8.08-10.23 degrees across a stripe half-period of 9 degrees; rejected for spatial-alias risk",
                                        "replacement": "adjacent neighbor-center correlators, following 10.1242/jeb.074732; no stimulus/downstream tuning; chosen from geometry before examining DNp15 responses"},
        "controls": ["full", "eye_frozen_at_initial_pose", "eye_to_motion_disconnected", "visual_to_central_disconnected"],
        "acceptance": {
            "scope": "engineering sensory-chain validation, not calcium/behavioral validation; no output fitting or parameter search",
            "eye_geometry": "unit viewing axes, reciprocal same-eye neighbors, exact source arc-distance agreement and preserved measured asymmetry",
            "analytic_checks": "rigid rotations/translations, inverse-pose scene equivalence, foreground occlusion, off-axis visibility, constant-radiance conservation, causal prefixes",
            "quadrature_reference": {"radial": 9, "azimuthal": 64},
            "quadrature_max_abs_light_error": .05,
            "timestep_stage_relative_error": .05,
            "static_and_disconnected_max_abs_activity": 0.,
            "propagation": "both eyes, ON/OFF motion channels, projected HS/H2 and both DNp15 cells show nonzero pose-driven activity; inspect signed raw intermediate traces",
            "stability": "frozen LPi/central contraction, effective transition and zero-input decay preflights at production and halved numerical timestep; bounded convex sensory filters",
            "reproducibility": "repeat samples and all stage traces exactly; independent run must match trace digests and machine record",
            "biological_limits": "No new quantitative optical calibration, full-eye T4 retinotopy, reproduced DNp15 recurrent enhancement or motor claim"}}
    text = json.dumps(spec, indent=2)+"\n"
    SPEC.parent.mkdir(parents=True, exist_ok=True)
    if SPEC.exists() and SPEC.read_text(encoding="utf8") != text:
        raise ValueError("refusing to replace a differing predeclared specification")
    # Keep the exact-byte contract identical to Git's explicit LF specification.
    SPEC.write_text(text, encoding="utf8", newline="\n")
    registry_path = ROOT/"data/manifest.json"
    registry = json.loads(registry_path.read_text(encoding="utf8"))
    paths = {e["path"] for e in registry["sources"]}
    for material in registry["materializations"]:
        paths.update(e["path"] for e in material["files"])
    if not any(e["path"] in paths for e in manifest["files"]):
        entry = {"name": "EXP-008 pinned measured bilateral eye geometry", "files": manifest["files"]}
        original = registry_path.read_text(encoding="utf8")
        end = original.rfind("\n  ]")
        if end < 0:
            raise ValueError("unrecognized registry layout")
        addition = "\n".join("    "+line for line in json.dumps(entry, indent=2).splitlines())
        registry_path.write_text(original[:end]+",\n"+addition+original[end:], encoding="utf8")
    print("Predeclared specification frozen; no downstream output evaluated.")


def extract(data, neighborhood):
    lens, cone = np.asarray(data["lens"]), np.asarray(data["cone"])
    match = np.asarray(data["i_match"]).reshape(-1)
    n = len(lens)
    if not np.array_equal(np.sort(match), np.arange(1, n+1)):
        raise ValueError("cone-to-lens mapping is not a one-based permutation")
    order = np.argsort(match)
    left = np.asarray(data["ind_left_lens"], dtype=bool)
    if not np.array_equal(left, np.asarray(data["ind_left_cone"], dtype=bool)[order]):
        raise ValueError("lens/cone eye assignment mismatch")
    raw = lens-cone[order]
    raw /= np.linalg.norm(raw, axis=1)[:, None]
    if not np.allclose(raw, np.asarray(data["ucl_rot"])[order], atol=1e-12):
        raise ValueError("saved directions disagree with lens-to-tip geometry")
    neighbors = np.full((n, 6), -1, dtype=np.int64)
    grid = np.empty((n, 2), dtype=np.int64)
    for label, mask in [("left", left), ("right", ~left)]:
        ids = np.flatnonzero(mask)
        nb = np.asarray(neighborhood[f"nb_ind_{label}"])
        xy = np.asarray(neighborhood[f"ind_xy_{label}"])
        if (not np.array_equal(np.sort(nb[:, 0]), np.arange(1, len(ids)+1))
                or not np.array_equal(np.sort(xy[:, 0]), np.arange(1, len(ids)+1))):
            raise ValueError("neighborhood homes do not cover the eye")
        nb = nb[np.argsort(nb[:, 0]), 1:]
        valid = np.isfinite(nb)
        if np.any(nb[valid] != np.round(nb[valid])) or np.any((nb[valid] < 1) | (nb[valid] > len(ids))):
            raise ValueError("invalid one-based eye-local neighbor")
        local = np.full(nb.shape, -1, dtype=np.int64)
        local[valid] = ids[nb[valid].astype(int)-1]
        neighbors[ids] = local
        grid[ids] = xy[np.argsort(xy[:, 0]), 1:].astype(np.int64)
        if not np.array_equal(xy[:, 1:], np.round(xy[:, 1:])):
            raise ValueError("nonintegral primary eye-grid coordinates")
        # Independently recompute author's recorded arc distances after permutation.
        direction = np.asarray(data["ucl_rot_sm"])[order][ids]
        computed = np.full(nb.shape, np.nan)
        rows, slots = np.nonzero(valid)
        computed[rows, slots] = np.arccos(np.clip(np.sum(direction[rows]*direction[nb[valid].astype(int)-1], axis=1), -1, 1))
        published = np.asarray(neighborhood[f"nb_dist_ucl_{label}"])[:, 1:]
        if not np.allclose(computed, published, atol=1e-12, equal_nan=True):
            raise ValueError("published neighbor angular distances disagree")
    return EyeGeometry(lens/1000, np.asarray(data["ucl_rot_sm"])[order], left, neighbors, grid)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--freeze-specification", action="store_true")
    args = parser.parse_args()
    FOLDER.mkdir(parents=True, exist_ok=True)
    entries = []
    for source in SOURCE_PATHS:
        path = FOLDER / Path(source).name
        if args.download and not path.exists():
            response = requests.get(BASE+source, timeout=60)
            response.raise_for_status()
            path.write_bytes(response.content)
        entries.append({"path": path.relative_to(ROOT).as_posix(), "url": BASE+source,
                        "bytes": path.stat().st_size, "sha256": sha256_file(path), "redistributed": False})
    # Source CSV explicitly specifies micrometres; do not guess scale from head size.
    if "µm" not in (FOLDER / "lens.csv").read_text(encoding="latin1")[:200]:
        raise ValueError("unverified source position units")
    import rdata
    geometry = extract(rdata.read_rda(FOLDER / "20240701.RData"), rdata.read_rda(FOLDER / "20240701_nb.RData"))
    target = FOLDER / "geometry.npz"
    if target.exists():
        old = EyeGeometry.load(target)
        if not all(np.array_equal(getattr(old, k), getattr(geometry, k)) for k in geometry.__dict__):
            raise ValueError("refusing to replace differing eye geometry")
    else:
        np.savez_compressed(target, **geometry.__dict__)
    entries.append({"path": target.relative_to(ROOT).as_posix(), "bytes": target.stat().st_size,
                    "sha256": sha256_file(target), "redistributed": False})
    manifest = {"primary_doi": "10.1038/s41586-025-09276-5", "source_commit": COMMIT,
                "specimen": "20240701; female microCT; not the MaleCNS individual",
                "source_repository_license": "GPL-3.0; source/arrays remain local; no R implementation copied",
                "coordinate_frame": "author aligned +X anterior, +Y left, +Z dorsal; micrometres converted to mm",
                "viewing_axes": "author's smoothed normalized lens-minus-photoreceptor-tip vectors; cone order permuted to lens order",
                "left_count": int(geometry.left.sum()), "right_count": int((~geometry.left).sum()), "files": entries}
    text = json.dumps(manifest, indent=2)+"\n"
    manifest_path = FOLDER / "manifest.json"
    if manifest_path.exists() and manifest_path.read_text() != text:
        raise ValueError("source manifest changed")
    manifest_path.write_text(text, encoding="utf8")
    print(text)
    if args.freeze_specification:
        freeze_specification(manifest)


if __name__ == "__main__":
    main()
