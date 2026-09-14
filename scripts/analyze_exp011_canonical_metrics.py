"""Compute the frozen EXP-011 local metrics from immutable sensor archives.

This is a diagnostic continuation of the canonical-local run.  It never
rerenders sensors, changes a verifier, clips visibility diagnostics, or writes
bulk arrays.  It uses the frozen local readout and an exactly equivalent
streaming reconstruction of the independent correlator/stencil diagnostic.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_exp008_eye import evidence_sha256
from scripts.run_exp005_sensory import relative_error
from scripts.run_exp008_eye import held_light
from scripts.run_exp009_retinotopy import load_resolved
from src.retinotopic_eye import RetinotopicReadout
from src.retinotopy_diagnostics import correlators, error_field
from src.sensory_chain import SensoryParameters
from src.provenance import sha256_file


SPEC_PATH = ROOT / "experiments/EXP-011-acceptance-convergence/specification.json"
MANIFEST_PATH = ROOT / "experiments/EXP-011-acceptance-convergence/archive-manifest.json"
ARCHIVE_ROOT = ROOT / "results/exp011_sensors_direct"
OUTPUT_PATH = ROOT / "experiments/EXP-011-acceptance-convergence/local-metrics.json"
LEVELS = {
    "coarse": {"sensor_dt_ms": 1.0, "radial": 8, "azimuthal": 16},
    "production": {"sensor_dt_ms": 0.5, "radial": 16, "azimuthal": 32},
    "reference": {"sensor_dt_ms": 0.25, "radial": 32, "azimuthal": 64},
}
CONDITIONS = ("static", "yaw_positive_Z", "yaw_negative_Z",
              "translation_positive_X", "translation_negative_X")
STAGES = ("delayed", "neighbor_correlators", "signed_bd", "local_emissions", "pooled_channels")


def digest_arrays(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(arrays.items()):
        array = np.asarray(value, dtype="<f8", order="C")
        digest.update(key.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def finite_stage(array: np.ndarray) -> bool:
    return bool(np.isfinite(array).all())


def error_summary(a: np.ndarray, b: np.ndarray) -> dict:
    """Return the frozen max-norm metric and compact entity/energy diagnostics."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.ndim < 2 or not finite_stage(a) or not finite_stage(b):
        raise ValueError("coincident finite stage arrays required")
    delta = np.abs(a - b)
    denominator = max(float(np.abs(a).max()), float(np.abs(b).max()), 1e-12)
    entity_delta = delta.reshape(len(delta), delta.shape[1], -1)
    peaks = entity_delta.max(axis=(0, 2))
    energy = np.square(entity_delta).sum(axis=(0, 2))
    order = np.argsort(-energy, kind="stable")
    total = float(energy.sum())
    return {
        "max_abs_difference": float(delta.max()),
        "global_normalizer": denominator,
        "relative_error": float(delta.max()) / denominator,
        "entity_count": int(delta.shape[1]),
        "entities_above_1pct_global": int(np.count_nonzero(peaks > .01 * denominator)),
        "entities_above_5pct_global": int(np.count_nonzero(peaks > .05 * denominator)),
        "entity_peak_p50_p95_max": [float(x) for x in np.percentile(peaks, [50, 95, 100])],
        "squared_error_energy": total,
        "top_10_energy_fraction": float(energy[order[:10]].sum() / total) if total else 0.,
        "top_1pct_energy_fraction": float(energy[order[:max(1, int(np.ceil(len(energy) * .01)))]].sum() / total)
        if total else 0.,
        "facet_400_max_abs": float(peaks[np.flatnonzero(_READOUT.centers == 400)[0]])
        if len(peaks) == len(_READOUT.centers) and np.any(_READOUT.centers == 400) else None,
        "facet_711_max_abs": float(peaks[np.flatnonzero(_READOUT.centers == 711)[0]])
        if len(peaks) == len(_READOUT.centers) and np.any(_READOUT.centers == 711) else None,
    }


def streaming_correlators(samples: np.ndarray, sensor_dt_ms: float, p: SensoryParameters,
                          readout: RetinotopicReadout) -> dict[str, np.ndarray]:
    """Exact streaming form of retinotopy_diagnostics.correlators().

    The recurrence is the same output-before-update recurrence as filter_states:
    the current sample forms q, the delayed state contains the previous q, and
    both states update only after the diagnostic output is recorded.
    """
    samples = np.asarray(samples, float)
    ratio = sensor_dt_ms / p.dt_ms
    if not np.isclose(ratio, round(ratio)) or ratio < 1:
        raise ValueError("sensor clock must be an integer multiple of neural clock")
    repeat = int(round(ratio))
    count = round(p.duration_ms / p.dt_ms)
    stride = round(2.0 / p.dt_ms)
    if len(samples) * repeat < count:
        raise ValueError("sensor samples do not cover neural interval")
    n_saved = (count + stride - 1) // stride
    n_facets = len(samples[0])
    delayed = np.zeros((n_facets, 2), float)
    adaptation = samples[0].copy()
    c_saved = np.empty((n_saved, len(readout.centers), 6, 2), float)
    stencil_saved = np.empty((n_saved, len(readout.centers), 2, 2), float)
    delayed_saved = np.empty((n_saved, n_facets, 2), float)
    ha = p.dt_ms / p.highpass_tau_ms
    hd = p.dt_ms / p.delay_tau_ms
    for k in range(count):
        sample = samples[k // repeat]
        hp = sample - adaptation
        q = np.stack([np.maximum(hp, 0.), np.maximum(-hp, 0.)], axis=-1)
        if k % stride == 0:
            index = k // stride
            c = (delayed[readout.neighbors] * q[readout.centers, None, :]
                 - q[readout.neighbors] * delayed[readout.centers, None, :])
            c_saved[index] = c
            stencil_saved[index] = np.stack([
                (c[:, 0] + c[:, 5] - c[:, 2] - c[:, 3]) / 4,
                (c[:, 1] - c[:, 4]) / 2,
            ], axis=-1)
            delayed_saved[index] = delayed
        adaptation += ha * (sample - adaptation)
        delayed += hd * (q - delayed)
    return {"delayed": delayed_saved, "neighbor_correlators": c_saved, "signed_bd": stencil_saved}


def validate_streaming_equivalence(raw: np.ndarray, sensor_dt_ms: float, readout: RetinotopicReadout,
                                   p: SensoryParameters, prefix_length: int) -> bool:
    """Check the compact streaming diagnostic against the frozen implementation."""
    short = raw[:prefix_length]
    duration = len(short) * sensor_dt_ms
    held = held_light(short, sensor_dt_ms, p.dt_ms, duration)
    prefix_p = replace(p, duration_ms=duration, onset_ms=0., offset_ms=duration)
    direct = correlators(held, readout, prefix_p, stride=round(2 / p.dt_ms))
    streamed = streaming_correlators(short, sensor_dt_ms, prefix_p, readout)
    return all(np.array_equal(direct[key], streamed[key]) for key in streamed)


def load_light(condition: str, level: str) -> np.ndarray:
    path = ARCHIVE_ROOT / f"{condition}_{level}.npz"
    with np.load(path, allow_pickle=False) as data:
        required = {"sensor_time_ms", "light", "mixed", "blocked_mass", "metadata"}
        if set(data.files) != required:
            raise ValueError(f"unexpected archive fields: {path}")
        light = data["light"].copy()
        if not finite_stage(light) or np.any((light < 0) | (light > 1)):
            raise ValueError(f"invalid light archive: {path}")
        return light


def evaluate_bank(raw: np.ndarray, sensor_dt_ms: float, p: SensoryParameters,
                  readout: RetinotopicReadout) -> dict[str, np.ndarray]:
    held = held_light(raw, sensor_dt_ms, p.dt_ms, p.duration_ms)
    save_stride = round(2.0 / p.dt_ms)
    channels, local = readout.responses(held, p, save_stride=save_stride)
    pooled = readout.pool(local)
    if not np.array_equal(pooled, channels[::save_stride]):
        raise RuntimeError("local/pool coincidence invariant failed")
    stages = streaming_correlators(raw, sensor_dt_ms, p, readout)
    stages["local_emissions"] = local
    stages["pooled_channels"] = pooled
    stages["channels"] = channels
    return stages


def compare_banks(a: dict[str, np.ndarray], b: dict[str, np.ndarray], *, include_light=False) -> dict:
    result = {}
    for key in STAGES:
        result[key] = error_summary(a[key], b[key])
    if include_light:
        result["physical_light"] = error_summary(a["light"], b["light"])
    return result


def bank_with_light(raw: np.ndarray, sensor_dt_ms: float, p: SensoryParameters,
                    readout: RetinotopicReadout) -> dict[str, np.ndarray]:
    bank = evaluate_bank(raw, sensor_dt_ms, p, readout)
    bank["light"] = raw
    return bank


def pair_from_raw(raw_a: np.ndarray, dt_a: float, raw_b: np.ndarray, dt_b: float,
                  p: SensoryParameters, readout: RetinotopicReadout) -> dict:
    a = bank_with_light(raw_a, dt_a, p, readout)
    b = bank_with_light(raw_b, dt_b, p, readout)
    result = compare_banks(a, b, include_light=False)
    if np.isclose(dt_a, dt_b):
        coincident_b = raw_b
    else:
        ratio = dt_a / dt_b
        if not np.isclose(ratio, round(ratio)) or ratio < 1:
            raise ValueError("light clocks do not have a nested comparison")
        coincident_b = raw_b[::int(round(ratio))]
    if raw_a.shape != coincident_b.shape:
        raise ValueError("coincident raw light arrays required")
    result["physical_light"] = error_summary(raw_a, coincident_b)
    del a, b
    return result


def verify_frozen_sources(spec: dict) -> dict:
    checks = {}
    for entry in spec["frozen_files"]:
        path = ROOT / entry["path"]
        checks[entry["path"]] = sha256_file(path) == entry["sha256"]
    return checks


def verify_archive_manifest() -> dict:
    """Verify every canonical archive before any metric array is loaded."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf8"))
    if manifest["source_commit"] != "5e9ec94d9c982615441ab67f1ffca9f9d783c636":
        raise RuntimeError("archive manifest source checkpoint changed")
    expected = {entry["name"]: entry for entry in manifest["archives"]}
    actual = sorted(ARCHIVE_ROOT.glob("*.npz"))
    if sorted(path.name for path in actual) != sorted(expected):
        raise RuntimeError("canonical archive set differs from manifest")
    checks = {}
    for path in actual:
        entry = expected[path.name]
        actual_sha256 = sha256_file(path)
        checks[path.name] = {
            "bytes": path.stat().st_size,
            "expected_bytes": entry["bytes"],
            "sha256": actual_sha256,
            "expected_sha256": entry["sha256"],
            "passed": bool(path.stat().st_size == entry["bytes"] and actual_sha256 == entry["sha256"]),
        }
    if not all(value["passed"] for value in checks.values()):
        raise RuntimeError("canonical archive SHA-256 verification failed")
    return {"path": str(MANIFEST_PATH.relative_to(ROOT)).replace("\\", "/"),
            "source_commit": manifest["source_commit"],
            "archive_count": manifest["archive_count"],
            "total_bytes": manifest["total_bytes"],
            "all_sha256_verified": True,
            "files": checks}


def eye_to_local_disconnection(readout: RetinotopicReadout, bank: dict[str, np.ndarray]) -> dict[str, float]:
    """Sever the eye-to-local interface; unlike a constant-input null, no light enters it."""
    disconnected_local = np.zeros_like(bank["local_emissions"])
    disconnected_pooled = readout.pool(disconnected_local)
    disconnected = {
        "local_emissions": disconnected_local,
        "pooled_channels": disconnected_pooled,
        "delayed": np.zeros_like(bank["delayed"]),
        "neighbor_correlators": np.zeros_like(bank["neighbor_correlators"]),
        "signed_bd": np.zeros_like(bank["signed_bd"]),
    }
    return {key: float(np.abs(value).max()) for key, value in disconnected.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    spec = json.loads(SPEC_PATH.read_text(encoding="utf8"))
    source_checks = verify_frozen_sources(spec)
    if not all(source_checks.values()):
        raise RuntimeError("frozen EXP-011 source integrity failed")
    archive_manifest = verify_archive_manifest()
    _, eye_spec, geometry, _ = load_resolved()
    p = SensoryParameters(**eye_spec["early_vision"]["parameters"])
    neural = replace(p, dt_ms=0.25)
    neural_half = replace(p, dt_ms=0.125)
    global _READOUT
    _READOUT = RetinotopicReadout.compile(geometry)

    # Check the independent streaming diagnostic at representative conditions
    # and all three optical resolutions before accepting full metrics.
    prefix_specs = (
        ("static", "coarse", 16),
        ("yaw_positive_Z", "production", 32),
        ("yaw_negative_Z", "reference", 32),
        ("translation_positive_X", "coarse", 16),
        ("translation_negative_X", "reference", 32),
    )
    prefix_checks = []
    for condition, level, prefix_length in prefix_specs:
        prefix = load_light(condition, level)
        passed = validate_streaming_equivalence(
            prefix, LEVELS[level]["sensor_dt_ms"], _READOUT, neural, prefix_length)
        prefix_checks.append({"condition": condition, "level": level,
                              "prefix_samples": prefix_length, "passed": passed})
    streaming_ok = all(row["passed"] for row in prefix_checks)
    if not streaming_ok:
        raise RuntimeError("streaming correlator equivalence failed on a representative prefix")

    raw = {condition: {level: load_light(condition, level) for level in LEVELS}
           for condition in CONDITIONS}
    measurements = {"optical": {}, "temporal": {}, "combined": {}, "neural_half_step": {}}
    controls = {}
    digests = {}
    finite = True

    for condition in CONDITIONS:
        print(f"metrics {condition}", flush=True)
        coarse, production, reference = (raw[condition][x] for x in ("coarse", "production", "reference"))

        # Isolate optical quadrature at coincident physical poses: both inputs
        # are placed on the coarser shared physical clock before the identical
        # 0.25 ms neural hold.
        measurements["optical"][condition] = {
            "coarse_to_production": pair_from_raw(coarse, 1.0, production[::2], 1.0, neural, _READOUT),
            "production_to_reference": pair_from_raw(production, 0.5, reference[::2], 0.5, neural, _READOUT),
        }

        # Temporal refinement uses the same reference optical samples at exact
        # nested clocks and the same 0.25 ms neural update.
        measurements["temporal"][condition] = {
            "1.0_to_0.5ms": pair_from_raw(reference[::4], 1.0, reference[::2], 0.5, neural, _READOUT),
            "0.5_to_0.25ms": pair_from_raw(reference[::2], 0.5, reference, 0.25, neural, _READOUT),
        }

        # Native levels are used for the combined candidate comparison.
        measurements["combined"][condition] = {
            "coarse_to_production": pair_from_raw(coarse, 1.0, production, 0.5, neural, _READOUT),
            "production_to_reference": pair_from_raw(production, 0.5, reference, 0.25, neural, _READOUT),
        }

        # Neural half-step uses identical production sensory samples and only
        # changes the local neural update clock.
        prod = evaluate_bank(production, 0.5, neural, _READOUT)
        half = evaluate_bank(production, 0.5, neural_half, _READOUT)
        measurements["neural_half_step"][condition] = {
            key: error_summary(prod[key], half[key]) for key in STAGES
        }
        digests[condition] = {
            "production_neural": digest_arrays({k: prod[k] for k in STAGES}),
            "neural_half": digest_arrays({k: half[k] for k in STAGES}),
        }

        # Static, null-input, true eye-to-local disconnection, and pre-motion
        # causal controls use only the declared local readout; no downstream
        # stages are instantiated.
        controls[condition] = {}
        for level, dt in (("coarse", 1.0), ("production", 0.5), ("reference", 0.25)):
            samples = raw[condition][level]
            bank = evaluate_bank(samples, dt, neural, _READOUT)
            constant = np.repeat(samples[:1], len(samples), axis=0)
            null = evaluate_bank(constant, dt, neural, _READOUT)
            disconnection = eye_to_local_disconnection(_READOUT, bank)
            entry = {
                "finite_local": finite_stage(bank["local_emissions"]),
                "finite_pooled_channels": finite_stage(bank["pooled_channels"]),
                "constant_input_null_max_abs_local": float(np.abs(null["local_emissions"]).max()),
                "constant_input_null_max_abs_pooled": float(np.abs(null["pooled_channels"]).max()),
                "eye_to_local_disconnection_max_abs": disconnection,
                "static_condition_max_abs_local": float(np.abs(bank["local_emissions"]).max()) if condition == "static" else None,
                "static_condition_max_abs_pooled": float(np.abs(bank["pooled_channels"]).max()) if condition == "static" else None,
            }
            if condition != "static":
                before = bank["local_emissions"][:round(2000 / 2.0)]
                entry["response_before_pose_change_max_abs_local"] = float(np.abs(before).max())
            else:
                entry["response_before_pose_change_max_abs_local"] = 0.0
            controls[condition][level] = entry
            finite &= bool(entry["finite_local"] and entry["finite_pooled_channels"])
            del bank, null, constant, disconnection
        del prod, half

    def trend(bank: dict, first: str, second: str) -> dict:
        rows = {}
        for stage in ("physical_light",) + STAGES:
            a = bank[first][stage]["max_abs_difference"]
            b = bank[second][stage]["max_abs_difference"]
            rows[stage] = {"preceding": a, "refined": b, "passed": bool(b <= a + 1e-12)}
        return {"stages": rows, "passed": all(row["passed"] for row in rows.values())}

    trend_results = {}
    for category in ("optical", "temporal", "combined"):
        trend_results[category] = {}
        for condition in CONDITIONS:
            trend_results[category][condition] = trend(measurements[category][condition],
                                                         "coarse_to_production" if category != "temporal" else "1.0_to_0.5ms",
                                                         "production_to_reference" if category != "temporal" else "0.5_to_0.25ms")

    # Candidate gate is reported, not used to trigger any downstream work.
    # Only final separated refinements and production/reference combined
    # refinement belong here; preceding pairs remain diagnostics/trends.
    gate_rows = {}
    for condition in CONDITIONS:
        rows = {
            "combined_production_to_reference": {
                stage: bool(measurements["combined"][condition]["production_to_reference"][stage]["relative_error"] <= .05)
                for stage in STAGES
            },
            "optical_production_to_reference": {
                stage: bool(measurements["optical"][condition]["production_to_reference"][stage]["relative_error"] <= .05)
                for stage in STAGES
            },
            "temporal_0.5_to_0.25ms": {
                stage: bool(measurements["temporal"][condition]["0.5_to_0.25ms"][stage]["relative_error"] <= .05)
                for stage in STAGES
            },
            "neural_half_step": {
                stage: bool(measurements["neural_half_step"][condition][stage]["relative_error"] <= .05)
                for stage in STAGES
            },
        }
        rows["production_reference_physical_light_1e-4"] = bool(
            measurements["combined"][condition]["production_to_reference"]["physical_light"]["max_abs_difference"] <= 1e-4)
        rows["trends"] = bool(all(trend_results[c][condition]["passed"] for c in trend_results))
        gate_rows[condition] = rows

    control_passed = all(
        entry["finite_local"] and entry["finite_pooled_channels"]
        and entry["constant_input_null_max_abs_local"] == 0.0
        and entry["constant_input_null_max_abs_pooled"] == 0.0
        and all(value == 0.0 for value in entry["eye_to_local_disconnection_max_abs"].values())
        and entry["static_condition_max_abs_local"] in (None, 0.0)
        and entry["static_condition_max_abs_pooled"] in (None, 0.0)
        and entry["response_before_pose_change_max_abs_local"] == 0.0
        for condition in controls.values() for entry in condition.values()
    )
    for condition in CONDITIONS:
        gate_rows[condition]["controls"] = control_passed
    candidate_checks = []
    for row in gate_rows.values():
        for category in ("combined_production_to_reference", "optical_production_to_reference",
                         "temporal_0.5_to_0.25ms", "neural_half_step"):
            candidate_checks.extend(row[category].values())
        candidate_checks.extend((row["production_reference_physical_light_1e-4"],
                                 row["trends"], row["controls"]))
    candidate_passed = bool(all(candidate_checks) and control_passed and streaming_ok and finite)

    report = {
        "experiment_id": spec["experiment_id"],
        "status": "diagnostic_continuation_after_visibility_verifier_failure_corrected",
        "scope": "immutable canonical light arrays only; no rerendering and no downstream LPi/HS/H2/DNp15 evaluation",
        "canonical_initial_verifier_failure_preserved": True,
        "verifier_failure_field": "blocked_mass",
        "verifier_failure_is_not_rescued": True,
        "frozen_levels": LEVELS,
        "common_local_output_ms": 2.0,
        "neural_dt_ms": {"candidate": 0.25, "half_step_reference": 0.125},
        "frozen_source_checks": source_checks,
        "archive_manifest": archive_manifest,
        "streaming_correlator_equivalence_prefixes": prefix_checks,
        "streaming_correlator_equivalence_prefix": streaming_ok,
        "measurements": measurements,
        "trend": trend_results,
        "controls": controls,
        "control_gate_passed": control_passed,
        "finite_stage_arrays": finite,
        "candidate_local_gate_passed": candidate_passed,
        "gate_rows": gate_rows,
        "deterministic_stage_digests": digests,
        "archive_paths": [str(ARCHIVE_ROOT / f"{c}_{level}.npz") for c in CONDITIONS for level in LEVELS],
        "implementation": {
            "script": "scripts/analyze_exp011_canonical_metrics.py",
            "candidate_gate_pairs": {
                "combined": "production_to_reference",
                "optical": "production_to_reference",
                "temporal": "0.5_to_0.25ms",
                "neural": "production sensory samples at 0.25 vs 0.125 ms",
            },
            "preceding_pairs_retained_as_diagnostics": True,
            "constant_input_null_separate_from_eye_to_local_disconnection": True,
            "frozen_optical_archives_unchanged": True,
            "no_visibility_input_to_local_transform": True,
            "no_downstream_evaluation": True,
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"candidate_local_gate_passed": candidate_passed,
                      "control_gate_passed": control_passed,
                      "streaming_correlator_equivalence_prefix": streaming_ok}, indent=2))


if __name__ == "__main__":
    main()
