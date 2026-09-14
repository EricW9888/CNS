"""Sample only the new half-millisecond poses; reuse verified EXP-008 light."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_exp008_eye import replay_sensor, trace_digest
from scripts.run_exp009_retinotopy import load_resolved
from src.compound_eye import EyeSampler, Scene, rotation_z
from src.sensory_chain import SensoryParameters

SPEC = ROOT / "experiments/EXP-010-retinotopic-refinement/specification.json"


def sample_reference(sampler, scene, p, pose, old_light):
    """Exact old samples at 1ms; fresh scalar ray samples at odd half-ms times."""
    count = round(p.duration_ms / .5)
    if old_light.shape != (count // 2, len(sampler.geometry.directions)):
        raise ValueError("invalid 1ms light shape")
    result = np.empty((count, old_light.shape[1]))
    result[::2] = old_light
    time = np.arange(count) * .5
    elapsed = np.clip(time - p.onset_ms, 0, p.offset_ms - p.onset_ms) / 1000
    cache = {}
    for k in range(1, count, 2):
        seconds = float(elapsed[k])
        if seconds not in cache:
            cache[seconds] = sampler.sample(scene, rotation_z(np.deg2rad(pose[0]) * seconds),
                                            [pose[1] * seconds, 0., 0.])
            if len(cache) % 250 == 0:
                print(f"  new poses {len(cache)}/2002", flush=True)
        result[k] = cache[seconds]
    # Fresh checks include old, new, baseline and held poses. No interpolation.
    for k in (0, 4001, 6000, 7773, 9000):
        seconds = elapsed[k]
        fresh = sampler.sample(scene, rotation_z(np.deg2rad(pose[0]) * seconds), [pose[1] * seconds, 0., 0.])
        if not np.array_equal(fresh, result[k]):
            raise ValueError("sample reuse differs from direct ray replay")
    return time, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensors", type=Path, default=ROOT / "results/exp008_eye_final")
    parser.add_argument("--output", type=Path, default=ROOT / "results/exp010_reference")
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to((ROOT / "results").resolve()):
        raise ValueError("sensory bulk must remain under ignored results")
    _, eye_spec, g, _ = load_resolved()
    spec = json.loads(SPEC.read_text())
    old_record = json.loads((ROOT / "experiments/EXP-008-compound-eye/record.json").read_text())
    p = SensoryParameters(**eye_spec["early_vision"]["parameters"])
    sampler = EyeSampler.compile(g, **{k: eye_spec["optics"][k] for k in ("fwhm_degrees", "radial", "azimuthal")})
    scene = Scene(**eye_spec["scene"])
    args.output.mkdir(parents=True, exist_ok=True)
    for condition in spec["conditions"]:
        if condition == "static":
            continue
        target = args.output / f"{condition}.npz"
        if target.exists():
            print(f"Existing reference: {condition}; verification belongs to result replay", flush=True)
            continue
        _, light = replay_sensor(args.sensors, condition, g, p, 1., refinement=True)
        if trace_digest({"sensor_light": light}) != old_record["sensor_refinement_digests"][condition]:
            raise ValueError("source light differs from EXP-008")
        print(f"Rendering fixed reference: {condition}", flush=True)
        time, fine = sample_reference(sampler, scene, p, eye_spec["pose_conditions"][condition], light)
        np.savez_compressed(target, sensor_time_ms=time, light=fine)
        print(f"Saved {condition}: {trace_digest({'sensor_light': fine})}", flush=True)


if __name__ == "__main__":
    main()
