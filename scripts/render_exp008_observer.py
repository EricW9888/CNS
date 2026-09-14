"""Read-only EXP-008 observer: saved scalar facets and neural traces, no simulation.

The cutaway scene is reconstructed for humans from the frozen scene/pose
description. It is never an input to a sensor, motion kernel, or neural model.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter
import numpy as np
from PIL import Image, __version__ as PILLOW_VERSION

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_exp008_eye import evidence_sha256
from scripts.run_exp008_eye import trace_digest
from src.compound_eye import EyeGeometry, LocalReadout, Scene, rotation_z
from src.provenance import sha256_file
from src.sensory_chain import CHANNELS

SPEC = ROOT/"experiments/EXP-008-compound-eye/specification.json"
RECORD = SPEC.with_name("record.json")
STAGES = ("LPi", "HS/H2 drive", "HS", "H2", "H2rn", "uLPTCrn", "bIPS")
LEFT, RIGHT = "#007f91", "#cf581f"
INK, MUTED, PAPER = "#172a38", "#5c6b75", "#f7f9fa"
MAX_MEDIA_BYTES = 5*1024*1024
FRAME_INTERVAL_MS = 60


@dataclass(frozen=True)
class SavedTrial:
    spec: dict
    record: dict
    condition: str
    geometry: EyeGeometry
    axes: dict
    arrays: dict
    central_rows: tuple
    LPi_rows: tuple

    def indices(self, time_ms):
        """Last recorded sample at/before display time; never interpolate values."""
        if not np.isfinite(time_ms) or not 0 <= time_ms < self.spec["early_vision"]["parameters"]["duration_ms"]:
            raise ValueError("display time outside saved trial")
        return tuple(int(np.searchsorted(self.arrays[k], time_ms, side="right")-1)
                     for k in ("sensor_time_ms", "time_ms"))

    def pose(self, time_ms):
        p = self.spec["early_vision"]["parameters"]
        seconds = np.clip(time_ms-p["onset_ms"], 0, p["offset_ms"]-p["onset_ms"])/1000
        rate, speed = self.spec["pose_conditions"][self.condition]
        return rotation_z(np.deg2rad(rate)*seconds), np.array([speed*seconds, 0., 0.])

    def stage_means(self, neural_index):
        """Display-only signed arithmetic means; DN cells remain separate/raw."""
        rows = []
        for stage in STAGES:
            if stage == "LPi":
                metadata, values = self.LPi_rows, self.arrays["LPi"][neural_index]
                masks = [[r["somaSide"] == side for r in metadata] for side in ("L", "R")]
            elif stage == "HS/H2 drive":
                lookup = {r["body_id"]: r for r in self.central_rows}
                values = self.arrays["HS_H2_drive"][neural_index]
                masks = [[lookup[int(i)]["side"] == side for i in self.arrays["HS_H2_body_ids"]]
                         for side in ("L", "R")]
            else:
                values = self.arrays["central"][neural_index]
                masks = [[r["stage"] == stage and r["side"] == side for r in self.central_rows]
                         for side in ("L", "R")]
            if not all(any(mask) for mask in masks):
                raise ValueError("missing displayed stage/side")
            rows.append([float(values[np.asarray(mask)].mean()) for mask in masks])
        return np.asarray(rows)


def load_saved(folder, condition="yaw_positive_Z"):
    spec, record = (json.loads(p.read_text(encoding="utf8")) for p in (SPEC, RECORD))
    if sha256_file(SPEC) != record["specification_sha256"]:
        raise ValueError("specification differs from frozen EXP-008")
    for path, expected in record["implementation_sha256"].items():
        if evidence_sha256(ROOT/path) != expected:
            raise ValueError(f"frozen implementation changed: {path}")
    for entry in spec["frozen_files"]:
        if evidence_sha256(ROOT/entry["path"]) != entry["sha256"]:
            raise ValueError(f"frozen baseline changed: {entry['path']}")
    if condition not in record["deterministic_trace_digests"]:
        raise ValueError("condition is not recorded")
    with np.load(Path(folder)/f"{condition}.npz", allow_pickle=False) as saved:
        arrays = {k: saved[k].copy() for k in saved.files}
    with np.load(Path(folder)/"sensor_axes.npz", allow_pickle=False) as saved:
        axes = {k: saved[k].copy() for k in saved.files}
    stages = ("channels", "LPi", "HS_H2_drive", "central")
    if trace_digest({"sensor_light": arrays["light"], **{k: arrays[k] for k in stages}}) != record["deterministic_trace_digests"][condition]:
        raise ValueError("saved sensor/neural values differ from frozen trace digest")
    p = spec["early_vision"]["parameters"]
    for key, dt in [("sensor_time_ms", spec["sensor_dt_ms"]), ("time_ms", p["dt_ms"])]:
        if not np.array_equal(arrays[key], np.arange(round(p["duration_ms"]/dt))*dt):
            raise ValueError("saved time axis differs from EXP-008")
    source = spec["eye_source"]["files"][-1]
    if sha256_file(ROOT/source["path"]) != source["sha256"]:
        raise ValueError("measured source geometry differs from frozen provenance")
    geometry = EyeGeometry.load(ROOT/source["path"])
    for k, value in geometry.__dict__.items():
        if not np.array_equal(value, axes[k]):
            raise ValueError(f"saved sensor axes differ from measured geometry: {k}")
    readout = LocalReadout.compile(geometry, **{k: spec["early_vision"][k] for k in ("azimuth_range", "elevation_limit")})
    for side, centers, endpoints in zip(("left", "right"), readout.centers, readout.endpoints):
        if not (np.array_equal(axes[f"{side}_readout_centers"], centers)
                and np.array_equal(axes[f"{side}_readout_endpoints"], endpoints)):
            raise ValueError("saved readout mask/endpoints differ from EXP-008")
    rows = tuple(record["central_per_body"][condition])
    LPi_rows = tuple(json.loads((ROOT/"experiments/EXP-005-sensory-to-DNp15/specification.json").read_text(encoding="utf8"))["LPi_identity"])
    expected_ids = {"central_body_ids": [r["body_id"] for r in rows],
                    "LPi_body_ids": [r["bodyId"] for r in LPi_rows],
                    "HS_H2_body_ids": [r["body_id"] for r in rows if r["stage"] in {"HS", "H2"}]}
    for k, ids in expected_ids.items():
        if not np.array_equal(arrays[k], ids):
            raise ValueError(f"unverified neuron order: {k}")
    if not np.array_equal(arrays["channel_names"], CHANNELS):
        raise ValueError("unverified motion channel order")
    if arrays["light"].shape != (len(arrays["sensor_time_ms"]), len(geometry.directions)):
        raise ValueError("invalid facet axis")
    for k, id_key in [("channels", "channel_names"), ("LPi", "LPi_body_ids"),
                      ("HS_H2_drive", "HS_H2_body_ids"), ("central", "central_body_ids")]:
        if arrays[k].shape != (len(arrays["time_ms"]), len(arrays[id_key])):
            raise ValueError(f"invalid stage shape: {k}")
    for a in (*arrays.values(), *axes.values(), *geometry.__dict__.values()):
        a.setflags(write=False)
    return SavedTrial(spec, record, condition, geometry, axes, arrays, rows, LPi_rows)


def facet_projection(directions, left):
    """Lambert equal-area projection of measured axes around the lateral eye axis.

    Both panels use anterior (+X) rightwards and dorsal (+Z) upwards. No facet
    directions are reflected, gridded, interpolated, or replaced with pixels.
    """
    u = np.asarray(directions)
    denominator = 1+(1 if left else -1)*u[:, 1]
    if np.any(denominator <= 0):
        raise ValueError("projection contains an antipodal viewing direction")
    return np.sqrt(2/denominator)[:, None]*u[:, [0, 2]]


class Observer:
    def __init__(self, trial):
        self.trial = trial
        self.fig = plt.figure(figsize=(12.8, 8.2), dpi=100, facecolor=PAPER)
        self.fig.text(.035, .954, "A world becomes neural activity", color=INK, fontsize=24, weight="bold")
        rate = trial.spec["pose_conditions"][trial.condition][0]
        self.fig.text(.035, .917, f"EXP-008  /  prescribed Z rotation {rate:+g}°/s  /  saved signals only  /  no motor feedback",
                      color=MUTED, fontsize=11)
        self.clock = self.fig.text(.97, .954, "", ha="right", color=INK, fontsize=13, weight="bold")
        for x, title, caption in [(.035, "01  OBSERVER SCENE", "For humans only · schematic pose"),
                                  (.385, "02  FLY SENSORY INPUT", "Measured compound-eye sampled light"),
                                  (.72, "03  NEURAL PATHWAY", "Provisional motion → identified downstream cells")]:
            self.fig.text(x, .838, title, color=INK, fontsize=12, weight="bold")
            self.fig.text(x, .812, caption, color=MUTED, fontsize=9)
        self.fig.text(.355, .585, "→", color=MUTED, fontsize=25)
        self.fig.text(.667, .585, "→", color=MUTED, fontsize=25)
        self.timeline = self.fig.add_axes([.385, .878, .585, .009])
        self.timeline.set_xlim(0, 6); self.timeline.set_ylim(0, 1); self.timeline.axis("off")
        for lo, hi, color in [(0, 2, "#dbe3e8"), (2, 4, "#d6eadc"), (4, 6, "#dbe3e8")]:
            self.timeline.axvspan(lo, hi, color=color)
        self.time_marker = self.timeline.axvline(0, color=INK, lw=2)
        self.fig.text(.385, .898, "Baseline 0–2 s         Pose change 2–4 s         Final-pose hold 4–6 s", color=MUTED, fontsize=8)
        self.scene_ax = self.fig.add_axes([.025, .32, .32, .465], projection="3d", facecolor=PAPER)
        self._scene()
        self.fig.text(.035, .298, "10-mm painted wall (cutaway) + opaque occluder", color=MUTED, fontsize=9)
        self.fig.text(.035, .276, "Head glyph enlarged; axes/rays are observer overlays.", color=MUTED, fontsize=8)
        self.pose_label = self.fig.text(.035, .224, "", color=INK, fontsize=10)
        self.eye_artists = []
        for side, left, y, color in [("left", True, .54, LEFT), ("right", False, .255, RIGHT)]:
            ax = self.fig.add_axes([.385, y, .255, .24], facecolor=PAPER)
            ids = np.flatnonzero(trial.geometry.left == left)
            xy = facet_projection(trial.geometry.directions[ids], left)
            scatter = ax.scatter(*xy.T, c=np.zeros(len(ids)), cmap="gray", norm=Normalize(0, 1),
                                 s=14, edgecolors="#b7c0c6", linewidths=.2)
            centers = trial.axes[f"{side}_readout_centers"]
            flanks = np.setdiff1d(np.unique(trial.axes[f"{side}_readout_endpoints"]), centers)
            for used, size, width in [(flanks, 19, .65), (centers, 23, 1.)]:
                projected = facet_projection(trial.geometry.directions[used], left)
                ax.scatter(*projected.T, facecolors="none", edgecolors=color, s=size, linewidths=width)
            ax.set(xlim=(-1.6, 1.6), ylim=(-1.6, 1.6), aspect="equal", xticks=[], yticks=[])
            for spine in ax.spines.values(): spine.set_visible(False)
            ax.set_title(f"{side.upper()} EYE · {len(ids)} facets · {len(centers)} readout centers", fontsize=9, color=color, pad=2)
            ax.text(.98, .02, "anterior →", transform=ax.transAxes, ha="right", color=MUTED, fontsize=7)
            ax.text(.02, .98, "↑ dorsal", transform=ax.transAxes, va="top", color=MUTED, fontsize=7)
            self.eye_artists.append((ids, scatter))
        self.fig.text(.385, .236, "Rings = readout centers; thin colored rims = input flanks.", color=MUTED, fontsize=8)
        self.fig.text(.385, .216, "Other facets displayed only. Light scale fixed at 0 black → 1 white.", color=MUTED, fontsize=8)
        self.motion_ax = self.fig.add_axes([.742, .637, .228, .10], facecolor=PAPER)
        self.motion = self.motion_ax.imshow(np.zeros((2, 8)), cmap="YlOrBr", aspect="auto",
                                            vmin=0, vmax=max(trial.arrays["channels"].max(), 1e-12))
        self.motion_ax.set(yticks=[0, 1], yticklabels=["L", "R"], xticks=range(8),
                           xticklabels=["a", "b", "c", "d", "a", "b", "c", "d"])
        self.motion_ax.tick_params(length=0, labelsize=9)
        self.motion_ax.axvline(3.5, color="white", lw=3)
        counts = [len(trial.axes[f"{side}_readout_centers"]) for side in ("left", "right")]
        self.motion_ax.set_title(f"Provisional local motion · {counts[0]} L / {counts[1]} R centers", fontsize=9, color=INK, pad=22)
        self.motion_ax.text(.25, 1.08, "ON / T4 proxy", transform=self.motion_ax.transAxes, ha="center", fontsize=8, color=MUTED)
        self.motion_ax.text(.75, 1.08, "OFF / T5 proxy", transform=self.motion_ax.transAxes, ha="center", fontsize=8, color=MUTED)
        self.fig.text(.742, .602, f"Fixed channel scale: 0 → {trial.arrays['channels'].max():.3g}", fontsize=8, color=MUTED)
        self.stage_ax = self.fig.add_axes([.79, .29, .18, .265], facecolor=PAPER)
        self.stage_ax.axvline(0, color="#b7c0c6", lw=.8)
        self.stage_markers = []
        for side, color, offset in [("L", LEFT, -.12), ("R", RIGHT, .12)]:
            line, = self.stage_ax.plot(np.zeros(len(STAGES)), np.arange(len(STAGES))+offset,
                                       "o", color=color, ms=5, label=side)
            self.stage_markers.append(line)
        limit = max(abs(trial.arrays[k]).max() for k in ("LPi", "HS_H2_drive", "central"))
        self.stage_ax.set(xlim=(-1.05*limit, 1.05*limit), ylim=(len(STAGES)-.5, -.5),
                          yticks=range(len(STAGES)), yticklabels=STAGES)
        self.stage_ax.tick_params(length=0, labelsize=8)
        self.stage_ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x*1000:g}"))
        self.fig.text(.742, .573, "Downstream stages", fontsize=10, color=INK)
        self.fig.text(.89, .573, "● L", fontsize=9, color=LEFT)
        self.fig.text(.935, .573, "● R", fontsize=9, color=RIGHT)
        self.stage_ax.set_xlabel("Signed means ×10⁻³ · fixed shared scale", fontsize=8, color=MUTED)
        for spine in self.stage_ax.spines.values(): spine.set_visible(False)
        self.fig.text(.035, .191, "04  DNp15", color=INK, fontsize=15, weight="bold")
        self.fig.text(.035, .166, "Identified cells", color=MUTED, fontsize=9)
        self.dn_labels, self.dn_lines = [], []
        self.dn_ax = self.fig.add_axes([.21, .10, .76, .105], facecolor=PAPER)
        self.dn_ax.axvspan(2, 4, color="#d6eadc", alpha=.55)
        self.dn_ax.axhline(0, color="#b7c0c6", lw=.6)
        self.dn_cursor = self.dn_ax.axvline(0, color=INK, lw=1)
        time = trial.arrays["time_ms"]/1000
        for side, color, y in [("L", LEFT, .126), ("R", RIGHT, .101)]:
            index = next(i for i, r in enumerate(trial.central_rows) if r["stage"] == "DNp15" and r["side"] == side)
            values = trial.arrays["central"][:, index]
            self.dn_ax.plot(time, values, color=color, alpha=.17, lw=1)
            line, = self.dn_ax.plot([], [], color=color, lw=1.8)
            self.dn_lines.append((index, line))
            label = self.fig.text(.035, y, "", fontsize=9, color=color)
            self.dn_labels.append((index, label))
        indices = [i for i, _ in self.dn_lines]
        values = trial.arrays["central"][:, indices]
        lo, hi = values.min(), values.max(); margin=max(hi-lo, 1e-12)*.13
        self.dn_ax.set(xlim=(0, 6), ylim=(lo-margin, hi+margin), xlabel="Saved experiment time (s)")
        self.dn_ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        self.dn_ax.tick_params(labelsize=8, colors=MUTED)
        for spine in self.dn_ax.spines.values(): spine.set_color("#dbe3e8")
        self.fig.text(.035, .023, "Measured geometry · assumed 5° optics · central-local readout only · dimensionless neural proxies, not voltage or behavior",
                      fontsize=9, color=MUTED)

    def _scene(self):
        scene = Scene(**self.trial.spec["scene"])
        # Explicit overlay layers avoid mplot3d hiding pose/rays behind its cutaway.
        self.scene_ax.computed_zorder = False
        # Observer cutaway: remove the near hemisphere for visibility, never in the model.
        az, el = np.meshgrid(np.linspace(-np.pi/2, np.pi/2, 81), np.linspace(-np.pi/2, np.pi/2, 33))
        u = np.stack([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)], axis=-1)
        paint = replace(scene, occluder_radius_mm=0).radiance(np.zeros_like(u), u)
        self.scene_ax.plot_surface(*(scene.radius_mm*u).transpose(2, 0, 1),
                                   facecolors=plt.cm.gray(paint), rstride=1, cstride=1,
                                   shade=False, linewidth=0, antialiased=False, zorder=1)
        theta, phi = np.meshgrid(np.linspace(0, np.pi, 13), np.linspace(0, 2*np.pi, 25))
        sphere = np.stack([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)], axis=-1)
        points = np.asarray(scene.occluder_center)+scene.occluder_radius_mm*sphere
        self.scene_ax.plot_surface(*points.transpose(2, 0, 1), color="#19242d", shade=False, linewidth=0, zorder=2)
        self.scene_ax.text(4., 0., 1.6, "Occluder", color=INK, fontsize=8, zorder=5)
        theta = np.linspace(0, 2*np.pi, 61)
        self.head_shape = np.column_stack([1.0*np.cos(theta), 1.4*np.sin(theta), np.zeros_like(theta)])
        self.head_line, = self.scene_ax.plot([], [], [], color=INK, lw=1.5, zorder=4)
        self.heading_line, = self.scene_ax.plot([], [], [], color="#8651af", lw=2.5, marker="o", markevery=[1], ms=4, zorder=5)
        self.eye_glyphs, self.view_rays = [], []
        for left, color in [(True, LEFT), (False, RIGHT)]:
            ids = np.flatnonzero(self.trial.geometry.left == left)
            glyph, = self.scene_ax.plot([], [], [], ".", color=color, ms=1.8, zorder=4)
            self.eye_glyphs.append((ids, glyph))
            center = self.trial.axes[f"{'left' if left else 'right'}_readout_centers"][len(self.trial.axes[f"{'left' if left else 'right'}_readout_centers"])//2]
            ray, = self.scene_ax.plot([], [], [], color=color, lw=1., alpha=.8, zorder=5)
            self.view_rays.append((center, ray))
        self.scene_ax.set(xlim=(-4, 10), ylim=(-10, 10), zlim=(-10, 10), box_aspect=(14, 20, 20))
        self.scene_ax.plot([0., 4.], [0., 0.], [-1., -1.], color=MUTED, lw=.8, ls="--", zorder=3)
        self.scene_ax.text(4.3, 0., -1., "+X", color=MUTED, fontsize=8, zorder=5)
        self.scene_ax.view_init(elev=23, azim=-110)
        self.scene_ax.set_axis_off()

    def draw(self, time_ms):
        trial = self.trial
        sensor_index, neural_index = trial.indices(time_ms)
        p = trial.spec["early_vision"]["parameters"]
        phase = "BASELINE" if time_ms < p["onset_ms"] else "POSE CHANGE" if time_ms < p["offset_ms"] else "FINAL-POSE HOLD"
        self.clock.set_text(f"{time_ms/1000:0.2f} s  ·  {phase}")
        self.time_marker.set_xdata([time_ms/1000]*2)
        r, t = trial.pose(time_ms)
        for points, line in [(self.head_shape, self.head_line), (np.array([[0., 0., 0.], [3., 0., 0.]]), self.heading_line)]:
            line.set_data_3d(*(points@r.T+t).T)
        for ids, line in self.eye_glyphs:
            line.set_data_3d(*(5*trial.geometry.lens_mm[ids]@r.T+t).T)
        for i, line in self.view_rays:
            origin = trial.geometry.lens_mm[i]@r.T+t
            line.set_data_3d(*np.stack([origin, origin+4*(trial.geometry.directions[i]@r.T)]).T)
        angle = np.rad2deg(np.arctan2(r[1, 0], r[0, 0]))
        self.pose_label.set_text(f"Prescribed head angle: {angle:0.1f}° about physical +Z\nNo simulated fly behavior")
        for ids, scatter in self.eye_artists:
            scatter.set_array(trial.arrays["light"][sensor_index, ids])
        self.motion.set_data(trial.arrays["channels"][neural_index].reshape(2, 8))
        means = trial.stage_means(neural_index)
        for side, line in enumerate(self.stage_markers): line.set_xdata(means[:, side])
        self.dn_cursor.set_xdata([time_ms/1000]*2)
        for index, line in self.dn_lines:
            line.set_data(trial.arrays["time_ms"][:neural_index+1]/1000,
                          trial.arrays["central"][:neural_index+1, index])
        for index, label in self.dn_labels:
            row = trial.central_rows[index]
            value = trial.arrays["central"][neural_index, index]
            label.set_text(f"{row['side']} {row['body_id']}   {value:+.2e}")
        # Check actual artist data on EVERY rendered frame, not only input hashes.
        for ids, scatter in self.eye_artists:
            if not np.array_equal(scatter.get_array(), trial.arrays["light"][sensor_index, ids]):
                raise ValueError("displayed facet values differ from saved input")
        if not np.array_equal(self.motion.get_array(), trial.arrays["channels"][neural_index].reshape(2, 8)):
            raise ValueError("displayed motion values differ from saved trace")
        for index, line in self.dn_lines:
            if not np.array_equal(line.get_ydata(), trial.arrays["central"][:neural_index+1, index]):
                raise ValueError("displayed DN trace differs from saved trace")
        for side, line in enumerate(self.stage_markers):
            if not np.array_equal(line.get_xdata(), means[:, side]):
                raise ValueError("displayed stage means differ from saved states")
        self.fig.canvas.draw()
        return Image.fromarray(np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3].copy())

    def close(self):
        plt.close(self.fig)


def save_animation(frames, poster, path, interval_ms):
    """Lossless sparse OVER updates; decoded pixels equal every full RGB frame."""
    updates, previous = [], None
    for frame in frames:
        pixels = np.asarray(frame.convert("RGB"))
        changed = np.ones(pixels.shape[:2], bool) if previous is None else np.any(pixels != previous, axis=2)
        rgba = np.zeros((*pixels.shape[:2], 4), dtype=np.uint8)
        rgba[changed, :3] = pixels[changed]
        rgba[changed, 3] = 255
        updates.append(Image.fromarray(rgba))
        previous = pixels
    poster.convert("RGBA").save(path, format="PNG", save_all=True, append_images=updates,
                default_image=True, duration=interval_ms, loop=0,
                disposal=0, blend=1, compress_level=9)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT/"results/exp008_eye_final")
    parser.add_argument("--output", type=Path, default=ROOT/"results/exp008_observer")
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to((ROOT/"results").resolve()):
        raise ValueError("bulk visualization output must stay under ignored results/")
    trial = load_saved(args.bundle)
    print("Frozen trace digest, measured axes, readout endpoints, neuron IDs and timing verified.", flush=True)
    observer = Observer(trial)
    times = np.arange(0., trial.spec["early_vision"]["parameters"]["duration_ms"], float(FRAME_INTERVAL_MS))
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        poster = observer.draw(3200.)
        frames = []
        for i, time in enumerate(times):
            frames.append(observer.draw(float(time)))
            if i % 20 == 0: print(f"Rendering frame {i}/{len(times)}", flush=True)
        animation, overview = (args.output/name for name in ("EXP-008-observer.png", "EXP-008-observer-overview.png"))
        save_animation(frames, poster, animation, FRAME_INTERVAL_MS)
        poster.save(overview, compress_level=9)
        manifest = {"purpose": "post-hoc observer only; no model execution or scientific record writes",
                    "condition": trial.condition, "frozen_trace_digest": trial.record["deterministic_trace_digests"][trial.condition],
                    "specification_sha256": trial.record["specification_sha256"],
                    "render_source_sha256": evidence_sha256(Path(__file__)),
                    "input_sha256": {name: sha256_file(args.bundle/name) for name in (f"{trial.condition}.npz", "sensor_axes.npz")},
                    "matplotlib": matplotlib.__version__, "pillow": PILLOW_VERSION, "numpy": np.__version__,
                    "display_frame_times_ms": times.tolist(), "poster_time_ms": 3200., "interval_ms": FRAME_INTERVAL_MS,
                    "every_frame_artist_values_checked": True,
                    "interpolation": "none: last saved sensor/neural sample at or before frame time; plot lines connect exact samples",
                    "sample_indices": [trial.indices(float(t)) for t in times],
                    "media_sha256": {p.name: sha256_file(p) for p in (animation, overview)},
                    "media_bytes": {p.name: p.stat().st_size for p in (animation, overview)}}
        (args.output/"render_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf8")
        if args.promote:
            if any(p.stat().st_size > MAX_MEDIA_BYTES for p in (animation, overview)):
                raise ValueError("media exceeds repository size policy; retain under ignored results/")
            for path in (animation, overview): shutil.copyfile(path, ROOT/"figures"/path.name)
        print(json.dumps(manifest["media_bytes"]), flush=True)
    finally:
        observer.close()


if __name__ == "__main__":
    main()
