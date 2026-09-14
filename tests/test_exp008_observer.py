"""Post-hoc rendering contracts; synthetic fixtures never enter EXP-008."""

from dataclasses import replace
from itertools import combinations
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import pytest

from scripts.render_exp008_observer import (Observer, SavedTrial, facet_projection,
                                            load_saved, save_animation)
from src.compound_eye import EyeGeometry
from src.sensory_chain import CHANNELS

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/"results/exp008_eye_final"


def small_trial():
    u = np.array([[1., 1., 0.], [0., 1., 1.], [1., -1., 0.], [0., -1., 1.]])/np.sqrt(2)
    geometry = EyeGeometry(np.zeros((4, 3)), u, np.array([True, True, False, False]),
                           np.full((4, 6), -1, dtype=int), np.zeros((4, 2), dtype=int))
    rows = tuple({"body_id": 100+i, "stage": stage, "side": side, "type": stage}
                 for i, (stage, side) in enumerate((stage, side) for stage in
                    ("HS", "H2", "H2rn", "uLPTCrn", "bIPS", "DNp15") for side in ("L", "R")))
    arrays = {"sensor_time_ms": np.array([0., 2000., 4000.]), "time_ms": np.arange(6)*1000.,
              "light": np.array([[.1, .2, .3, .4], [.5, .6, .7, .8], [.2, .4, .6, .8]]),
              "channels": np.arange(96).reshape(6, 16)*1e-5,
              "central": np.arange(72).reshape(6, 12)*1e-5,
              "LPi": np.arange(12).reshape(6, 2)*1e-5,
              "HS_H2_drive": np.arange(24).reshape(6, 4)*1e-5,
              "HS_H2_body_ids": np.array([100, 101, 102, 103])}
    axes = {"left_readout_centers": np.array([0]), "right_readout_centers": np.array([2]),
            "left_readout_endpoints": np.array([[[1, 1], [0, 0], [1, 1], [0, 0]]]),
            "right_readout_endpoints": np.array([[[3, 3], [2, 2], [3, 3], [2, 2]]])}
    spec = {"early_vision": {"parameters": {"onset_ms": 2000., "offset_ms": 4000., "duration_ms": 6000.}},
            "pose_conditions": {"yaw_positive_Z": [45., 0.]}, "scene": {}}
    LPi = ({"bodyId": 1, "somaSide": "L"}, {"bodyId": 2, "somaSide": "R"})
    for a in (*arrays.values(), *axes.values(), *geometry.__dict__.values()): a.setflags(write=False)
    return SavedTrial(spec, {}, "yaw_positive_Z", geometry, axes, arrays, rows, LPi)


@pytest.fixture(scope="module")
def recorded():
    if not (BUNDLE/"yaw_positive_Z.npz").exists():
        pytest.skip("ignored EXP-008 saved bundle not installed")
    return load_saved(BUNDLE)


def assert_display_matches(observer, trial, time):
    s, n = trial.indices(time)
    for ids, scatter in observer.eye_artists:
        np.testing.assert_array_equal(scatter.get_array(), trial.arrays["light"][s, ids])
        assert scatter.norm.vmin == 0 and scatter.norm.vmax == 1
    np.testing.assert_array_equal(observer.motion.get_array(), trial.arrays["channels"][n].reshape(2, 8))
    for i, line in observer.dn_lines:
        np.testing.assert_array_equal(line.get_ydata(), trial.arrays["central"][:n+1, i])
        np.testing.assert_array_equal(line.get_xdata(), trial.arrays["time_ms"][:n+1]/1000)
    for side, line in enumerate(observer.stage_markers):
        np.testing.assert_array_equal(line.get_xdata(), trial.stage_means(n)[:, side])


def assert_layout_clear(observer):
    renderer = observer.fig.canvas.get_renderer()
    texts = [t for t in observer.fig.texts if t.get_text()]
    for a, b in combinations(texts, 2):
        assert not a.get_window_extent(renderer).overlaps(b.get_window_extent(renderer)), (a.get_text(), b.get_text())
    footer = next(t for t in texts if t.get_text().startswith("Measured geometry"))
    assert not observer.dn_ax.xaxis.label.get_window_extent(renderer).overlaps(footer.get_window_extent(renderer))
    assert observer.stage_ax.xaxis.get_offset_text().get_text() == ""
    for tick in observer.stage_ax.get_xticklabels():
        assert not observer.stage_ax.xaxis.label.get_window_extent(renderer).overlaps(tick.get_window_extent(renderer))


def test_sample_selection_holds_saved_values_without_future_interpolation():
    trial = small_trial()
    assert trial.indices(0) == (0, 0)
    assert trial.indices(1999.9) == (0, 1)
    assert trial.indices(2000) == (1, 2)
    assert trial.indices(5999.5) == (2, 5)
    for bad in (-1, 6000, np.nan):
        with pytest.raises(ValueError): trial.indices(bad)


def test_pose_is_the_frozen_prescription_not_a_neural_or_body_response():
    trial = small_trial()
    for time, angle in [(0., 0.), (2000., 0.), (3000., np.pi/4), (5000., np.pi/2)]:
        r, t = trial.pose(time)
        np.testing.assert_allclose(r@[1., 0., 0.], [np.cos(angle), np.sin(angle), 0.], atol=1e-15)
        np.testing.assert_array_equal(t, 0)


def test_measured_direction_projection_has_independent_angular_reference():
    for left, sign in [(True, 1), (False, -1)]:
        directions = np.array([[0., sign, 0.], [1., 0., 0.], [0., 0., 1.]])
        np.testing.assert_allclose(facet_projection(directions, left),
                                   [[0., 0.], [np.sqrt(2), 0.], [0., np.sqrt(2)]])
    with pytest.raises(ValueError): facet_projection(np.array([[0., -1., 0.]]), True)


def test_stage_means_preserve_sign_and_use_identified_sides():
    trial = small_trial()
    means = trial.stage_means(3)
    np.testing.assert_array_equal(means[0], trial.arrays["LPi"][3])
    np.testing.assert_allclose(means[1], [trial.arrays["HS_H2_drive"][3, [0, 2]].mean(),
                                         trial.arrays["HS_H2_drive"][3, [1, 3]].mean()])
    np.testing.assert_array_equal(means[2:], trial.arrays["central"][3, :10].reshape(5, 2))
    arrays = {**trial.arrays, "central": -trial.arrays["central"]}
    np.testing.assert_array_equal(replace(trial, arrays=arrays).stage_means(3)[2:], -means[2:])


def test_observer_never_executes_sensor_or_neural_simulation(monkeypatch):
    from src.compound_eye import EyeSampler, LocalReadout
    from src.sensory_chain import VisualProjection
    from src.binocular_physiology import BinocularNetwork
    def forbidden(*args, **kwargs): raise AssertionError("model executed inside observer")
    for cls, method in [(EyeSampler, "sample"), (LocalReadout, "channels"),
                        (VisualProjection, "simulate"), (BinocularNetwork, "simulate")]:
        monkeypatch.setattr(cls, method, forbidden)
    trial = small_trial()
    before = {k: a.copy() for k, a in trial.arrays.items()}
    observer = Observer(trial)
    try:
        for time in (0., 3200., 5950.):
            observer.draw(time)
            assert_display_matches(observer, trial, time)
        for k, a in before.items(): np.testing.assert_array_equal(trial.arrays[k], a)
    finally:
        observer.close()


def test_observer_frame_repeat_is_pixel_deterministic():
    observer = Observer(small_trial())
    try:
        a = np.asarray(observer.draw(3200.)).copy()
        observer.draw(0.)
        np.testing.assert_array_equal(a, np.asarray(observer.draw(3200.)))
    finally:
        observer.close()


def test_text_layout_has_no_caption_time_label_or_unit_collisions():
    observer = Observer(small_trial())
    try:
        for time in (0., 3200., 5940.):
            observer.draw(time)
            assert_layout_clear(observer)
    finally:
        observer.close()


def test_apng_is_lossless_deterministic_and_preserves_timing(tmp_path):
    rng = np.random.default_rng(7)
    base = rng.integers(0, 256, size=(16, 24, 3), dtype=np.uint8)
    pixels = [base.copy() for _ in range(3)]
    pixels[1][2:6, 4:9] = 0
    pixels[2][10:15, 20:22] = 255  # also restore the previous black patch
    frames = [Image.fromarray(p) for p in pixels]
    a, b = tmp_path/"a.png", tmp_path/"b.png"
    save_animation(frames, frames[1], a, 50)
    save_animation(frames, frames[1], b, 50)
    assert a.read_bytes() == b.read_bytes()
    with Image.open(a) as image:
        assert image.default_image and image.n_frames == 4 and image.info["loop"] == 0
        np.testing.assert_array_equal(np.asarray(image.convert("RGB")), np.asarray(frames[1]))
        for i, frame in enumerate(frames, 1):
            image.seek(i)
            assert image.info["duration"] == 50
            np.testing.assert_array_equal(np.asarray(image.convert("RGB")), np.asarray(frame))


def test_recorded_maps_and_neural_artists_match_exact_samples(recorded):
    observer = Observer(recorded)
    try:
        for time in (0., 2050., 3200., 5950.):
            observer.draw(time)
            assert_display_matches(observer, recorded, time)
            assert_layout_clear(observer)
        assert [len(ids) for ids, _ in observer.eye_artists] == [857, 852]
    finally:
        observer.close()


@pytest.mark.parametrize("field", ["directions", "central_body_ids", "sensor_time_ms"])
def test_modified_bundle_metadata_is_rejected(recorded, tmp_path, field):
    if field == "directions":
        shutil.copyfile(BUNDLE/"yaw_positive_Z.npz", tmp_path/"yaw_positive_Z.npz")
        axes = {**recorded.axes, "directions": recorded.axes["directions"][::-1]}
        np.savez_compressed(tmp_path/"sensor_axes.npz", **axes)
    else:
        shutil.copyfile(BUNDLE/"sensor_axes.npz", tmp_path/"sensor_axes.npz")
        arrays = {**recorded.arrays, field: recorded.arrays[field][::-1]}
        np.savez_compressed(tmp_path/"yaw_positive_Z.npz", **arrays)
    with pytest.raises(ValueError): load_saved(tmp_path)


def test_actual_readout_masks_preserve_peripheral_display_only_status(recorded):
    for side, left, count, used_count in [("left", True, 65, 87), ("right", False, 66, 88)]:
        centers = recorded.axes[f"{side}_readout_centers"]
        endpoints = np.unique(recorded.axes[f"{side}_readout_endpoints"])
        assert len(centers) == count and len(endpoints) == used_count
        assert np.all(recorded.geometry.left[endpoints] == left)
        assert len(endpoints) < np.count_nonzero(recorded.geometry.left == left)/8
