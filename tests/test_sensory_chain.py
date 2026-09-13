from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.malecns_io import ConnectomeGraph
from src.sensory_chain import CHANNELS, SensoryParameters, VisualProjection, central_drive, luminance_movie, motion_channels

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def parameters():
    return replace(SensoryParameters(), onset_ms=50, offset_ms=250, duration_ms=400)


def test_static_flicker_and_causality(parameters):
    p = parameters
    time, image = luminance_movie(p, [0, 0])
    assert not motion_channels(image, p).any()
    flicker = np.broadcast_to((.5+.3*np.sin(time/20))[:, None, None, None], image.shape)
    assert not motion_channels(flicker, p).any()
    _, moving = luminance_movie(p, [1, -1])
    rates = motion_channels(moving, p)
    assert not rates[time <= p.onset_ms].any()
    changed = moving.copy()
    changed[500:] = .1
    np.testing.assert_array_equal(rates[:501], motion_channels(changed, p)[:501])
    np.testing.assert_array_equal(rates, motion_channels(moving, p))


def test_direction_polarity_eye_symmetries(parameters):
    p = parameters
    time, movie = luminance_movie(p, [1, -1])
    rates = motion_channels(movie, p).reshape(-1, 2, 2, 4)
    swapped = motion_channels(movie[:, ::-1], p).reshape(rates.shape)
    inverted = motion_channels(1-movie, p).reshape(rates.shape)
    np.testing.assert_allclose(swapped, rates[:, ::-1], atol=1e-15)
    np.testing.assert_allclose(inverted, rates[:, :, ::-1], atol=1e-15)
    reverse = motion_channels(luminance_movie(p, [-1, 1])[1], p).reshape(rates.shape)
    np.testing.assert_allclose(reverse[:, :, :, :2], rates[:, :, :, [1, 0]], atol=2e-15)
    vertical = motion_channels(luminance_movie(p, [1, -1], axis="y")[1], p).reshape(rates.shape)
    np.testing.assert_allclose(vertical[:, :, :, 2:], rates[:, :, :, :2], atol=2e-15)
    assert not rates[:, :, :, 2:].any()
    late = rates[(time >= 200) & (time < 250)].mean(axis=0)
    assert np.all(late[0, :, 0] > 0) and np.all(late[0, :, 1] == 0)
    assert np.all(late[1, :, 1] > 0) and np.all(late[1, :, 0] == 0)
    half = motion_channels(luminance_movie(replace(p, contrast=p.contrast/2), [1, -1])[1], p)
    np.testing.assert_allclose(half, rates.reshape(-1, 16)/4, atol=1e-15)


def test_emission_filter_independent_scalar_reference(parameters):
    p = parameters
    _, images = luminance_movie(p, [1, 0])
    actual = motion_channels(images, p)
    # Separate scalar spatial loops validate vector layout, delay and signs.
    adaptation = images[0, 0, 0].copy()
    delay_on = np.zeros(p.side_pixels)
    delay_off = np.zeros(p.side_pixels)
    state = np.zeros(4)
    reference = []
    for image in images[:, 0, 0]:
        reference.append(state.copy())
        hp = image-adaptation
        current = [np.maximum(hp, 0), np.maximum(-hp, 0)]
        delayed = [delay_on, delay_off]
        target = []
        for c, d in zip(current, delayed):
            value = sum(d[j-1]*c[j]-c[j-1]*d[j] for j in range(p.side_pixels))/p.side_pixels
            target.extend([max(value, 0), max(-value, 0)])
        state += p.dt_ms/p.emission_tau_ms*(np.array(target)-state)
        adaptation += p.dt_ms/p.highpass_tau_ms*(image-adaptation)
        delay_on += p.dt_ms/p.delay_tau_ms*(current[0]-delay_on)
        delay_off += p.dt_ms/p.delay_tau_ms*(current[1]-delay_off)
    np.testing.assert_allclose(actual[:, [0, 1, 4, 5]], reference, atol=2e-15)


def synthetic_graph():
    rows = [(i+1, kind, side, "T4" if kind.startswith("T4") else "T5", "acetylcholine")
            for i, (side, kind) in enumerate(c.split("_") for c in CHANNELS)]
    rows += [(100+i, "LPi12" if i%2 else "LPi21", "L" if i < 3 else "R", "LPi", "gaba") for i in range(6)]
    rows += [(200+i, "H2" if i in [3, 7] else "HSE", "L" if i < 4 else "R", "H2" if i in [3, 7] else "HS", "acetylcholine") for i in range(8)]
    nodes = pd.DataFrame(rows, columns=["bodyId", "type", "somaSide", "stage", "consensus_nt"])
    edges = pd.DataFrame([(a, b, float(1+(a+b)%7)) for a in list(range(1, 17))+list(range(100, 106))
                          for b in list(range(100, 106))+list(range(200, 208)) if a != b],
                         columns=["pre_body", "post_body", "synapse_count"])
    return ConnectomeGraph(nodes, edges)


def test_projection_matches_raw_counts_and_independent_dense_update(parameters):
    graph = synthetic_graph()
    projection = VisualProjection.compile(graph, parameters)
    by_id = graph.nodes.set_index("bodyId")
    target_ids = list(projection.target_ids)
    sources = list(projection.motion_ids)+list(projection.LPi_ids)
    dense = np.zeros((len(target_ids), len(sources)))
    for row in graph.edges.itertuples():
        dense[target_ids.index(row.post_body), sources.index(row.pre_body)] = row.synapse_count * (-1 if by_id.loc[row.pre_body, "stage"] == "LPi" else 1)
    dense *= parameters.chemical_gain/np.abs(dense).sum(axis=1)[:, None]
    rng = np.random.default_rng(35)
    rates = rng.random((60, 16))
    bodies = rates[:, projection.motion_channel_indices]
    np.testing.assert_allclose(projection.motion_to_targets @ rates.T, dense[:, :len(bodies[0])] @ bodies.T, atol=2e-16)
    initial = rng.normal(size=6)
    for disabled in [False, True]:
        traces, drive = projection.simulate(rates, no_LPi_output=disabled, initial_LPi=initial)
        state = initial.copy()
        recurrence = dense[:, 16:] if not disabled else np.zeros((14, 6))
        for k in range(len(rates)):
            np.testing.assert_allclose(traces[k], state, atol=2e-15)
            total = dense[:, :16] @ bodies[k] + recurrence @ np.maximum(state, 0)
            np.testing.assert_allclose(drive[k], total[6:], atol=2e-15)
            state += parameters.dt_ms/parameters.LPi_tau_ms*(-state+total[:6])
        assert projection.preflight(no_LPi_output=disabled)["passed"]
    shuffled = ConnectomeGraph(graph.nodes.sample(frac=1, random_state=3), graph.edges.sample(frac=1, random_state=2))
    other = VisualProjection.compile(shuffled, parameters)
    np.testing.assert_array_equal(other.motion_to_targets.toarray(), projection.motion_to_targets.toarray())


def test_drive_axis_and_rejections(parameters):
    values = np.arange(24).reshape(3, 8)
    result = central_drive(range(8), values, range(12, -1, -1))
    np.testing.assert_array_equal(result[:, -1:-9:-1], values)
    assert not result[:, :5].any()
    with pytest.raises(ValueError):
        central_drive([1, 1], np.zeros((3, 2)), [1, 2])
    for changes in [{"dt_ms": 100}, {"contrast": 1}, {"side_pixels": 1}, {"chemical_gain": 1}, {"speed_degrees_s": np.nan}]:
        with pytest.raises(ValueError):
            replace(parameters, **changes)
    with pytest.raises(ValueError):
        motion_channels(np.ones((5, 2, 16, 16))*1.1, parameters)


def test_resolved_visual_graph_compression_and_frozen_central():
    folder = ROOT / "data/malecns_exp005_sensory"
    if not (folder / "nodes.parquet").exists():
        pytest.skip("ignored scientific materialization not installed")
    graph = ConnectomeGraph(pd.read_parquet(folder / "nodes.parquet"), pd.read_parquet(folder / "edges.parquet"))
    p = SensoryParameters()
    visual = VisualProjection.compile(graph, p)
    assert len(visual.motion_ids) == 7556 and len(graph.edges) == 23873
    assert visual.preflight()["passed"] and visual.preflight(no_LPi_output=True)["passed"]
    # Raw edge accumulation is independent of sparse compiler and compression.
    rates = np.random.default_rng(52).random(16)
    LPi = np.arange(6)/6
    source = dict(zip(visual.motion_ids, rates[visual.motion_channel_indices]))
    source.update(dict(zip(visual.LPi_ids, -LPi)))
    denominator = graph.edges.groupby("post_body").synapse_count.sum()
    expected = graph.edges.assign(value=graph.edges.pre_body.map(source)*graph.edges.synapse_count).groupby("post_body").value.sum()
    expected = expected.reindex(visual.target_ids)/denominator.reindex(visual.target_ids)*p.chemical_gain
    np.testing.assert_allclose(visual.motion_to_targets @ rates+visual.LPi_to_targets @ LPi, expected, atol=2e-15)
    from scripts.run_exp004_physiology import load_frozen
    from src.binocular_physiology import BinocularNetwork, PhysiologyParameters
    from src.provenance import sha256_file
    spec, central = load_frozen()
    successor = json.loads((ROOT / "experiments/EXP-005-sensory-to-DNp15/specification.json").read_text())
    assert successor["central_specification"]["sha256"] == sha256_file(ROOT / successor["central_specification"]["path"])
    assert successor["central_parameters"] == spec["parameters"]
    assert successor["observation_contract"] == spec["observation_contract"]
    network = BinocularNetwork.compile(central, spec["electrical_pairs"], PhysiologyParameters(**spec["parameters"]))
    assert set(visual.input_ids) == set(network.nodes[network.nodes.stage.isin(["HS", "H2"])].bodyId)
    drive = central_drive(visual.input_ids, np.zeros((20, 8)), network.nodes.bodyId)
    assert not network.simulate(drive).any()


def test_actual_cascade_causality_disconnection_and_coupled_decay(parameters):
    folder = ROOT / "data/malecns_exp005_sensory"
    if not (folder / "nodes.parquet").exists():
        pytest.skip("ignored scientific materialization not installed")
    from scripts.run_exp004_physiology import load_frozen
    from src.binocular_physiology import BinocularNetwork, PhysiologyParameters
    central_spec, graph = load_frozen()
    network = BinocularNetwork.compile(graph, central_spec["electrical_pairs"], PhysiologyParameters(**central_spec["parameters"]))
    visual_graph = ConnectomeGraph(pd.read_parquet(folder / "nodes.parquet"), pd.read_parquet(folder / "edges.parquet"))
    visual = VisualProjection.compile(visual_graph, parameters)
    time, images = luminance_movie(parameters, [1, -1])
    channels = motion_channels(images, parameters)
    LPi, projection = visual.simulate(channels)
    drive = central_drive(visual.input_ids, projection, network.nodes.bodyId)
    trace = network.simulate(drive)
    assert not trace[time <= parameters.onset_ms].any()
    assert np.max(abs(LPi)) > 0
    assert np.max(abs(trace[:, network.nodes.stage.eq("DNp15")])) > 1e-10
    assert not network.simulate(np.zeros_like(drive)).any()
    changed = drive.copy()
    changed[500:] += 2
    np.testing.assert_array_equal(trace[:501], network.simulate(changed)[:501])
    # Test decay of the connected cascade, not just isolated recurrent blocks.
    zero = np.zeros((round(2000/parameters.dt_ms)+1, 16))
    LPi_decay, input_decay = visual.simulate(zero, initial_LPi=np.ones(6))
    central_decay = network.simulate(central_drive(visual.input_ids, input_decay, network.nodes.bodyId),
                                     initial_state=np.linspace(-1, 1, len(network.nodes)))
    assert np.max(abs(LPi_decay[-1])) < 1e-8
    assert np.max(abs(central_decay[-1])) < 1e-8
