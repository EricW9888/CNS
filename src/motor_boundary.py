"""CvNA2 raster measurements and an algebraic calibration identifiability check.

No neural-to-body decoder or fitted movement model is defined here.
"""

from __future__ import annotations

import numpy as np


def trace_panel(rgb, panel, contract, threshold):
    """Trace only a unique contiguous dark run; ambiguous columns fail closed."""
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Expected an RGB uint8 native plot")
    if not 0 <= threshold <= 255:
        raise ValueError("Invalid dark threshold")
    x0, x1 = panel["x_pixels"]
    y0, y1 = panel["y_pixels"]
    t0, t1 = panel["time_ms"]
    v0, v1 = panel["velocity_deg_s"]
    first, last = panel["trace_columns"]
    top, bottom = contract["trace_y_rows"]
    if not (0 <= x0 < first <= last < x1 < image.shape[1]
            and 0 <= y0 < top <= bottom < y1 < image.shape[0]
            and t1 > t0 and v0 > v1):
        raise ValueError("Invalid calibration / trace bounds")
    columns = np.arange(first, last + 1)
    center, half_width = [], []
    for column in columns:
        rows = np.flatnonzero(image[top:bottom+1, column].max(axis=1) <= threshold) + top
        runs = np.split(rows, np.flatnonzero(np.diff(rows) != 1) + 1)
        runs = [run for run in runs if len(run) >= contract["minimum_run_thickness_pixels"]]
        if len(runs) != 1:
            raise ValueError(f"Missing or ambiguous mean trace at column {column}")
        rows = runs[0]
        if len(rows) > contract["maximum_run_thickness_pixels"]:
            raise ValueError(f"Trace too thick at column {column}")
        center.append((rows[0] + rows[-1]) / 2)
        half_width.append((rows[-1] - rows[0]) / 2)
    times = t0 + (columns - x0) * (t1 - t0) / (x1 - x0)
    scale = (v0 - v1) / (y1 - y0)
    values = v0 - (np.asarray(center) - y0) * scale
    # Half line width plus one pixel of y calibration uncertainty.
    errors = (np.asarray(half_width) + 1) * scale
    return times, values, errors


def extract_target(rgb, contract):
    start, stop = contract["sample_time_ms"]
    step = contract["sample_step_ms"]
    if step <= 0 or stop <= start or (stop - start) % step:
        raise ValueError("Invalid source-frame sampling")
    times = np.arange(start, stop + step / 2, step, dtype=float)
    output = {"time_ms": times.tolist(), "components": {}}
    for name, panel in contract["panels"].items():
        thresholds = [contract["dark_threshold"], *contract["threshold_sensitivity"]]
        variants = [trace_panel(rgb, panel, contract, threshold) for threshold in thresholds]
        native_t, native_v, _ = variants[0]
        pixel_ms = (panel["time_ms"][1] - panel["time_ms"][0]) / (panel["x_pixels"][1] - panel["x_pixels"][0])
        if times[0] - pixel_ms < native_t[0] or times[-1] + pixel_ms > native_t[-1]:
            raise ValueError("Sample + x-calibration envelope outside native trace")
        values = np.interp(times, native_t, native_v)
        lows, highs = [], []
        for nt, nv, error in variants:
            for jitter in [-pixel_ms, 0, pixel_ms]:
                lows.append(np.interp(times + jitter, nt, nv - error))
                highs.append(np.interp(times + jitter, nt, nv + error))
        low = np.min(lows, axis=0)
        high = np.max(highs, axis=0)
        baseline = window_mask(times, contract["baseline_window_ms_inclusive"])
        active = window_mask(times, contract["response_window_ms_inclusive"])
        base = float(values[baseline].mean())
        delta = values - base
        delta_low = low - high[baseline].mean()
        delta_high = high - low[baseline].mean()
        active_t = times[active]
        def integral(v):
            return float(np.trapezoid(v[active], active_t / 1000))
        extreme = int(np.flatnonzero(active)[np.argmax(np.abs(delta[active]))])
        # Compatible extremum samples, not a confidence interval: line envelopes
        # may leave several (possibly non-contiguous) peak locations unresolved.
        minimum_abs = np.where((delta_low <= 0) & (delta_high >= 0), 0,
                               np.minimum(np.abs(delta_low), np.abs(delta_high)))
        possible = active & (np.maximum(np.abs(delta_low), np.abs(delta_high)) >= np.max(minimum_abs[active]))
        summary = {
            "baseline_mean_deg_s": base,
            "absolute_mean_deg_s": float(values[active].mean()),
            "delta_mean_deg_s": float(delta[active].mean()),
            "delta_mean_extraction_envelope_deg_s": [float(delta_low[active].mean()), float(delta_high[active].mean())],
            "absolute_component_integral_degrees": integral(values),
            "delta_component_integral_degrees": integral(delta),
            "delta_component_integral_extraction_envelope_degrees": [integral(delta_low), integral(delta_high)],
            "largest_abs_delta_absolute_velocity_deg_s": float(values[extreme]),
            "largest_abs_delta_velocity_deg_s": float(delta[extreme]),
            "largest_abs_delta_time_ms": float(times[extreme]),
            "extremum_time_resolution_plus_axis_error_ms": float(step + pixel_ms),
            "extraction_compatible_extremum_sample_times_ms": times[possible].tolist(),
            "max_trace_envelope_half_range_deg_s": float(np.max((high - low) / 2)),
            "threshold_center_max_difference_deg_s": float(max(np.max(np.abs(np.interp(times, nt, nv) - values)) for nt, nv, _ in variants)),
        }
        output["components"][name] = {
            "absolute_velocity_deg_s": values.tolist(), "absolute_low_deg_s": low.tolist(),
            "absolute_high_deg_s": high.tolist(), "delta_velocity_deg_s": delta.tolist(),
            "delta_low_deg_s": delta_low.tolist(), "delta_high_deg_s": delta_high.tolist(),
            "summary": summary,
        }
    return output


def window_mask(times, window):
    values = np.asarray(times, dtype=float)
    if (values.ndim != 1 or not len(values) or not np.isfinite(values).all()
            or np.any(np.diff(values) <= 0) or len(window) != 2
            or window[0] > window[1] or window[0] < values[0] or window[1] > values[-1]):
        raise ValueError("Invalid time axis / measurement window")
    mask = (values >= window[0]) & (values <= window[1])
    if mask.sum() < 2:
        raise ValueError("Measurement window requires at least two samples")
    return mask


def indistinguishable_led_family(led_effect, dn_drive, coefficients):
    """Even a known LED movement kernel cannot identify the independent DN gain.

    Returns arbitrary-unit algebraic examples, NOT physiological predictions.
    """
    led = np.asarray(led_effect, dtype=float)
    dn = np.asarray(dn_drive, dtype=float)
    gains = np.asarray(coefficients, dtype=float)
    if (led.ndim != 1 or dn.ndim != 1 or not len(led) or not len(dn)
            or gains.ndim != 1 or not len(gains) or np.any(gains < 0)
            or not all(np.isfinite(a).all() for a in [led, dn, gains])):
        raise ValueError("Invalid illustrative identifiability inputs")
    return np.broadcast_to(led, (len(gains), len(led))).copy(), gains[:, None] * dn
