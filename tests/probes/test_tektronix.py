"""Tests for Tektronix CSV loading helpers."""

from __future__ import annotations

import numpy as np
import xarray as xr

from golem_toolkit.loading import load_tek64_csv
from golem_toolkit.probes.tektronix import load_probe_waveforms, select_first_cycles


def test_select_first_cycles():
    f0 = 10_000.0
    n = 1000
    dt = 1e-6
    t = np.arange(n) * dt
    ds = xr.Dataset({"CH1": ("t", np.sin(2 * np.pi * f0 * t))}, coords={"t": t})
    out = select_first_cycles(ds, f0, n_cycles=2.0)
    duration = float(out["t"].values[-1] - out["t"].values[0])
    assert abs(duration - 2.0 / f0) < 2 * dt


def test_load_tek64_csv_real_export(tmp_path):
    """Round-trip against MSO64 ALL layout (skiprows=12, TEK64 default)."""
    from golem_toolkit.loading import load_tek64_csv

    f0 = 5_000.0
    n = 500
    dt = 1e-6
    t = np.arange(n) * dt
    lines = ["x,0\n"] * 11
    lines[4] = f"x,{dt}\n"
    lines[5] = f"x,{n}\n"
    rows = ["TIME,CH1,CH2,CH3,CH4\n"]
    for ti, v1, v2 in zip(t, np.sin(2 * np.pi * f0 * t), np.cos(2 * np.pi * f0 * t)):
        rows.append(f"{ti},{v1},{v2},0.0,0.0\n")
    csv_path = tmp_path / "tek.csv"
    csv_path.write_text("".join(lines + rows), encoding="utf-8")

    ds = load_tek64_csv(csv_path, channels=["CH1", "CH2"])
    assert "t" in ds.coords
    assert ds.sizes["t"] == n
    assert "CH1" in ds


def test_load_probe_waveforms_synthetic_via_monkeypatch(tmp_path):
    """Round-trip shape check using a minimal fake CSV is skipped; test phasor path only."""
    f0 = 5_000.0
    n = 2000
    dt = 1e-6
    t = np.arange(n) * dt
    ch1 = 0.5 * np.sin(2 * np.pi * f0 * t)
    ch2 = 2.0 * np.sin(2 * np.pi * f0 * t)

    # Build minimal Tektronix-style CSV (11 metadata lines, then column header)
    lines = ["x,0\n"] * 11
    lines[4] = f"x,{dt}\n"
    lines[5] = f"x,{n}\n"
    rows = ["TIME,CH1,CH2,CH3,CH4\n"]
    for ti, v1, v2 in zip(t, ch1, ch2):
        rows.append(f"{ti},{v1},{v2},0.0,0.0\n")
    csv_path = tmp_path / "fake_tek.csv"
    csv_path.write_text("".join(lines + rows), encoding="utf-8")

    i_raw, u_raw = load_probe_waveforms(
        csv_path, f0_hz=f0, shunt_resistance_ohm=10.0, n_cycles=1.0
    )
    assert i_raw.sizes["t"] == u_raw.sizes["t"]
    assert i_raw.attrs["units"] == "A"
    assert u_raw.attrs["units"] == "V"
    assert i_raw.attrs["scope_channel"] == "CH1"
    assert u_raw.attrs["scope_channel"] == "CH2"

    i_swapped, u_swapped = load_probe_waveforms(
        csv_path,
        f0_hz=f0,
        shunt_resistance_ohm=10.0,
        i_channel="CH2",
        u_channel="CH1",
        n_cycles=1.0,
    )
    assert i_swapped.attrs["scope_channel"] == "CH2"
    assert u_swapped.attrs["scope_channel"] == "CH1"

    ds = load_tek64_csv(csv_path, channels=["CH1", "CH2"])
    i_from_ds, u_from_ds = load_probe_waveforms(
        ds, f0_hz=f0, shunt_resistance_ohm=10.0, n_cycles=1.0
    )
    assert i_from_ds.sizes == i_raw.sizes
    np.testing.assert_allclose(i_from_ds.values, i_raw.values)
    np.testing.assert_allclose(u_from_ds.values, u_raw.values)
