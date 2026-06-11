"""Tests for probe IV calibration and processing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from golem_toolkit.probes import IVCalibration, load_iv_calibration, process_iv_signals

FIXTURE_NC = Path(__file__).parent / "fixtures" / "active_probe_calib.nc"


@pytest.fixture(scope="module")
def calibration():
    return IVCalibration.from_netcdf(FIXTURE_NC)


def test_load_iv_calibration():
    ds = load_iv_calibration(FIXTURE_NC)
    assert "open_fixture_effective_capacitance" in ds
    assert "dc_leak_conductance_s" in ds.attrs


def test_parasitics_at_frequency(calibration):
    paras = calibration.parasitics_at_frequency(20_000.0)
    assert np.isfinite(paras["open_fixture_leak_conductance"])
    assert np.isfinite(paras["open_fixture_effective_capacitance"])
    assert abs(paras["open_fixture_effective_capacitance"]) > 0


def test_ac_sine_processing(calibration):
    f0 = 20_000.0
    n = 4000
    dt = 1e-6
    t = np.arange(n) * dt
    cal = calibration.at_frequency(f0)
    u_tip_phasor = 10.0 + 0.0j
    u_div_phasor = u_tip_phasor * cal.voltage_divider_transfer
    i_phasor = 0.05 + 0.0j

    U_raw = xr.DataArray(
        np.real(u_div_phasor * np.exp(1j * 2 * np.pi * f0 * t)), coords={"t": t}, dims=["t"]
    )
    I_raw = xr.DataArray(np.real(i_phasor * np.exp(1j * 2 * np.pi * f0 * t)), coords={"t": t}, dims=["t"])

    ds = process_iv_signals(
        I_raw, U_raw, calibration=calibration, mode="ac", f0_hz=f0, subtract_zero_offsets=False
    )
    assert set(ds.data_vars) >= {"U_true", "I_total", "I_parasitic", "I_probe"}
    assert ds.attrs["processing_mode"] == "ac"
    assert abs(np.max(np.abs(ds["U_true"].values)) - 10.0) < 0.5
    assert "U_true_phasor" in ds.attrs
    assert "I_probe_phasor" in ds.attrs


def test_ac_probe_is_total_minus_parasitic(calibration):
    f0 = 20_000.0
    n = 4000
    dt = 1e-6
    t = np.arange(n) * dt
    cal = calibration.at_frequency(f0)
    u_div = 5.0 * cal.voltage_divider_transfer
    U_raw = xr.DataArray(np.real(u_div * np.exp(1j * 2 * np.pi * f0 * t)), coords={"t": t}, dims=["t"])
    I_raw = xr.DataArray(0.02 * np.sin(2 * np.pi * f0 * t), coords={"t": t}, dims=["t"])
    ds = process_iv_signals(
        I_raw, U_raw, calibration=calibration, mode="ac", f0_hz=f0, subtract_zero_offsets=False
    )
    np.testing.assert_allclose(ds["I_probe"].values, ds["I_total"].values - ds["I_parasitic"].values)


def test_ac_probe_keeps_harmonics(calibration):
    """I_probe = I_total - I_parasitic; harmonics and noise stay in the waveform."""
    from golem_toolkit.probes._signal import fft_complex_amplitude, sinusoid_from_complex

    f0 = 20_000.0
    n = 4000
    dt = 1e-6
    t = np.arange(n) * dt
    cal = calibration.at_frequency(f0)
    u_tip_phasor = 10.0 + 0.0j
    u_div_phasor = u_tip_phasor * cal.voltage_divider_transfer
    i_phasor = 0.05 + 0.0j
    harmonic = 0.01 * np.sin(2 * np.pi * 2 * f0 * t)

    U_raw = xr.DataArray(
        np.real(u_div_phasor * np.exp(1j * 2 * np.pi * f0 * t)), coords={"t": t}, dims=["t"]
    )
    I_raw = xr.DataArray(
        np.real(i_phasor * np.exp(1j * 2 * np.pi * f0 * t)) + harmonic, coords={"t": t}, dims=["t"]
    )

    ds = process_iv_signals(
        I_raw, U_raw, calibration=calibration, mode="ac", f0_hz=f0, subtract_zero_offsets=False
    )
    i_probe_phasor = ds.attrs["I_probe_phasor"]
    pure_f0 = sinusoid_from_complex(t, f0, i_probe_phasor)
    np.testing.assert_allclose(ds["I_probe"].values, pure_f0 + harmonic, rtol=1e-10, atol=1e-12)
    assert "I_probe_phasor" in ds.attrs
    assert abs(fft_complex_amplitude(ds["I_probe"], f0) - i_probe_phasor) < 1e-10


def test_dc_ramp_processing(calibration):
    n = 500
    t = np.linspace(0, 1e-3, n)
    u = np.linspace(-5, 5, n)
    dc = calibration.dc_limits()
    i = dc.dc_leak_conductance_s * u * 2.0

    U_raw = xr.DataArray(u * float(np.real(dc.voltage_divider_transfer)), coords={"t": t}, dims=["t"])
    I_raw = xr.DataArray(i / dc.shunt_path_correction_real, coords={"t": t}, dims=["t"])

    ds = process_iv_signals(I_raw, U_raw, calibration=calibration, mode="dc", subtract_zero_offsets=False)
    np.testing.assert_allclose(ds["U_true"].values, u, rtol=0.05)
    np.testing.assert_allclose(ds["I_probe"].values, i - dc.dc_leak_conductance_s * u, rtol=0.15)
