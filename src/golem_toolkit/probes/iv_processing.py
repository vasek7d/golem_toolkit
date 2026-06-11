"""Process raw probe IV waveforms using active-probe calibration.

AC mode (single tone at ``f0_hz``) builds phasors at the drive frequency, then
time traces. See ``docs/probes/IV_CORRECTION.md`` for the full pipeline.

Summary at f0:

- ``U_true_phasor = FFT(U_work) / H`` — divider correction (CH2); no deskew on shots
- ``I_total_phasor = FFT(I_work) × exp(j2πf0Δt) × K`` — deskew aligns CH1 to CH2, then shunt path
- ``I_parasitic_phasor = Y_open × K × U_true_phasor``
- ``I_probe_phasor = I_total_phasor − I_parasitic_phasor``

Waveforms: ``U_true`` and ``I_total`` use :func:`~golem_toolkit.probes._signal.replace_fundamental`
(calibrated f0, harmonics/noise kept). ``I_parasitic`` is a pure f0 sinusoid.
``I_probe = I_total − I_parasitic`` (not a separate ``replace_fundamental`` on raw current).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import xarray as xr

from golem_toolkit.probes._signal import (
    fft_complex_amplitude,
    normalize_time_coord,
    replace_fundamental,
    sinusoid_from_complex,
    time_coord_values,
)
from golem_toolkit.probes.iv_calibration import IVCalibration

ProcessingMode = Literal["ac", "dc"]


def process_iv_signals(
    I_raw: xr.DataArray,
    U_raw: xr.DataArray,
    *,
    calibration: xr.Dataset | IVCalibration,
    mode: ProcessingMode,
    f0_hz: float | None = None,
    subtract_zero_offsets: bool = True,
) -> xr.Dataset:
    """
    Correct raw shunt current and divider output to tip voltage and probe current.

    Parameters
    ----------
    I_raw
        Total current through the shunt (A), coordinate ``t`` or ``time``.
    U_raw
        Divider output voltage (V), same time coordinate as ``I_raw``.
    calibration
        Calibration dataset or :class:`IVCalibration` wrapper.
    mode
        ``\"ac\"`` for single-tone AC at ``f0_hz``; ``\"dc\"`` for slow IV sweeps.
    f0_hz
        Drive frequency in Hz (required for ``mode='ac'``).
    subtract_zero_offsets
        Subtract divider/shunt zero offsets stored in the calibration file.

    Returns
    -------
    xarray.Dataset
        Variables ``I_raw``, ``U_raw``, ``U_true``, ``I_total``, ``I_parasitic``, ``I_probe``
        on coordinate ``t``. AC mode also sets attrs ``U_true_phasor``, ``I_probe_phasor``,
        ``f0_hz``, and applied calibration values. Use the phasor attrs for conductance at f0;
        use the waveforms for IV plots (``I_probe`` retains harmonics and noise).
    """
    cal = calibration if isinstance(calibration, IVCalibration) else IVCalibration(calibration)
    I_raw = normalize_time_coord(I_raw)
    U_raw = normalize_time_coord(U_raw)
    if I_raw.sizes["t"] != U_raw.sizes["t"]:
        raise ValueError("I_raw and U_raw must share the same length on coordinate t")

    if mode == "ac":
        if f0_hz is None:
            raise ValueError("f0_hz is required for mode='ac'")
        return _process_ac(I_raw, U_raw, cal, float(f0_hz), subtract_zero_offsets)
    if mode == "dc":
        return _process_dc(I_raw, U_raw, cal, subtract_zero_offsets)
    raise ValueError(f"Unknown mode: {mode!r}")


def _apply_offsets(
    I_raw: xr.DataArray, U_raw: xr.DataArray, cal: IVCalibration, subtract: bool
) -> tuple[xr.DataArray, xr.DataArray]:
    if not subtract:
        return I_raw, U_raw
    dc = cal.dc_limits()
    I = I_raw - dc.shunt_zero_offset_a
    U = U_raw - dc.divider_zero_offset_v
    return I, U


def _process_ac(
    I_raw: xr.DataArray,
    U_raw: xr.DataArray,
    cal: IVCalibration,
    f0_hz: float,
    subtract_zero_offsets: bool,
) -> xr.Dataset:
    """AC single-tone correction at ``f0_hz`` (see module docstring)."""
    I_work, U_work = _apply_offsets(I_raw, U_raw, cal, subtract_zero_offsets)
    t = time_coord_values(I_work)
    t0 = float(t[0])
    t_rel = t - t0

    freq_cal = cal.at_frequency(f0_hz)
    h = freq_cal.voltage_divider_transfer
    y_open = freq_cal.open_fixture_admittance
    k = freq_cal.shunt_path_correction
    dt = freq_cal.scope_current_voltage_deskew_s

    # Phasors at f0 (see IV_CORRECTION.md): divider on U; deskew + K on I.
    u_phasor = fft_complex_amplitude(U_work, f0_hz) / h
    i_phasor = fft_complex_amplitude(I_work, f0_hz) * np.exp(1j * 2.0 * np.pi * f0_hz * dt) * k
    y_open_eff = y_open * k
    i_par_phasor = y_open_eff * u_phasor
    i_probe_phasor = i_phasor - i_par_phasor

    # Waveforms: replace f0 in U_total and I_total; I_parasitic is f0-only; I_probe by subtraction.
    u_gain_only = U_work.to_numpy() / abs(h)
    u_true = replace_fundamental(t_rel, u_gain_only, f0_hz, u_phasor)
    i_total = replace_fundamental(t_rel, I_work.to_numpy(), f0_hz, i_phasor)
    i_parasitic = sinusoid_from_complex(t_rel, f0_hz, i_par_phasor)
    i_probe = i_total - i_parasitic

    return xr.Dataset(
        {
            "I_raw": I_raw,
            "U_raw": U_raw,
            "U_true": xr.DataArray(u_true, dims=["t"], coords={"t": t}),
            "I_total": xr.DataArray(i_total, dims=["t"], coords={"t": t}),
            "I_parasitic": xr.DataArray(i_parasitic, dims=["t"], coords={"t": t}),
            "I_probe": xr.DataArray(i_probe, dims=["t"], coords={"t": t}),
        },
        attrs={
            "processing_mode": "ac",
            "f0_hz": f0_hz,
            "U_true_phasor": complex(u_phasor),
            "I_probe_phasor": complex(i_probe_phasor),
            "scope_current_voltage_deskew_applied_s": dt,
            "voltage_divider_transfer_applied": complex(h),
            "open_fixture_admittance_applied": complex(y_open),
            "shunt_path_correction_applied": complex(k),
            "legacy_voltage_divider_transfer": "H_divider",
            "legacy_open_fixture_admittance": "Y_open",
            "legacy_shunt_path_correction": "K_current",
            "legacy_scope_deskew": "delta_t_scope_s",
        },
    )


def _process_dc(
    I_raw: xr.DataArray,
    U_raw: xr.DataArray,
    cal: IVCalibration,
    subtract_zero_offsets: bool,
) -> xr.Dataset:
    I_work, U_work = _apply_offsets(I_raw, U_raw, cal, subtract_zero_offsets)
    t = time_coord_values(I_work)
    dc = cal.dc_limits()

    h_dc = dc.voltage_divider_transfer
    gain = float(np.real(h_dc)) if np.real(h_dc) != 0.0 else abs(h_dc)
    u_true = U_work.to_numpy() / gain
    i_total = I_work.to_numpy() * dc.shunt_path_correction_real
    i_parasitic = dc.dc_leak_conductance_s * u_true
    i_probe = i_total - i_parasitic

    return xr.Dataset(
        {
            "I_raw": I_raw,
            "U_raw": U_raw,
            "U_true": xr.DataArray(u_true, dims=["t"], coords={"t": t}),
            "I_total": xr.DataArray(i_total, dims=["t"], coords={"t": t}),
            "I_parasitic": xr.DataArray(i_parasitic, dims=["t"], coords={"t": t}),
            "I_probe": xr.DataArray(i_probe, dims=["t"], coords={"t": t}),
        },
        attrs={
            "processing_mode": "dc",
            "voltage_divider_transfer_dc_applied": complex(h_dc),
            "dc_leak_conductance_applied": dc.dc_leak_conductance_s,
            "shunt_path_correction_dc_applied": dc.shunt_path_correction_real,
            "legacy_dc_leak_conductance": "G_leak_S",
        },
    )
