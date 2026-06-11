"""Signal helpers for probe IV processing."""

from __future__ import annotations

import numpy as np
import xarray as xr


def normalize_time_coord(da: xr.DataArray) -> xr.DataArray:
    """Rename the time dimension/coord to ``t`` if needed."""
    if "t" in da.dims or "t" in da.coords:
        return da
    if "time" in da.dims:
        return da.rename({"time": "t"})
    if "time" in da.coords and da.dims == ("time",):
        return da.rename(time="t")
    raise ValueError(f"DataArray must have a 't' or 'time' coordinate, got dims={da.dims}")


def time_coord_values(da: xr.DataArray) -> np.ndarray:
    da = normalize_time_coord(da)
    return da["t"].to_numpy().astype(np.float64)


def fft_complex_amplitude(da: xr.DataArray, f0_hz: float) -> complex:
    """One-sided complex sinusoid amplitude at ``f0_hz`` (peak cosine convention)."""
    da = normalize_time_coord(da)
    x = da.to_numpy().astype(np.float64)
    n = len(x)
    t = time_coord_values(da)
    dt = float(t[1] - t[0]) if n > 1 else 1.0
    spectrum = np.fft.rfft(x)
    k = int(round(f0_hz * n * dt))
    k = min(max(k, 0), len(spectrum) - 1)
    return complex(2.0 * spectrum[k] / n)


def sinusoid_from_complex(time_s: np.ndarray, f0_hz: float, amplitude: complex) -> np.ndarray:
    return np.real(amplitude * np.exp(1j * 2.0 * np.pi * f0_hz * time_s))


def replace_fundamental(
    time_s: np.ndarray,
    signal: np.ndarray,
    f0_hz: float,
    target_phasor: complex,
) -> np.ndarray:
    """Replace the f0 fundamental with a calibrated phasor; keep harmonics and noise.

    Equivalent to subtracting the measured f0 sinusoid and adding
    ``sinusoid_from_complex(time_s, f0_hz, target_phasor)``. Used for ``U_true`` and
    ``I_total`` in AC processing. Not used for ``I_probe`` (that is ``I_total − I_parasitic``).
    """
    da = xr.DataArray(signal, coords={"t": time_s})
    c_raw = fft_complex_amplitude(da, f0_hz)
    fund_raw = sinusoid_from_complex(time_s, f0_hz, c_raw)
    fund_target = sinusoid_from_complex(time_s, f0_hz, target_phasor)
    return signal - fund_raw + fund_target
