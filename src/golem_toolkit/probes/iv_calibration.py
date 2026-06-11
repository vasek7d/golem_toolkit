"""Load and query active-probe IV calibration NetCDF files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

LEGACY_ALIASES = {
    "H_divider": "voltage_divider_transfer",
    "Y_open": "open_fixture_admittance",
    "K_current": "shunt_path_correction",
    "G_leak_S": "dc_leak_conductance_s",
    "C_open": "open_fixture_effective_capacitance",
    "C_open_corrected": "open_fixture_effective_capacitance_corrected",
    "G_open": "open_fixture_leak_conductance",
    "G_open_corrected": "open_fixture_leak_conductance_corrected",
    "delta_t_scope_s": "scope_current_voltage_deskew_s",
    "shunt_ohm": "shunt_resistance_ohm",
    "zero_offset_U": "divider_zero_offset_v",
    "zero_offset_I": "shunt_zero_offset_a",
}


def _interp_complex(ds: xr.Dataset, base: str, f0_hz: float) -> complex:
    freq = ds["freq"].to_numpy()
    re = ds[f"{base}_real"].to_numpy()
    im = ds[f"{base}_imag"].to_numpy()
    if f0_hz <= freq[0]:
        return complex(re[0], im[0])
    if f0_hz >= freq[-1]:
        return complex(re[-1], im[-1])
    re_i = np.interp(f0_hz, freq, re)
    im_i = np.interp(f0_hz, freq, im)
    return complex(re_i, im_i)


def _interp_real(ds: xr.Dataset, name: str, f0_hz: float) -> float:
    freq = ds["freq"].to_numpy()
    values = ds[name].to_numpy()
    return float(np.interp(f0_hz, freq, values))


@dataclass(frozen=True)
class FrequencyCalibration:
    """Calibration values at one frequency.

    ``voltage_divider_transfer`` is H measured with inter-channel deskew applied in the
    direct-input calibration (divider transfer function). ``scope_current_voltage_deskew_s``
    is the same Δt applied explicitly to the current phasor when processing shots.
    """

    f0_hz: float
    voltage_divider_transfer: complex
    open_fixture_admittance: complex
    shunt_path_correction: complex
    open_fixture_effective_capacitance: float
    open_fixture_effective_capacitance_corrected: float
    open_fixture_leak_conductance: float
    open_fixture_leak_conductance_corrected: float
    scope_current_voltage_deskew_s: float


@dataclass(frozen=True)
class DCCalibration:
    """DC / quasi-static calibration limits."""

    voltage_divider_transfer: complex
    shunt_path_correction_real: float
    dc_leak_conductance_s: float
    dc_effective_capacitance_f: float
    divider_zero_offset_v: float
    shunt_zero_offset_a: float


class IVCalibration:
    """Wrapper around the calibration ``xr.Dataset``."""

    def __init__(self, ds: xr.Dataset):
        self.ds = ds

    @classmethod
    def from_netcdf(cls, path: str | Path) -> IVCalibration:
        return cls(load_iv_calibration(path))

    def at_frequency(self, f0_hz: float) -> FrequencyCalibration:
        return FrequencyCalibration(
            f0_hz=float(f0_hz),
            voltage_divider_transfer=_interp_complex(self.ds, "voltage_divider_transfer", f0_hz),
            open_fixture_admittance=_interp_complex(self.ds, "open_fixture_admittance", f0_hz),
            shunt_path_correction=_interp_complex(self.ds, "shunt_path_correction", f0_hz),
            open_fixture_effective_capacitance=_interp_real(
                self.ds, "open_fixture_effective_capacitance", f0_hz
            ),
            open_fixture_effective_capacitance_corrected=_interp_real(
                self.ds, "open_fixture_effective_capacitance_corrected", f0_hz
            ),
            open_fixture_leak_conductance=_interp_real(self.ds, "open_fixture_leak_conductance", f0_hz),
            open_fixture_leak_conductance_corrected=_interp_real(
                self.ds, "open_fixture_leak_conductance_corrected", f0_hz
            ),
            scope_current_voltage_deskew_s=float(self.ds.attrs["scope_current_voltage_deskew_s"]),
        )

    def parasitics_at_frequency(self, f0_hz: float) -> dict[str, float]:
        cal = self.at_frequency(f0_hz)
        return {
            "open_fixture_effective_capacitance": cal.open_fixture_effective_capacitance,
            "open_fixture_effective_capacitance_corrected": cal.open_fixture_effective_capacitance_corrected,
            "open_fixture_leak_conductance": cal.open_fixture_leak_conductance,
            "open_fixture_leak_conductance_corrected": cal.open_fixture_leak_conductance_corrected,
        }

    def dc_limits(self) -> DCCalibration:
        attrs = self.ds.attrs
        return DCCalibration(
            voltage_divider_transfer=complex(
                float(attrs["voltage_divider_transfer_dc_real"]),
                float(attrs["voltage_divider_transfer_dc_imag"]),
            ),
            shunt_path_correction_real=float(attrs["shunt_path_correction_dc_real"]),
            dc_leak_conductance_s=float(attrs["dc_leak_conductance_s"]),
            dc_effective_capacitance_f=float(attrs.get("dc_effective_capacitance_f", 0.0)),
            divider_zero_offset_v=float(attrs["divider_zero_offset_v"]),
            shunt_zero_offset_a=float(attrs["shunt_zero_offset_a"]),
        )


def load_iv_calibration(path: str | Path) -> xr.Dataset:
    """Load and validate an active-probe calibration NetCDF file."""
    ds = xr.load_dataset(path)
    required_vars = [
        "voltage_divider_transfer_real",
        "voltage_divider_transfer_imag",
        "open_fixture_admittance_real",
        "open_fixture_admittance_imag",
        "shunt_path_correction_real",
        "shunt_path_correction_imag",
        "open_fixture_effective_capacitance",
        "open_fixture_effective_capacitance_corrected",
        "open_fixture_leak_conductance",
        "open_fixture_leak_conductance_corrected",
    ]
    missing = [name for name in required_vars if name not in ds]
    if missing:
        raise ValueError(f"Calibration file missing variables: {missing}")
    required_attrs = [
        "scope_current_voltage_deskew_s",
        "divider_zero_offset_v",
        "shunt_zero_offset_a",
        "dc_leak_conductance_s",
        "voltage_divider_transfer_dc_real",
        "shunt_path_correction_dc_real",
    ]
    missing_attrs = [name for name in required_attrs if name not in ds.attrs]
    if missing_attrs:
        raise ValueError(f"Calibration file missing attrs: {missing_attrs}")
    return ds
