"""Map Tektronix scope CSV data to probe IV inputs."""

from __future__ import annotations

from pathlib import Path

import xarray as xr

from golem_toolkit.loading.DAS_systems import load_tek64_csv

# Backward-compatible alias (coord ``t``, via golem_toolkit.loading)
load_tektronix_csv = load_tek64_csv


def select_first_cycles(ds: xr.Dataset, f0_hz: float, n_cycles: float) -> xr.Dataset:
    """Return the first ``n_cycles`` of a periodic capture."""
    if "t" not in ds.coords and "t" not in ds.dims:
        raise ValueError("Dataset must have coordinate 't' (as from load_tek64_csv)")
    t = ds["t"].to_numpy()
    start = float(t[0])
    stop = start + n_cycles / f0_hz
    return ds.sel(t=slice(start, stop))


def probe_waveforms_from_dataset(
    ds: xr.Dataset,
    *,
    f0_hz: float,
    shunt_resistance_ohm: float,
    i_channel: str = "CH1",
    u_channel: str = "CH2",
    n_cycles: float | None = None,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Build ``(I_raw, U_raw)`` from a scope dataset (e.g. from :func:`load_tek64_csv`).

    Parameters
    ----------
    ds
        Dataset with coordinate ``t`` and scope channels (``CH1``, …).
    f0_hz
        Drive frequency in Hz (used when ``n_cycles`` trims the record).
    shunt_resistance_ohm
        Shunt resistance used to convert the current-channel voltage to amperes.
    i_channel, u_channel
        Scope channels for shunt current and divider output.
    n_cycles
        If set, keep only the first ``n_cycles`` at ``f0_hz``. ``None`` keeps the full record.
    """
    if i_channel not in ds or u_channel not in ds:
        missing = [ch for ch in (i_channel, u_channel) if ch not in ds]
        raise KeyError(f"Dataset missing scope channel(s): {missing}")

    work = select_first_cycles(ds, f0_hz, n_cycles) if n_cycles is not None else ds

    t = work["t"].to_numpy() - float(work["t"].values[0])
    i_raw = xr.DataArray(
        (work[i_channel] / shunt_resistance_ohm).to_numpy(),
        coords={"t": t},
        dims=["t"],
        attrs={"units": "A", "long_name": "shunt current", "scope_channel": i_channel},
    )
    u_raw = xr.DataArray(
        work[u_channel].to_numpy(),
        coords={"t": t},
        dims=["t"],
        attrs={"units": "V", "long_name": "divider output", "scope_channel": u_channel},
    )
    return i_raw, u_raw


def load_probe_waveforms(
    source: str | Path | xr.Dataset,
    *,
    f0_hz: float,
    shunt_resistance_ohm: float,
    i_channel: str = "CH1",
    u_channel: str = "CH2",
    n_cycles: float | None = None,
    skiprows: int = 12,
    time_units: str = "s",
) -> tuple[xr.DataArray, xr.DataArray]:
    """Return ``(I_raw, U_raw)`` for :func:`process_iv_signals`.

    Pass a Tektronix CSV path **or** an existing dataset from
    :func:`~golem_toolkit.loading.DAS_systems.load_tek64_csv` / :class:`TEK64`.

    Parameters
    ----------
    source
        CSV path, or an ``xr.Dataset`` already loaded (not read again).
    f0_hz
        Drive frequency in Hz (must be supplied explicitly).
    shunt_resistance_ohm
        Shunt resistance used to convert the current-channel voltage to amperes.
        Read from calibration: ``calib.attrs['shunt_resistance_ohm']``.
    i_channel
        Scope channel wired to the shunt (voltage across shunt → divided by
        ``shunt_resistance_ohm`` to get ``I_raw`` in amperes). Default ``CH1``.
    u_channel
        Scope channel for the divider output (``U_raw`` in volts). Default ``CH2``.
    n_cycles
        If set, keep only the first ``n_cycles`` at ``f0_hz``. ``None`` keeps the full record.
    skiprows, time_units
        Passed to :func:`load_tek64_csv` when ``source`` is a path (ignored for datasets).
    """
    if isinstance(source, xr.Dataset):
        ds = source
    else:
        channels = list(dict.fromkeys([i_channel, u_channel]))
        ds = load_tek64_csv(
            source,
            channels=channels,
            skiprows=skiprows,
            time_units=time_units,
        )
    return probe_waveforms_from_dataset(
        ds,
        f0_hz=f0_hz,
        shunt_resistance_ohm=shunt_resistance_ohm,
        i_channel=i_channel,
        u_channel=u_channel,
        n_cycles=n_cycles,
    )
