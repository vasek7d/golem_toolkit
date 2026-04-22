import numpy as np
# import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
import hashlib, json, tempfile, os, contextlib, re

from golem_toolkit.loading import basic as load

def _fingerprint_for_cache(key: dict) -> str:
    """Stable short fingerprint for any JSON-serializable object."""
    return hashlib.sha1(
        json.dumps(key, sort_keys=True, default=str).encode()
    ).hexdigest()[:10]

def cache_file_for(
    shot_no,
    data_url,
    das_settings,
    n_channels,
    names=None,
    skiprows=0,
    time_units="s",
    cache_dir="DATA_CACHE",
):
    """Return the exact Path where this call would cache the dataset."""
    if names is None:
        names = ["t"] + [f"CH{i}" for i in range(1, n_channels+1)]
    key = {
        "shot_no": shot_no,
        "url": data_url,
        "settings": das_settings,
        "n_channels": n_channels,
        "names": names,
        "skiprows": skiprows,
        "time_units": time_units,
        "version": 1,
    }
    fp = _fingerprint_for_cache(key)
    return Path(cache_dir) / f"DAS_{shot_no}_{fp}.nc"


def list_cache(cache_dir="DATA_CACHE", shot_no=None):
    """
    List cached .nc files. If shot_no is given, only list caches for that shot.
    Returns a list of Paths sorted by modified time (newest first).
    """
    cdir = Path(cache_dir)
    if not cdir.exists():
        return []
    pattern = f"DAS_{shot_no}_" if shot_no is not None else "DAS_"
    files = [p for p in cdir.glob("*.nc") if p.name.startswith(pattern)]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def clear_cache(cache_dir="DATA_CACHE", shot_no=None, keep_latest=False, verbose=1):
    """
    Delete cache files.
      - If shot_no is None: delete all caches in cache_dir.
      - If shot_no is set: delete only that shot's caches.
      - If keep_latest=True: for each shot, keep the newest file and delete the rest.

    Returns (deleted_paths: list[Path], kept_paths: list[Path])
    """
    cdir = Path(cache_dir)
    if not cdir.exists():
        return ([], [])

    if shot_no is None:
        # group by shot inferred from filename DAS_<shot>_<fp>.nc
        groups = {}
        for p in cdir.glob("DAS_*_*.nc"):
            m = re.match(r"DAS_(.+?)_[0-9a-f]{10}\.nc$", p.name)
            if not m:
                continue
            sh = m.group(1)
            groups.setdefault(sh, []).append(p)
    else:
        groups = {str(shot_no): list_cache(cache_dir, shot_no=shot_no)}

    deleted, kept = [], []
    for sh, files in groups.items():
        files_sorted = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
        if keep_latest and files_sorted:
            kept.append(files_sorted[0])
            to_delete = files_sorted[1:]
        else:
            to_delete = files_sorted

        for p in to_delete:
            try:
                p.unlink()
                deleted.append(p)
                if verbose:
                    print(f"[clear_cache] Deleted {p}")
            except Exception as e:
                if verbose:
                    print(f"[clear_cache] Could not delete {p} ({e})")

    if verbose and kept:
        for p in kept:
            print(f"[clear_cache] Kept latest for shot: {p}")

    return (deleted, kept)

def load_DAS(
    shot_no,
    data_url,
    das_settings,
    n_channels,
    names=None,
    skiprows=0,
    time_units="s",
    verbose=0,
    *,
    cache=False,
    cache_dir="DATA_CACHE",
    force_refresh=False,
):
    """Load DAS data into an xarray.Dataset with optional caching."""

    # ---------- Cache setup ----------
    if names is None:
        names = ["t"] + [f"CH{i}" for i in range(1, n_channels+1)]

    if cache:
        key = {
            "shot_no": shot_no,
            "url": data_url,
            "settings": das_settings,
            "n_channels": n_channels,
            "names": names,
            "skiprows": skiprows,
            "time_units": time_units,
            "version": 1,
        }
        fingerprint = _fingerprint_for_cache(key)
        
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(exist_ok=True)
        cache_file = cache_dir / f"DAS_{shot_no}_{fingerprint}.nc"
        if verbose > 1:
            print(f"[load_DAS] cache_file = {cache_file}")

    # ---------- Try cache ----------
    if cache and cache_file.exists() and not force_refresh:
        if verbose:
            print(f"[load_DAS] Loading from cache {cache_file}")
        try:
            return xr.load_dataset(cache_file)
        except Exception as e:
            if verbose:
                print(f"[load_DAS] Cache read failed ({e}); rebuilding...")

    # ---------- Load fresh data ----------
    pd_data = load.load_array(data_url, names=names, index_col="t", verbose=verbose, skiprows=skiprows)
    if pd_data is None:
        return None

    data = pd_data.to_xarray()
    DAS = xr.Dataset()

    for channel, settings in das_settings.items():
        DAS[settings["var_name"]] = data[channel] * settings["scaling_factor"]
        DAS[settings["var_name"]].attrs = settings["attrs"]

    DAS.attrs = {"shot_no": shot_no}

    factor = 1 if time_units == "s" else 1e3
    DAS = DAS.rename({"index": "t"}) if "index" in DAS.dims else DAS
    DAS = DAS.assign_coords(t=DAS.t * factor)
    DAS.t.attrs["units"] = time_units

    # ---------- Save to cache ----------
    if cache:
        if verbose:
            print(f"[load_DAS] Saving to cache {cache_file}")
        fd, tmp_name = tempfile.mkstemp(suffix=".nc", dir=str(cache_dir))
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            DAS.to_netcdf(tmp_path)
            tmp_path.replace(cache_file)   # atomic replace
        except Exception:
            with contextlib.suppress(Exception):
                print(f"[load_DAS] Saving to cache failed!")
                tmp_path.unlink()
            raise

    return DAS

def plot_DAS(DAS_dataset, DAS_name=None, figsize=None, filename=None):
    if DAS_dataset is None:
        raise ValueError("DAS_dataset is None.")

    # Normalize supported inputs to (name, DataArray) pairs.
    if isinstance(DAS_dataset, xr.DataArray):
        items = [(DAS_dataset.name or "signal", DAS_dataset)]
        attrs = getattr(DAS_dataset, "attrs", {})
    elif isinstance(DAS_dataset, xr.Dataset):
        items = list(DAS_dataset.data_vars.items())
        attrs = getattr(DAS_dataset, "attrs", {})
    elif hasattr(DAS_dataset, "items"):
        items = list(DAS_dataset.items())
        attrs = {}
    else:
        raise TypeError(
            "DAS_dataset must be xarray Dataset/DataArray or dict-like of DataArrays."
        )

    if len(items) == 0:
        raise ValueError("DAS_dataset has no data variables to plot.")

    set_figsize = figsize if figsize else (8, max(1, len(items)) * 3)
    fig, ax = plt.subplots(
        len(items), 1,
        figsize=set_figsize,
        dpi=300,
        tight_layout=True,
        sharex=True,
        squeeze=False,
    )
    axes = ax.ravel()

    shot_no = attrs.get("shot_no")
    supt = f"#{shot_no}" if shot_no is not None else "DAS signals"
    if DAS_name:
        supt += f" (DAS = {DAS_name})"
    fig.suptitle(supt)

    for axi, (var, da) in zip(axes, items):
        if da is None:
            axi.text(0.5, 0.5, f"{var}: no data", ha="center", va="center")
            axi.set_axis_off()
            continue

        if isinstance(da, xr.DataArray):
            # Use native xarray plotting when coordinate context exists.
            if da.ndim == 0:
                axi.text(0.5, 0.5, f"{var}: scalar = {da.item()}", ha="center", va="center")
                axi.set_axis_off()
                continue
            if da.ndim > 1:
                da = da.squeeze(drop=True)
                if da.ndim > 1:
                    da = da.isel({d: 0 for d in da.dims[1:]})

            if "t" in da.coords:
                da.plot(ax=axi, x="t")
            else:
                da.plot(ax=axi)
            probe_name = da.attrs.get("probe", var)
        else:
            # Last-resort fallback for plain array-like objects.
            axi.plot(np.asarray(da))
            probe_name = var

        axi.set_title(f"probe {probe_name}")
        axi.grid(True)

    if filename:
        fig.savefig(filename)

    plt.show()

class REDPITAYA:
    def __init__(self, das_settings, DAS_name="REDPITAYA", data_url_template=None):
        self.DAS_name = DAS_name

        # Default template if not provided
        if data_url_template is None:
            data_url_template = (
                "http://golem.fjfi.cvut.cz/shotdir/{shot_no}/Devices/ITs/SignalLab-a/data.csv"
            )
        self.data_url_template = data_url_template

        self.n_channels = 2  # number of DAS channels
        self.skiprows = 1
        self.das_settings = das_settings

    def get_data_url(self, shot_no):
        return self.data_url_template.format(shot_no=shot_no)

    def load_data(self, shot_no, **load_DAS_kwargs):
        data_url = self.get_data_url(shot_no)
        self.data = load_DAS(shot_no,
                             data_url,
                             self.das_settings,
                             self.n_channels,
                             skiprows=self.skiprows,
                             **load_DAS_kwargs)

        # return self.data

    def plot(self, **kwargs):
        try:
            data = getattr(self, "data")
        except Exception as e:
            print(f"The data has not been loaded yet! Error: {e}")
        plot_DAS(self.data, self.DAS_name, **kwargs)


class TEK64:
    def __init__(self, das_settings, DAS_name="TEK64", data_url_template=None):
        self.DAS_name = DAS_name

        # Default template if not provided
        if data_url_template is None:
            data_url_template = (
                "http://golem.fjfi.cvut.cz/shotdir/{shot_no}/Diagnostics/LangBallPenProbe/DAS_raw_data_dir/TektrMSO64_ALL.csv"
            )
        self.data_url_template = data_url_template

        self.n_channels = 4  # number of DAS channels
        self.skiprows = 12
        self.das_settings = das_settings

    def get_data_url(self, shot_no):
        return self.data_url_template.format(shot_no=shot_no)

    def load_data(self, shot_no, **load_DAS_kwargs):
        data_url = self.get_data_url(shot_no)
        self.data = load_DAS(shot_no,
                             data_url,
                             self.das_settings,
                             self.n_channels,
                             skiprows=self.skiprows,
                             **load_DAS_kwargs)
        
    

        # return self.data

    def plot(self, **kwargs):
        try:
            data = getattr(self, "data")
        except Exception as e:
            print(f"The data has not been loaded yet! \nError: {e}")

        plot_DAS(self.data, self.DAS_name, **kwargs)
        
        
class PAPOUCH:
    def __init__(self, das_settings, DAS_name="PAPOUCH", data_url_template=None, channel_files=None):
        self.DAS_name = DAS_name

        if data_url_template is None:
            data_url_template = (
                "http://golem.fjfi.cvut.cz/shots/{shot_no}/Diagnostics/Diamagnetism/"
            )

        if channel_files is None:
            channel_files = {
                "CH1": "U_loop.csv",
                "CH2": "U_rog-coil-out.csv",
                "CH3": "U_rog-coil-in.csv",
                "CH4": "U_diam-in.csv",
                "CH5": "U_diam-out.csv",
                "CH6": "U_tor-hfs.csv",
                "CH7": "U_tor-lfs.csv",
            }

        self.data_url_template = data_url_template
        self.n_channels = 7  # number of DAS channels
        self.skiprows = 0
        self.das_settings = das_settings
        self.channel_files = channel_files

    def get_data_url(self, shot_no):
        return self.data_url_template.format(shot_no=shot_no)

    def load_data(
        self,
        shot_no,
        time_units="s",
        verbose=0,
        cache=False,
        cache_dir="DATA_CACHE",
        force_refresh=False,
    ):
        # ---------- Cache setup ----------
        if cache:
            key = {
                "shot_no": shot_no,
                "type": "PAPOUCH",
                "data_url_template": self.data_url_template,
                "channel_files": self.channel_files,
                "das_settings": self.das_settings,
                "time_units": time_units,
                "version": 1,
            }
            fingerprint = _fingerprint_for_cache(key)

            cache_dir = Path(cache_dir)
            cache_dir.mkdir(exist_ok=True)
            cache_file = cache_dir / f"DAS_{shot_no}_{fingerprint}.nc"
            if verbose > 1:
                print(f"[PAPOUCH.load_data] cache_file = {cache_file}")

        # ---------- Try cache ----------
        if cache and cache_file.exists() and not force_refresh:
            if verbose:
                print(f"[PAPOUCH.load_data] Loading from cache {cache_file}")
            try:
                self.data = xr.load_dataset(cache_file)
                return self.data
            except Exception as e:
                if verbose:
                    print(f"[PAPOUCH.load_data] Cache read failed ({e}); rebuilding...")

        # ---------- Load fresh data ----------
        data_vars = {}
        base_url = self.get_data_url(shot_no)

        for channel, settings in self.das_settings.items():
            filename = self.channel_files.get(channel)
            if filename is None:
                if verbose > 0:
                    print(f"Warning: Missing filename mapping for {channel}, skipping.")
                continue
            url = base_url + filename
            try:
                pd_data = load.load_array(
                    url,
                    names=["t", channel],
                    index_col=0,
                    verbose=verbose,
                )
                if pd_data is None:
                    if verbose > 0:
                        print(f"Warning: Empty data for {channel} at {url}")
                    continue
                da = pd_data.to_xarray()[channel] * settings["scaling_factor"]
                da = da.rename(settings["var_name"])
                da.attrs = dict(settings["attrs"])
                da.attrs["source"] = url
                data_vars[settings["var_name"]] = da
            except Exception as e:
                if verbose > 0:
                    print(f"Warning: Could not load {channel} for shot {shot_no}: {e}")

        if not data_vars:
            raise ValueError(f"No data could be loaded for shot {shot_no}")

        self.data = xr.Dataset(data_vars)
        self.data.attrs["shot_no"] = shot_no
        self.data.attrs["source"] = "PAPOUCH DAS"
        factor = 1 if time_units == "s" else 1e3
        self.data = self.data.rename({"index": "t"}) if "index" in self.data.dims else self.data
        self.data = self.data.assign_coords(t=self.data.t * factor)
        self.data.t.attrs["units"] = time_units

        # ---------- Save to cache ----------
        if cache:
            if verbose:
                print(f"[PAPOUCH.load_data] Saving to cache {cache_file}")
            fd, tmp_name = tempfile.mkstemp(suffix=".nc", dir=str(cache_dir))
            os.close(fd)
            tmp_path = Path(tmp_name)
            try:
                self.data.to_netcdf(tmp_path)
                tmp_path.replace(cache_file)   # atomic replace
            except Exception:
                with contextlib.suppress(Exception):
                    print("[PAPOUCH.load_data] Saving to cache failed!")
                    tmp_path.unlink()
                raise

        return self.data

    def plot(self, **kwargs):
        try:
            data = getattr(self, "data")
        except Exception as e:
            print(f"The data has not been loaded yet! \nError: {e}")

        plot_DAS(self.data, self.DAS_name, **kwargs)


if __name__ == "__main__":

    I_measurement_resistor = 47
    U_divider_factor = 42

    tek_settings = {
        "CH2": {"var_name": "Ufl",
                "scaling_factor": U_divider_factor,
                "attrs": {"units": "V",
                          "probe": "LP"},
                },
    }


    redp_settings = {
        "CH1": {"var_name": "I",
                "scaling_factor": 1/I_measurement_resistor*1e3,
                "attrs": {"units": "mA",
                          "probe": "BPP"},
                },

        "CH2": {"var_name": "U",
                "scaling_factor": U_divider_factor,
                "attrs": {"units": "V",
                          "probe": "BPP"},
                },
    }

    papouch_settings = {
        "CH1": {"var_name": "U_loop",
                "scaling_factor": 1,
                "attrs": {"units": "V",
                          "probe": "U_loop"},
                },
        "CH2": {"var_name": "U_rog_coil_out",
                "scaling_factor": 1,
                "attrs": {"units": "V",
                          "probe": "Outer Rogcoil"},
                },
        "CH3": {"var_name": "U_rog_coil_in",
                "scaling_factor": 1,
                "attrs": {"units": "V",
                          "probe": "DIA-RING"},
                },
        "CH4": {"var_name": "U_diam_in",
                "scaling_factor": 1,
                "attrs": {"units": "V",
                          "probe": "DIA-RING"},
                },
        "CH5": {"var_name": "U_diam_out",
                "scaling_factor": 1,
                "attrs": {"units": "V",
                          "probe": "DIA-RING"},
                },
        "CH6": {"var_name": "U_tor_hfs",
                "scaling_factor": 1,
                "attrs": {"units": "V",
                          "probe": "DIA-RING"},
                },
        "CH7": {"var_name": "U_tor_lfs",
                "scaling_factor": 1,
                "attrs": {"units": "V",
                          "probe": "DIA-RING"},
                },
    }

    
    calib_shot = 49808  # kolem toho
    time_units = "ms"

    RP = REDPITAYA(redp_settings)
    TEK = TEK64(tek_settings)
    PAP = PAPOUCH(papouch_settings)
    

    RP.load_data(calib_shot, time_units=time_units, verbose=1)
    TEK.load_data(calib_shot, time_units=time_units, verbose=1)
    PAP.load_data(51672, time_units=time_units, verbose=1)

    # RP.plot()
    # TEK.plot()
    # PAP.plot()
    
    print(PAP.data)