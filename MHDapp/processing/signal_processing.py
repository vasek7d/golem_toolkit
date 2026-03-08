from loguru import logger

from processing.utils import CONSTANTS

import xarray as xr

import processing.utils as ut
import numpy as np
import scipy as sc



def spectrogram(shot_no: int, coil_id: int, nperseg: int, hop: int, win: str = 'hann') -> xr.DataArray:
    logger.info(f"Loading data for spectrogram computation: shot {shot_no}, coil {coil_id}")

    data = ut.load_MHDring_data(shot_no=shot_no)

    logger.debug(f"Loaded data: {data}")
    
    logger.info(f"Selecting time interval for spectrogram computation: shot {shot_no}, coil {coil_id}")
    data = ut.select_plasma_interval(shot_no=shot_no, data=data)

    win = sc.signal.windows.hann(nperseg)
    n_sample = np.shape(data.time)[0]

    SFT = sc.signal.ShortTimeFFT(win, hop, CONSTANTS["sampling_frequency"])
    
    Sx = xr.DataArray(
        data=SFT.stft(data.sel(channel=coil_id-1)),
        dims=["frequency", "time"],
        coords={
            "frequency": SFT.f,
            "time": np.add(SFT.t(n_sample)*1_000, data.time.values[0])
        }
    )

    return Sx


# TODO: specify type hints
def compute_pearson_correlation(data, ref_data):
    output = []
    for lag in range(0, int(np.ceil(np.shape(data)[0]/2))):
        output.append(sc.stats.pearsonr(ref_data[lag:-1], data[0:-1-lag]).statistic)

    output.reverse()

    for lag in range(1, int(np.ceil(np.shape(data)[0]/2))):
        output.append(sc.stats.pearsonr(ref_data[0:-1-lag], data[lag:-1]).statistic)

    return output


# TODO: specify type hints
def correlation(shot_no: int, l_interval: list[float], ref_coil: int, ma_win_len: int = 5) -> xr.DataArray:
    data = ut.load_MHDring_data(shot_no=shot_no)
    
    # Select only the time interval of interest
    data = data.sel(time=slice(l_interval[0], l_interval[1]))

    # Apply moving average to all signals
    # data = data.rolling({f'time_{ref_coil}': ma_win_len}, center=True).mean().dropna(dim=f'time_{ref_coil}')
    # Compute correlation of all signals with the reference coil
    # TODO: use xarray.DataArray
    corr = []
    for coil_id in ut.COIL_IDS:
        corr.append(
            compute_pearson_correlation(
                data=data.sel(channel=coil_id-1),
                ref_data=data.sel(channel=ref_coil-1)
                )
            )

    # Create lag time array
    lag = np.linspace((l_interval[1] - l_interval[0]) / 2, -(l_interval[1] - l_interval[0]) / 2, int(np.shape(corr[0])[0]))

    corr = xr.DataArray(
        data=corr,
        dims=["channel", "lag"],
        coords={
            "channel": data.channel,
            "lag": lag
        }
    )

    return corr