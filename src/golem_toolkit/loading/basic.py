import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import scipy.signal as signal
from scipy.integrate import cumulative_trapezoid

def load_parameter(url, param_type, verbose=0):
    try:
        if verbose:
            print(f"- Loading parameter: {url}")
        number = param_type(pd.read_csv(url, header=None).values[0, 0])
    except Exception as e:
        print("-- FAILED")
        print(e)
        return None
    else:
        if verbose:
            print("-- SUCCESS")
        return number

def load_array(url, index_col=None, names=None, verbose=0, skiprows=0):
    try:
        if verbose:
            print(f"- Loading array: {url}")
        data = pd.read_csv(
            url, delimiter=",", index_col=index_col, names=names,
            skiprows=skiprows, na_values=["inf", "-inf"],
        )
    except Exception as e:
        print("-- FAILED")
        print(e)
        return None
    else:
        if verbose:
            print("-- SUCCESS")

        num = data.select_dtypes(include=[np.number])
        cols_with_inf = num.columns[num.apply(lambda col: np.isinf(col).any())].tolist()
        if cols_with_inf:
            for col in cols_with_inf:
                inf_rows = num.index[np.isinf(num[col])]
                print(f"⚠️  Column {col!r} has infinities at rows: {list(inf_rows)}")
        return data
    
def remove_initial_offset(dataarray, t_cutoff=0, verbose=0):
    t0 = dataarray.t.isel(t=0)
    assert t_cutoff >= t0, "Cut-off time for the initial offset has to be higher than the starting time"
    offset_mean = dataarray.sel(t=slice(t0, t_cutoff)).mean()

    dataarray -= offset_mean

    if verbose:
        print(f"+++ {dataarray.name}: initial offset = {offset_mean.values:1.2e}")
    if verbose > 1:
        fig, ax = plt.subplots()
        fig.suptitle("Offset removal")
        dataarray.plot(ax=ax)
        ymin, ymax = dataarray.data.min(), dataarray.data.max()
        ax.vlines(t_cutoff, ymin, ymax, color='k')
        plt.show()
    return dataarray

def smooth_numpyarray(array, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(array, box, mode='same')
    return y_smooth

def smoothen_dataarray(dataarray, moving_avg_size, verbose=0):
    data_smooth = xr.apply_ufunc(
        smooth_numpyarray,
        dataarray.copy(),
        kwargs={"box_pts": moving_avg_size},
    )
    data_smooth.attrs = dataarray.attrs

    if verbose:
        print(data_smooth)

    return data_smooth


def create_spectrogram_da(dataarray, fs=None, nperseg=None, noverlap=None, 
                         window='hann', scaling='density', verbose=0):
    """
    Create a spectrogram from an xarray DataArray using scipy.signal.spectrogram
    
    Parameters:
    -----------
    dataarray : xarray.DataArray
        Input time series data
    fs : float, optional
        Sampling frequency (Hz). If None, will be automatically calculated from time coordinates
    nperseg : int, optional
        Length of each segment for FFT
    noverlap : int, optional
        Number of points to overlap between segments
    window : str or tuple, optional
        Window function to use
    scaling : str, optional
        Scaling mode ('density' or 'spectrum')
    verbose : int, optional
        Verbosity level
    
    Returns:
    --------
    xarray.Dataset
        Dataset containing 'linear' and 'db' DataArrays with spectrogram results
    """

    # Extract the data and time coordinates
    data = dataarray.values
    time_coord = dataarray.coords[dataarray.dims[0]]
    
    # Auto-detect sampling frequency if not provided
    if fs is None:
        if len(time_coord) > 1:
            dt = float(time_coord[1] - time_coord[0])
            fs = int(1.0 / dt)
            if verbose:
                print(f"Auto-detected sampling frequency: {fs:.0f} Hz (dt = {dt:.2e} s)")
        else:
            raise ValueError("Cannot auto-detect sampling frequency: need at least 2 time points")
    
    if verbose:
        print(f"Creating spectrogram for {dataarray.name}")
        print(f"  - Sampling rate: {fs} Hz")
        print(f"  - Data length: {len(dataarray)} points")
        print(f"  - Time range: {time_coord[0]:.3f} to {time_coord[-1]:.3f} s")
    
    # Calculate default parameters if not provided
    if nperseg is None:
        nperseg = min(256, len(data) // 4)
    if noverlap is None:
        noverlap = nperseg // 2
    
    if verbose:
        print(f"  - Window length: {nperseg} points")
        print(f"  - Overlap: {noverlap} points")
        print(f"  - Window function: {window}")
        print(f"  - Scaling: {scaling}")
    
    # Compute spectrogram using scipy
    frequencies, times, Sxx = signal.spectrogram(
        data, fs=fs, nperseg=nperseg, noverlap=noverlap, 
        window=window, scaling=scaling
    )
    
    # Create time coordinates for spectrogram
    # The times from spectrogram are relative to the start of the signal
    spectrogram_times = float(time_coord[0]) + times
 
    
    # Create the linear spectrogram DataArray
    spectrogram_linear = xr.DataArray(
        Sxx,
        coords={
            'frequency': frequencies,
            'time': spectrogram_times
        },
        dims=['frequency', 'time'],
        attrs={
            'long_name': 'Power spectral density (linear)',
            'units': 'V²/Hz' if scaling == 'density' else 'V²',
            'sampling_rate': fs,
            'window': window,
            'nperseg': nperseg,
            'noverlap': noverlap,
            'scaling': scaling,
            'original_dataarray_name': dataarray.name
        }
    )
    
    # Create dB scale spectrogram
    # Add small value to avoid log(0) issues
    Sxx_db = 10 * np.log10(Sxx + 1e-12)
    
    spectrogram_db = xr.DataArray(
        Sxx_db,
        coords={
            'frequency': frequencies,
            'time': spectrogram_times
        },
        dims=['frequency', 'time'],
        attrs={
            'long_name': 'Power spectral density (dB)',
            'units': 'dB',
            'sampling_rate': fs,
            'window': window,
            'nperseg': nperseg,
            'noverlap': noverlap,
            'scaling': scaling,
            'original_dataarray_name': dataarray.name
        }
    )
    
    # Create Dataset with both linear and dB scales
    spectrogram_dataset = xr.Dataset({
        'linear': spectrogram_linear,
        'db': spectrogram_db
    }, attrs={
        'sampling_rate': fs,
        'window': window,
        'nperseg': nperseg,
        'noverlap': noverlap,
        'scaling': scaling,
        'original_dataarray_name': dataarray.name,
        'description': 'Spectrogram with linear and dB scales'
    })
    
    if verbose:
        print(f"  - Spectrogram shape: {Sxx.shape}")
        print(f"  - Frequency range: {frequencies[0]:.1f} to {frequencies[-1]:.1f} Hz")
        print(f"  - Time range: {spectrogram_times[0]:.3f} to {spectrogram_times[-1]:.3f} s")
        print(f"  - Linear range: {Sxx.min():.2e} to {Sxx.max():.2e}")
        print(f"  - dB range: {Sxx_db.min():.1f} to {Sxx_db.max():.1f} dB")
    
    return spectrogram_dataset


def create_stft_da(dataarray, fs=None, nperseg=None, noverlap=None, window='hann', verbose=0):
    """
    Create STFT from an xarray DataArray
    
    Parameters:
    -----------
    dataarray : xarray.DataArray
        Input time series data
    fs : float, optional
        Sampling frequency (Hz). If None, will be automatically calculated from time coordinates
    nperseg : int, optional
        Length of each segment for FFT
    noverlap : int, optional
        Number of points to overlap between segments
    window : str or tuple, optional
        Window function to use
    verbose : int, optional
        Verbosity level
    
    Returns:
    --------
    xarray.Dataset
        Dataset containing 'magnitude' and 'phase' DataArrays with STFT results
    """
    
    # Extract the data and time coordinates
    data = dataarray.values
    time_coord = dataarray.coords[dataarray.dims[0]]
    
    # Auto-detect sampling frequency if not provided
    if fs is None:
        if len(time_coord) > 1:
            dt = float(time_coord[1] - time_coord[0])
            fs = 1.0 / dt
            if verbose:
                print(f"Auto-detected sampling frequency: {fs:.1f} Hz (dt = {dt:.6f} s)")
        else:
            raise ValueError("Cannot auto-detect sampling frequency: need at least 2 time points")
    
    if verbose:
        print(f"Creating STFT for {dataarray.name}")
        print(f"  - Sampling rate: {fs} Hz")
        print(f"  - Data length: {len(dataarray)} points")
        print(f"  - Time range: {time_coord[0]:.3f} to {time_coord[-1]:.3f} s")
    
    if nperseg is None:
        nperseg = min(256, len(data) // 4)
    if noverlap is None:
        noverlap = nperseg // 2
    
    if verbose:
        print(f"  - Window length: {nperseg} points")
        print(f"  - Overlap: {noverlap} points")
        print(f"  - Window function: {window}")
    
    # Compute STFT
    frequencies, times, Zxx = signal.stft(
        data, fs=fs, nperseg=nperseg, noverlap=noverlap, window=window
    )
    
    # Create time coordinates
    stft_times = float(time_coord[0]) + times
    
    # Create DataArray for magnitude (linear scale)
    magnitude_linear = xr.DataArray(
        np.abs(Zxx),
        coords={
            'frequency': frequencies,
            'time': stft_times
        },
        dims=['frequency', 'time'],
        attrs={
            'long_name': 'STFT magnitude (linear)',
            'units': 'V',
            'sampling_rate': fs,
            'window': window,
            'nperseg': nperseg,
            'noverlap': noverlap,
            'original_dataarray_name': dataarray.name
        }
    )
    
    # Create DataArray for magnitude (dB scale)
    magnitude_db = xr.DataArray(
        20 * np.log10(np.abs(Zxx) + 1e-12),  # 20*log10 for magnitude (not power)
        coords={
            'frequency': frequencies,
            'time': stft_times
        },
        dims=['frequency', 'time'],
        attrs={
            'long_name': 'STFT magnitude (dB)',
            'units': 'dB',
            'sampling_rate': fs,
            'window': window,
            'nperseg': nperseg,
            'noverlap': noverlap,
            'original_dataarray_name': dataarray.name
        }
    )
    
    # Create DataArray for phase
    phase_da = xr.DataArray(
        np.angle(Zxx),
        coords={
            'frequency': frequencies,
            'time': stft_times
        },
        dims=['frequency', 'time'],
        attrs={
            'long_name': 'STFT phase',
            'units': 'rad',
            'sampling_rate': fs,
            'window': window,
            'nperseg': nperseg,
            'noverlap': noverlap,
            'original_dataarray_name': dataarray.name
        }
    )
    
    # Create Dataset containing magnitude (linear), magnitude (dB), and phase
    stft_dataset = xr.Dataset({
        'magnitude_linear': magnitude_linear,
        'magnitude_db': magnitude_db,
        'phase': phase_da
    }, attrs={
        'sampling_rate': fs,
        'window': window,
        'nperseg': nperseg,
        'noverlap': noverlap,
        'original_dataarray_name': dataarray.name,
        'description': 'Short-Time Fourier Transform results with linear and dB magnitude scales'
    })
    
    if verbose:
        print(f"  - STFT shape: {Zxx.shape}")
        print(f"  - Frequency range: {frequencies[0]:.1f} to {frequencies[-1]:.1f} Hz")
        print(f"  - Time range: {stft_times[0]:.3f} to {stft_times[-1]:.3f} s")
        print(f"  - Magnitude range (linear): {np.abs(Zxx).min():.2e} to {np.abs(Zxx).max():.2e} V")
        print(f"  - Magnitude range (dB): {20*np.log10(np.abs(Zxx)+1e-12).min():.1f} to {20*np.log10(np.abs(Zxx)+1e-12).max():.1f} dB")
        print(f"  - Phase range: {np.angle(Zxx).min():.2f} to {np.angle(Zxx).max():.2f} rad")
    
    return stft_dataset


def cumulative_integrate(dataarray, dim=None, initial=0, verbose=0):
    """
    Perform cumulative trapezoidal integration on an xarray DataArray
    
    Parameters:
    -----------
    dataarray : xarray.DataArray
        Input data to integrate
    dim : str, optional
        Dimension along which to integrate. If None, uses the first dimension
    initial : float, optional
        Initial value for the cumulative integral (default: 0)
    verbose : int, optional
        Verbosity level
    
    Returns:
    --------
    xarray.DataArray
        DataArray containing the cumulative integral with same coordinates and attributes
    """
    if dim is None:
        dim = dataarray.dims[0]
    
    if verbose:
        print(f"Performing cumulative integration on {dataarray.name} along dimension '{dim}'")
        print(f"  - Initial value: {initial}")
        print(f"  - Data shape: {dataarray.shape}")
        print(f"  - Integration dimension: {dim}")
    
    # Get the coordinate values for the integration dimension
    x_coord = dataarray.coords[dim]
    
    # Perform cumulative trapezoidal integration
    integrated_data = cumulative_trapezoid(dataarray, x=x_coord, initial=initial)
    
    # Create new DataArray with same coordinates, dimensions, and attributes
    result = xr.DataArray(
        integrated_data,
        coords=dataarray.coords,
        dims=dataarray.dims,
        attrs=dataarray.attrs.copy()
    )
    
    # Update attributes to reflect integration
    if 'units' in result.attrs:
        # Try to infer new units (this is a simple heuristic)
        original_units = result.attrs['units']
        if original_units == 'V':
            result.attrs['units'] = 'Vs'
        elif original_units == 'A':
            result.attrs['units'] = 'As'
        elif original_units.endswith('/s'):
            result.attrs['units'] = original_units[:-2]
        else:
            result.attrs['units'] = f"{original_units}*s"
    
    result.attrs['long_name'] = f"Cumulative integral of {dataarray.name or 'data'}"
    result.attrs['integration_method'] = 'cumulative_trapezoid'
    result.attrs['integration_dimension'] = dim
    result.attrs['initial_value'] = initial
    
    if verbose:
        print(f"  - Result shape: {result.shape}")
        print(f"  - Result range: {result.min().values:.2e} to {result.max().values:.2e}")
        if 'units' in result.attrs:
            print(f"  - Units: {result.attrs['units']}")
    
    return result