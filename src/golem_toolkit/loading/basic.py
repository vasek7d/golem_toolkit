import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

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