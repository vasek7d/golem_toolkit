import numpy as np
# import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from golem_toolkit.loading import basic as load

def load_DAS(shot_no, data_url, das_settings, n_channels, names=None, skiprows=0, time_units="s", verbose=0):
    # data_url = f"http://golem.fjfi.cvut.cz/shotdir/{shot_no}/Devices/ITs/SignalLab-a/data.csv"

    if names is None:
        names = ["t"] + [f"CH{i}" for i in range(1, n_channels+1)]

    # print("names", names)
    pd_data = load.load_array(data_url, names=names,
                         index_col='t', verbose=verbose, skiprows=skiprows)
    
    if pd_data is None:
        return None
    else:
        data = pd_data.to_xarray()
    
        DAS = xr.Dataset()
    
        for channel, settings in das_settings.items():
            DAS[settings['var_name']] = data.get(
                channel)*settings['scaling_factor']
            DAS[settings['var_name']].attrs = settings['attrs']
        DAS.attrs = {"shot_no": shot_no}
    
        if time_units == "s":
            factor = 1
        elif time_units == "ms":
            factor = 1e3
        else:
            raise ValueError("time_units has to be either 's' or 'ms")
    
        DAS.coords["t"] = DAS.t * factor
        DAS.t.attrs["units"] = time_units
    
        # # mask = np.isinf(DC['I'].values)
        # # t_inf = DC.t.values[mask]
        # # print(t_inf)
        # # print(DC.sel(t=t_inf))
        return DAS

def plot_DAS(DAS_dataset, DAS_name=None, figsize=None):
    # Number of panels = number of data variables (exclude coords)
    try:
        n = len(DAS_dataset.data_vars)
        items_iter = DAS_dataset.data_vars.items()
    except AttributeError:
        # Fallback for dict-like input
        n = len(DAS_dataset)
        items_iter = DAS_dataset.items()

    if n == 0:
        raise ValueError("DAS_dataset has no data variables to plot.")

    if figsize:
        set_figsize = figsize
    else:
        set_figsize = (8, n*3)

    # Always return an array of axes; we'll flatten it
    fig, ax = plt.subplots(
        n, 1,
        figsize=set_figsize,
        dpi=300,
        tight_layout=True,
        sharex=True,
        squeeze=False,   # <— key: never return a scalar Axes
    )
    axes = ax.ravel()    # 1D array of Axes, works for n == 1 too

    supt = f"  #{DAS_dataset.attrs.get('shot_no')}"
    if DAS_name:
        supt += f" (DAS = {DAS_name})"
    fig.suptitle(supt)

    for axi, (var, da) in zip(axes, items_iter):
        da.plot(ax=axi)
        axi.set_title(f"probe {da.attrs.get('probe', var)}")
        axi.grid(True)

        # Example of axis formatting if you need it:
        # formatter = FuncFormatter(lambda x, pos: f"{x*1e3:.1f}")
        # axi.xaxis.set_major_formatter(formatter)
        # axi.set_xlabel("t [ms]")

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

        return self.data

    def plot(self):
        try:
            data = getattr(self, "data")
        except Exception as e:
            print(f"The data has not been loaded yet! Error: {e}")
        plot_DAS(self.data, self.DAS_name)


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

        return self.data

    def plot(self):
        try:
            data = getattr(self, "data")
        except Exception as e:
            print(f"The data has not been loaded yet! \nError: {e}")

        plot_DAS(self.data, self.DAS_name)
        
        
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

    RP = REDPITAYA(redp_settings)
    TEK = TEK64(tek_settings)
    
    calib_shot = 49808  # kolem toho
    time_units = "ms"

    RP.load_data(calib_shot, time_units=time_units, verbose=1)
    TEK.load_data(calib_shot, time_units=time_units, verbose=1)

    RP.plot()
    TEK.plot()

    