from golem_toolkit.loading import DAS_systems as DAS
import pytest
import xarray as xr
import matplotlib
# matplotlib.use("Agg", force=True)

I_measurement_resistor = 47
U_divider_factor = 42
time_units = "ms"
calib_shot = 49808  # kolem toho

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

@pytest.mark.network
def test_redpitaya_load_and_plot():
    """Load REDPITAYA data from GOLEM and ensure plotting works headlessly."""
    dev = DAS.REDPITAYA(redp_settings)
    dev.load_data(calib_shot, time_units=time_units, verbose=1)

    assert isinstance(dev.data, xr.Dataset), "Expected xarray.Dataset from REDPITAYA.load_data()"
    dev.plot()
    

@pytest.mark.network
def test_tek64_load_and_plot():
    """Load TEK64 data from GOLEM and ensure plotting works headlessly."""
    dev = DAS.TEK64(tek_settings)
    dev.load_data(calib_shot, time_units=time_units, verbose=1)
    
    assert isinstance(dev.data, xr.Dataset), "Expected xarray.Dataset from TEK64.load_data()"
    dev.plot()