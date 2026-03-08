import io
import os
import yaml

import streamlit as st

import requests

import pandas as pd
import xarray as xr



CWD = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR: str = "config"
URLS_CONFIG: str = "urls.yaml"
CONSTANTS_CONFIG: str = "constants.yaml"
APP_CONFIG: str = "app_defaults.yaml"

@st.cache_data
def _load_config(config_file_path: os.PathLike):
    if not os.path.isfile(config_file_path):
        raise FileNotFoundError(f"No config file found at {config_file_path} path.")

    with open(config_file_path) as file:
        return yaml.safe_load(file)


URLS = _load_config(os.path.join(CWD, os.pardir, CONFIG_DIR, URLS_CONFIG))
CONSTANTS = _load_config(os.path.join(CWD, os.pardir, CONFIG_DIR, CONSTANTS_CONFIG))
APP = _load_config(os.path.join(CWD, os.pardir, CONFIG_DIR, APP_CONFIG))

COIL_IDS = list(range(APP["min_coil_id"], APP["max_coil_id"]+1))



def is_poloidal_coil_channel(key: list[str]) -> bool:
    if key == "Time":
        return False
    if not key[-2:].isnumeric():
        return True
    if int(key[-2:]) <= 15:
        return True
    if int(key[-2:]) > 15:
        return False
    # If none of the above conditions are met, raise an error
    raise ValueError(f"Key '{key}' does not match any poloidal coil channel condition.")


def select_plasma_interval(shot_no: int, data: xr.DataArray) -> xr.DataArray:
    t_plasma_start_url = f"{URLS["base"]}{shot_no}{URLS["t_plasma_start"]}"
    t_plasma_start = float(pd.read_csv(t_plasma_start_url, header=None).values[0,0])

    t_plasma_end_url = f"{URLS["base"]}{shot_no}{URLS["t_plasma_end"]}"
    t_plasma_end = float(pd.read_csv(t_plasma_end_url, header=None).values[0,0])

    data = data.sel(time=slice(t_plasma_start, t_plasma_end))

    return data


def select_poloidal_coil_channels(keys: list[str]) -> list[str]:
    data_keys = []
    for key in keys:
        if is_poloidal_coil_channel(key):
            data_keys.append(key)
    return data_keys


def choose_url_based_on_shot(shot_no: int, url_type: str) -> str:
    if shot_no > 46999 or shot_no == 0:
        return f"{URLS['base']}{shot_no}{URLS[f'{url_type}_47000-now']}"
    else:
        return f"{URLS['base']}{shot_no}{URLS[f'{url_type}_0-46999']}"


def load_MHDring_data(shot_no: int) -> xr.DataArray:
    url = choose_url_based_on_shot(shot_no, "MHDring_data")

    response = requests.get(url)
    if response.status_code != 200:
        raise RuntimeError(f"Error loading MHDring datafrom {url}: {response.status_code}")

    # TODO: config.yaml defalult_deliminer
    if shot_no > 46999:
        try:
            data = pd.read_csv(url, delimiter=",")
        except Exception as e:
            raise RuntimeError(f"Error loading MHDring data for shot {shot_no}: {e}")
    else:
        content = requests.get(url).content
        data = pd.read_csv(io.StringIO(content.decode('iso-8859-1')),
                           sep = '\t',
                           decimal = ',',
                           names = ['Time'] + [f"ch{coil_id - 1}" for coil_id in COIL_IDS])
        # raise RuntimeError(f"Error loading MHDring data for shot {shot_no} from {url}: {e}")
        # data['ch0'] *= -1
        # data['ch8'] *= -1
        # data['ch12'] *= -1

    print(data)

    # Converts to ms
    data["Time"] *= 1_000

    data_keys = select_poloidal_coil_channels(keys=data.keys())
    
    data = xr.DataArray(data=data[data_keys],
                        dims=["time", "channel"],
                        coords={"time": data["Time"],
                                "channel": range(0,15+1)},
                        name="Mirnov_coils")

    return data