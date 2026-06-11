# golem-toolkit

GOLEM toolkit.

## Installation

This package is not yet available on PyPI or conda-forge. You need to download or clone the repository locally to install it.

### Using Anaconda

1. **Download or clone the repository** to your local machine.

2. **Navigate to the repository directory**:
   ```bash
   cd golem_toolkit
   ```

3. **(Optional) Create a new conda environment** with Python >=3.9:
   ```bash
   conda create -n golem-toolkit python>=3.9
   ```

4. **Activate your conda environment**:
   ```bash
   conda activate golem-toolkit
   ```

5. **Install the package** from the local directory:
   
   For regular installation:
   ```bash
   pip install .
   ```
   
   For editable/development installation (recommended for development):
   ```bash
   pip install -e .
   ```

6. **Optional: Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

## Active probe IV correction

Module `golem_toolkit.probes` corrects raw shunt current (CH1) and divider output (CH2) to tip voltage and probe current using calibration NetCDF files.

```python
from golem_toolkit.probes import load_iv_calibration, process_iv_signals

calib = load_iv_calibration("active_probe_calib.nc")
ds = process_iv_signals(I_raw, U_raw, calibration=calib, mode="ac", f0_hz=20_000.0)
# Conductance at f0: Re(ds.attrs["I_probe_phasor"] / ds.attrs["U_true_phasor"])
# IV plot: ds["U_true"] vs ds["I_probe"] (waveform keeps harmonics/noise)
```

Documentation: [docs/probes/IV_CORRECTION.md](docs/probes/IV_CORRECTION.md), [docs/probes/NAMING.md](docs/probes/NAMING.md).

Generate calibration from circuit_test: `python active_voltage_divider_test/export_iv_calibration.py`.

---

Licensed under the MIT License – see [LICENSE](LICENSE) for details.
