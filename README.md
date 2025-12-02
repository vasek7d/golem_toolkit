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

3. **Create a new conda environment** with Python >=3.9:
   ```bash
   conda create -n golem-toolkit python>=3.9
   ```

4. **Activate the environment**:
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

---

Licensed under the MIT License – see [LICENSE](LICENSE) for details.
