import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def _close_matplotlib_figures():
    yield
    plt.close("all")
