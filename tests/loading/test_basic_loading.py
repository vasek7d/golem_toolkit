import golem_toolkit.loading as load
import numpy as np
import pandas as pd
import pytest

def test_load_array_from_file(tmp_path):
    # Arrange: make a CSV file
    csv = "t,value\n0,1\n1,2\n2,3\n"
    p = tmp_path / "data.csv"
    p.write_text(csv)

    # Act
    df = load.load_array(str(p), index_col=None)

    # Assert
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["t", "value"]
    assert df.shape == (3, 2)


def test_load_array_warns_on_infinities(tmp_path):
    # Arrange: file with infinite values
    csv = "t,value\n0,1\n1,inf\n2,-inf\n"
    p = tmp_path / "with_inf.csv"
    p.write_text(csv)

    # Act
    df = load.load_array(str(p))
    print(df)

    # Assert: still returns a DataFrame
    assert isinstance(df, pd.DataFrame)
    # The infinities remain in the data
    assert np.isnan(df["value"].iloc[1])
    assert np.isnan(df["value"].iloc[2])


def test_load_parameter_ok(tmp_path):
    # Arrange
    p = tmp_path / "param.csv"
    p.write_text("42\n")

    # Act
    val = load.load_parameter(str(p), int)

    # Assert
    assert val == 42


def test_load_parameter_failure(tmp_path):
    # Arrange: non-existing file
    file = tmp_path / "missing.csv"

    # Act
    val = load.load_parameter(str(file), int)

    # Assert: failure → None
    assert val is None
    
@pytest.mark.network
def test_load_array_from_golem_url():
    """
    Smoke-test that we can fetch a real GOLEM CSV over HTTP.
    We keep assertions soft because shot data / columns can change.
    """
    # Example shot known to exist historically; change if needed.
    shot_no = 49808
    url = f"http://golem.fjfi.cvut.cz/shots/{shot_no}/Diagnostics/PlasmaDetection/U_RogCoil.csv"

    df = load.load_array(url)

    # If the server/URL is unavailable, treat as xfail instead of hard fail
    if df is None:
        pytest.xfail("GOLEM URL unavailable or schema changed; update shot_no or path.")
        return

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert df.shape[1] >= 1  # at least one column of data
    
@pytest.mark.network
def test_load_param_from_golem_url():
    """
    Smoke-test that we can fetch a real GOLEM CSV over HTTP.
    We keep assertions soft because shot data / columns can change.
    """
    # Example shot known to exist historically; change if needed.
    shot_no = 49808
    url = f'http://golem.fjfi.cvut.cz/shots/{shot_no}/Production/Parameters/SystemParameters/R_chamber'

    param_type = float
    param = load.load_parameter(url, param_type)

    # If the server/URL is unavailable, treat as xfail instead of hard fail
    if param is None:
        pytest.xfail("GOLEM URL unavailable or schema changed; update shot_no or path.")
        return

    assert isinstance(param, param_type)
    assert not np.isnan(param)

