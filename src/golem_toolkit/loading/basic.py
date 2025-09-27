import numpy as np
import pandas as pd

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