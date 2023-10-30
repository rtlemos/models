from typing import Tuple
import pandas as pd
import numpy as np


def get_dynamic_regression_simulated_data(
        num_instants: int,
        num_features: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    """
    Simulates a dataset with one response variable and `num_feature` features,
    for `num_instants` time points
    :param num_instants: number of time points
    :param num_features: number of features
    :return: pandas dataframe with time, response, and features, and dataframe with true coeffs
    """
    if num_instants < 877:
        times = pd.date_range(start='1951-01', end='2023-12', freq='MS')[0:num_instants]
    elif num_instants < 26634:
        times = pd.date_range(start='1951-01', end='2023-12', freq='D')[0:num_instants]
    else:
        times = pd.date_range(start='1951-01', end='2023-12', freq='S')[0:num_instants]

    np.random.seed(0)
    x = np.random.normal(size=[num_instants, num_features])
    t = np.arange(num_instants) * np.pi
    cutp = int(num_features / 2)
    c = np.transpose([
        (1 + 1 / (1 + i) * np.cos(t / (2 * ((1 + i) * num_instants))))
        if i < cutp else
        (-1 + 1 / (1 + i - cutp) * np.sin(t / (2 * ((1 + i - cutp) * num_instants))))
        for i in range(num_features)
    ])

    df = pd.concat(
        [pd.DataFrame({'time': times,
                       'response': 0.5 * np.random.standard_normal(size=num_instants) +
                                   np.sum(x * c, axis=1)}),
         pd.DataFrame(x, columns=['feature_' + str(i) for i in range(num_features)])],
        axis=1)
    coeff_df = pd.concat(
        [pd.DataFrame({'time': times}),
         pd.DataFrame(c, columns=['true_coeff_' + str(i) for i in range(num_features)])],
        axis=1)

    return df, coeff_df
