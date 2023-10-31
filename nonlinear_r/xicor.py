"""

** Paper **
Title: A NEW COEFFICIENT OF CORRELATION
Author: SOURAV CHATTERJEE
URL: https://arxiv.org/pdf/1909.10140.pdf

** Python Code **
Author: Ricardo Lemos
Date: Jan 9, 2022
Copyright: Apache 2.0
"""

import numpy as np
import numpy.typing as npt
from scipy.stats import norm


def xicor(x: npt.NDArray[float], y: npt.NDArray[float]) -> dict:
    """
    Computes Sourav Chatterjee's nonlinear correlation coefficient for continuous variables.
    From Chatterjee's (2021) abstract:

    Is it possible to define a coefficient of correlation which is
    (a) as simple as the classical coefficients like Pearson’s correlation or Spearman’s
    correlation, and yet
    (b) consistently estimates some simple and interpretable measure of the degree of dependence
    between the variables, which is 0 if and only if the variables are independent and 1 if and
    only if one is a measurable function of the other, and
    (c) has a simple asymptotic theory under the hypothesis of independence, like the classical
    coefficients?
    This article answers this question in the affirmative, by producing such a coefficient.
    No assumptions are needed on the distributions of the variables.
    There are several coefficients in the literature that converge to 0 if and only if the variables
    are independent, but none that satisfy any of the other properties mentioned above.

    :param x: sample of predictor variable (1D array, float)
    :param y: sample of response variable (1D array, float, same length as x)
    :return: dict with:
        statistic: Chatterjee's nonlinear correlation coefficient value (float)
        pvalue: asymptotic p-value (float), assuming no ties

    Reference:
        [1] Chatterjee S (2021). A new coefficient of correlation. JASA 116:536, 2009-2022, DOI: 10.1080/01621459.2020.1758115

        [2] Chatterjee S, Holmes S (2023). XICOR: Robust and generalized correlation coefficients. https://github.com/spholmes/XICOR, https://CRAN.R-project.org/package=XICOR.
    """

    _check_inputs(x, y)
    rank_y = _get_rank(y[np.argsort(x)])
    anti_rank_y = _get_anti_rank(rank_y)
    numerator = _get_numerator(rank_y)
    denominator = _get_denominator(anti_rank_y)
    xi = 1 - numerator / denominator
    p_value = _nonlinear_p_value(xi, len(x))
    return {'statistic': xi, 'pvalue': p_value}


###############################################
# Auxiliary functions #########################
###############################################

def _check_inputs(x: npt.NDArray[float], y: npt.NDArray[float]) -> None:
    if len(x) != len(y):
        raise ValueError('the two arrays have different lengths: ' +
                         str(len(x)) + ' vs ' + str(len(y)))


def _get_rank(z: npt.NDArray[float]) -> npt.NDArray[int]:
    temp = np.argsort(z)
    ranks = np.empty_like(temp)
    ranks[temp] = 1 + np.arange(len(z))
    return ranks


def _get_anti_rank(rank_y: npt.NDArray[int]) -> npt.NDArray[int]:
    return len(rank_y) - rank_y + 1


def _get_numerator(rank_y: npt.NDArray[int]) -> float:
    return len(rank_y) * np.sum([np.abs(r_next - r) for r_next, r in zip(rank_y[1:], rank_y[:-1])])


def _get_denominator(antirank_y: npt.NDArray[int]) -> float:
    return 2 * np.sum(antirank_y * (len(antirank_y) - antirank_y))


def _nonlinear_p_value(xi: float, n: int) -> float:
    return 1 - norm.cdf(xi * np.sqrt(n * 5 / 2))