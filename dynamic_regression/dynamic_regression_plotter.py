import numpy as np
import pandas as pd
import tensorflow as tf
from plotnine import (
    ggplot,
    theme,
    aes,
    geom_line,
    geom_point,
    geom_density,
    geom_boxplot,
    facet_grid,
    xlab,
    ylab,
    geom_qq,
    stat_qq_line,
    geom_abline,
    options
)

from dynamic_regression import DynamicRegression


class DynamicRegressionPlotter:

    def __init__(self, model: DynamicRegression):
        self.model = model

    @staticmethod
    def losses(losses: tf.Tensor) -> ggplot:
        """
        Plots optimizer loss as function of iteration

        :param losses: MAP or VB tensor provided by optimizer
        :return: line plot with x=iteration and y=loss
        """
        losses = losses.numpy()
        if np.min(losses) > 0:
            losses = np.log10(losses)
            yl = "log10(loss)"
        else:
            yl = "loss"
        df = pd.DataFrame({"iteration": range(len(losses)), "loss": losses})
        p = ggplot(df) + aes(x="iteration", y="loss") + geom_line() + ylab(yl)
        return p

    def timeseries_obs_vs_fit(
            self,
            lines: bool = False,
            points: bool = True,
            params: dict = None,
            num_model_samples: int = 30
    ):
        """
        Creates a time series plot of observations (x=time, y=value) and superimposes fit if wanted

        :param lines: represent time series as lines
        :param points: represent time series as points
        :param params: estimated parameters
        :param num_model_samples: number of model samples to be drawn
        :return: ggplot
        """

        df = pd.DataFrame({"instant": self.model.time, "value": tf.squeeze(self.model.obs).numpy()})
        p = ggplot(df) + aes(x="instant", y="value")

        # plotting model samples if desired
        if params is not None and num_model_samples > 0:
            s = self.model.get_simulated_data(params=params, num_samples=num_model_samples).numpy()
            for i in range(num_model_samples):
                dd = pd.DataFrame({"instant": self.model.time, "value": s[i, :, 0]})
                p += geom_line(mapping=aes(x="instant", y="value"), data=dd, color='lightgray')

        if points:
            p += geom_point()
        if lines:
            p += geom_line()
        return p

    def timeseries_regression_coeffs(
            self,
            params: dict
    ) -> ggplot:
        mu_distr = self.model.get_mu_distribution(params['sigma2'], params['tau2'], self.model.obs)
        mu_loc = mu_distr.loc.numpy()
        mu_sd = tf.sqrt(tf.linalg.diag_part(mu_distr.covariance())).numpy()
        df = pd.DataFrame({
            'instant': np.repeat(self.model.time, self.model.num_features),
            'param': self.model.features_names * self.model.num_instants,
            'estimate': mu_loc.flatten()})
        p = ggplot(df) + geom_line(aes(x="instant", y="estimate", group="param", color="param"))
        df2 = df.copy()
        df2['estimate'] = (mu_loc + 2 * mu_sd).flatten()
        p += geom_line(aes(x="instant", y="estimate", group="param"), linetype='dashed',
                       color='lightgray', data=df2)
        df3 = df.copy()
        df3['estimate'] = (mu_loc - 2 * mu_sd).flatten()
        p += geom_line(aes(x="instant", y="estimate", group="param"), linetype='dashed',
                       color='lightgray', data=df3)
        p += theme(legend_position=(0.5, 0.9))
        return p

    def forecast_residuals_qqplot(
            self,
            params: dict
    ) -> ggplot:
        """
        Produces a quantile-quantile plot of standardized one-step-forecast residuals

        :param params: model parameter estimates
        :return: quantile-quantile ggplot of standardized one-step-forecast residuals
        """

        df = pd.DataFrame({'std_residual': self.model.get_forecast_residuals(
            sigma2=params['sigma2'], tau2=params['tau2'], obs=self.model.obs, standardize=True)})

        p = (
                ggplot(df)
                + geom_abline(intercept=0, slope=1, linetype="dashed", color="black")
                + aes(sample="std_residual")
                + geom_qq()
                + stat_qq_line()
        )
        return p