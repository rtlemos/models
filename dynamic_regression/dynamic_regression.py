"""
Build a dynamic linear regression model using TFP:

\[
Y_t = \bm{x}'_t \bm{\mu}_t + \epsilon_t,
\bm{\mu}_t = \bm{\mu}_{t-1} + \bm{\nu}_t
\epsilon_t \sim N[0, 1 / \sigma^2]
\bm{\nu}_t \sim N[\bm{0}, 1 / \tau^2 \bm{I}]
\bm{\mu}_0 \sim N[\bm{0}, \bm{I}]
\sigma^2 \sim InvGamma[sigma2_concentration, sigma2_rate]
\tau^2 \sim InvGamma[tau2_concentration, tau2_rate]
\]

where
- we we place m, preferably standardized features in the m*1 vector $\bm{x}_t$
- the observation error $\epsilon_t$ is iid Normal
- the m regression coefficients $\bm{\mu}_t$ evolve as a random walk
- the m random walk shocks $\bm{\nu}_t$ are iid Normal
- before observing any data, we assume the regression coefficients are iid Normal[0,1]
"""

from typing import List, Tuple

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb

import pandas as pd
import numpy as np
import properscoring as ps

Root = tfd.JointDistributionCoroutine.Root


class DynamicRegression:

    def __init__(self):
        self.num_instants = None
        self.num_features = None

        # options
        self.sigma2_concentration = None
        self.sigma2_rate = None
        self.tau2_concentration = None
        self.tau2_rate = None
        self.object_seed = None
        self.latent_process_initial_scale = None

        # features
        self.features = None
        self.features_names = None

        # data
        self.time = None
        self.obs = None

        # metadata
        self.metadata = None

        # model
        self.model = None

    def methods(self) -> List[str]:
        """
        Lists all the public methods available in this class

        :return: list of public methods
        """
        return list(
            np.sort(
                [
                    attr
                    for attr in dir(self.__class__)
                    if callable(getattr(self.__class__, attr))
                       and attr.startswith("__") is False
                       and attr.startswith("_") is False
                       and attr.startswith("methods") is False
                ]
            )
        )

    def set_metadata(self, metadata: dict) -> None:
        """
        Stores metadata in the object

        :param metadata: dict with details about the data fed to this object
        """
        self.metadata = metadata

    def set_options(self, options: dict = None) -> None:
        """
        Sets up the options for model fitting

        :param options: dictionary of model fitting options
        :return: None
        """
        if options is None:
            print("dictionary `options` not provided; returning list of keys")
            print(
                [
                    "sigma2_mean",
                    "sigma2_var",
                    "tau2_mean",
                    "tau2_var",
                    "latent_process_initial_scale",
                    "object_seed"
                ]
            )
            return

        self.sigma2_concentration = self._to_float(
            options['sigma2_mean'] ** 2 / options['sigma2_var'])
        self.sigma2_rate = self._to_float(options['sigma2_mean'] / options['sigma2_var'])
        self.tau2_concentration = self._to_float(options['tau2_mean'] ** 2 / options['tau2_var'])
        self.tau2_rate = self._to_float(options['tau2_mean'] / options['tau2_var'])
        self.latent_process_initial_scale = self._to_float(options["latent_process_initial_scale"])
        self.object_seed = tfp.util.SeedStream(options["object_seed"], salt="dyn_regr_salt")

    def set_model(
            self,
            df: pd.DataFrame,
            response_name: str,
            features_names: List[str],
            time_name: str
    ) -> None:
        """
        Checks if the specifications look OK and builds the model
        """
        self._set_obs(df=df, response_name=response_name)
        self._set_features(df=df, features_names=features_names)
        self._set_dims(df=df, features_names=features_names)
        self._set_time(df=df, time_name=time_name)
        self._check_model()
        self.model = tfd.JointDistributionCoroutine(self._prior_generator)

    def get_sample_from_prior(
            self,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Samples (once) from the prior distribution of all unknowns in the model
        :return: tuple of tensors (sigma2, tau2, obs)
        """
        return self.model.sample(seed=self.object_seed)

    def fit_map(
        self,
        learning_rate: float = 0.1,
        num_steps: int = 1000,
        jit_compile: bool = False,
        ini: dict = None,
    ) -> Tuple[dict, tf.Tensor]:
        """
        Finds the Maximum A Posteriori (MAP) estimate for all the unknowns in the model

        :param learning_rate: optimizer (Adam) learning rate
        :param num_steps: number of optimizer iterations
        :param jit_compile: use just-in-time compilation?
        :param ini: dictionary with initial values for the unknowns
        :return: dictionary with MAP estimates for the unknowns, and tensor with losses
        """
        s2, t2 = self._get_first_guesses(ini)
        sigma2 = tfp.util.TransformedVariable(
            s2, bijector=tfb.Softplus(), name="map_sigma2"
        )
        tau2 = tfp.util.TransformedVariable(
            t2, bijector=tfb.Softplus(), name="map_tau2"
        )

        losses = tfp.math.minimize(
            loss_fn=lambda: -self.model.log_prob((sigma2, tau2, self.obs)),
            optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
            num_steps=num_steps,
            jit_compile=jit_compile,
        )
        return {
            "sigma2": sigma2,
            "tau2": tau2,
        }, losses

    def fit_vb(
            self,
            learning_rate: float = 0.1,
            num_steps: int = 1000,
            jit_compile: bool = False,
            bayes_map: dict = None,
            previous_vb_fit: tfd.JointDistributionCoroutine = None
    ) -> Tuple[tfd.JointDistributionCoroutine, tf.Tensor]:
        """
        Finds Variational Bayesian (VB) approximation to posterior distribution of model parameters

        :param learning_rate: optimizer (Adam) learning rate
        :param num_steps: number of optimizer iterations
        :param jit_compile: use just-in-time compilation?
        :param bayes_map: dictionary with initial values for the unknowns
        :param previous_vb_fit: output of VB fit to previous dataset
        :return: dictionary with VB estimates for the unknowns, and tensor with losses
        """

        if previous_vb_fit is not None:
            self._set_prior_from_vb_fit(previous_vb_fit=previous_vb_fit)
            ini = self._get_initial_values_from_vb_fit(previous_vb_fit=previous_vb_fit)
            q = self._get_surrogate_approximation(user_guesses=ini)
        else:
            q = self._get_surrogate_approximation(user_guesses=bayes_map)

        losses = self._variational_bayes(
            model=self.model,
            obs=self.obs,
            q=q,
            num_steps=num_steps,
            learning_rate=learning_rate,
            jit_compile=jit_compile,
            seed=self.object_seed,
        )
        return q, losses

    @staticmethod
    def get_vb_parameter_sample(
            q: tfd.JointDistributionCoroutine,
            num_samples: int = 1
    ) -> dict:
        """
        Takes the output of a Variational Bayes fit and samples from it

        :param q: VB output
        :param num_samples: number of samples from each parameter
        :return: dict with parameter samples
        """
        samples = q.sample(num_samples)
        return {
            'sigma2': samples[0],
            'tau2': samples[1]
        }

    def get_posterior_predictive_distribution(
            self, sigma2, tau2, obs
    ) -> tfd.Normal:
        """
        Provides the posterior predictive distribution for a single set of parameter values

        :param sigma2: (tensor) scalar measurement error precision
        :param tau2: (tensor) scalar evolution precision
        :param obs: tensor of observations
        :return: predictive distribution of observations, for all instants and sites
        """
        mu = self.get_mu_distribution(sigma2=sigma2, tau2=tau2, obs=obs).sample()
        predictive_mean = tf.squeeze(
            tf.matmul(self.features[..., :, :], mu[..., :, tf.newaxis]), -1)
        predictive_scale = tf.math.reciprocal(tf.math.sqrt(sigma2))
        post_predictive_distribution = tfd.Normal(
            loc=predictive_mean, scale=predictive_scale
        )
        return post_predictive_distribution

    def get_one_step_forecast_moments(
            self, sigma2, tau2, obs
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        ssm = self._get_state_space_model(sigma2, tau2)
        _, _, _, _, _, forecast_m, forecast_c = ssm.forward_filter(obs)
        return forecast_m, forecast_c

    def get_smoothed_mu_moments(self, sigma2, tau2, obs) -> Tuple[tf.Tensor, tf.Tensor]:
        ssm = self._get_state_space_model(sigma2, tau2)
        _, filtered_m, filtered_c, predicted_m, predicted_c, _, _ = ssm.forward_filter(obs)
        post_m, post_c = ssm.backward_smoothing_pass(
            filtered_m, filtered_c, predicted_m, predicted_c
        )
        return post_m, post_c

    def get_mu_distribution(self, sigma2, tau2, obs) -> tfd.MultivariateNormalTriL:
        post_m, post_c = self.get_smoothed_mu_moments(sigma2, tau2, obs)
        return tfd.MultivariateNormalTriL(loc=post_m, scale_tril=tf.linalg.cholesky(post_c))

    def get_forecast_residuals(self, sigma2, tau2, obs, standardize: bool):
        f_loc, f_scale = self.get_one_step_forecast_moments(
            sigma2, tau2, obs
        )
        residuals = tf.squeeze(obs - f_loc, -1)
        if standardize:
            residuals = residuals / tf.math.sqrt(f_scale[:, 0, 0])
        return residuals

    def get_simulated_data(
            self, params, num_samples: int
    ) -> tf.Tensor:
        """
        Draws sample(s) from the posterior predictive distribution, for all instants

        :param params: dictionary of point estimates for all model unknowns
        :param num_samples: number of samples
        :return: tensor with reconstructions [num_samples, num_instants]
        """
        random_draw = self.get_posterior_predictive_distribution(
            params["sigma2"],
            params["tau2"],
            self.obs,
        ).sample(num_samples)
        return random_draw

    def get_diagnostics(
            self,
            params: dict,
            time_endpoints: List[str] = None,
            model_name: str = ''
    ) -> dict:
        """
        Returns 1-step forecast diagnostics
        :param params: model parameter estimates
        :param time_endpoints: start and end timepoints for diagnostic calculation
        :param model_name: name of model or diagnostic analysis
        :return: dict with root mean squared error, mean absolute error, mean bias, fraction
                 of errors > 2 sdevs, and continuous rank probability score
        """
        if time_endpoints is None:
            st, en = [0, self.num_instants]
        else:
            st, en = self._find_endpoint_indices(time_endpoints)
        residuals = tf.squeeze(self.get_forecast_residuals(sigma2=params['sigma2'],
                                                           tau2=params['tau2'],
                                                           obs=self.obs,
                                                           standardize=False)).numpy()[st:en]
        std_residuals = tf.squeeze(self.get_forecast_residuals(sigma2=params['sigma2'],
                                                               tau2=params['tau2'],
                                                               obs=self.obs,
                                                               standardize=True)).numpy()[st:en]
        f_loc, f_scale = self.get_one_step_forecast_moments(
            params['sigma2'], params['tau2'], self.obs)
        crps = ps.crps_gaussian(x=self.obs.numpy()[st:en,0],
                                mu=tf.squeeze(f_loc).numpy()[st:en],
                                sig=tf.sqrt(tf.squeeze(f_scale)).numpy()[st:en])
        diagnostics = {
            'name': model_name,
            'rmse': np.sqrt(np.mean(np.square(residuals))),
            'mae': np.mean(np.abs(residuals)),
            'bias': np.mean(residuals),
            'frac_errors_gt_2sd': np.mean(np.abs(std_residuals) > 2),
            'crps': np.mean(crps)}
        return diagnostics

    ################################################################################################
    # Auxiliary (private) methods
    ################################################################################################

    def _set_obs(
            self,
            df: pd.DataFrame,
            response_name: str = 'value'
    ) -> None:
        """
        Sets the observations that will be fed to the model

        :param df: table with (at least) column `response_name`
        :param response_name: name of response variable column
        :return: None
        """

        self.obs = tf.convert_to_tensor(df[response_name], dtype=tf.float32)[:, tf.newaxis]

    def _set_time(
            self,
            df: pd.DataFrame,
            time_name: str = 'time'
    ) -> None:
        """
        Sets the timestamps that will be fed to the model

        :param df: table with (at least) column `response_name`
        :param time_name: name of column with timestamps
        :return: None
        """

        self.time = np.array(df[time_name])

    def _set_features(
            self,
            df: pd.DataFrame,
            features_names: List[str]
    ) -> None:
        """
        Sets up the features (aka predictors) that enter the model

        :param df: table with columns `features_names`
        :param features_names: names of features in df
        :return: None
        """

        self.features = self._to_float(df[features_names])[:, tf.newaxis, :]
        self.features_names = features_names

    def _set_dims(self, df: pd.DataFrame, features_names: List[str]) -> None:
        """
        Sets up the dimensions of the various components of the model

        :param df: table with columns `features_names`
        :param features_names: names of features in df
        :return: None
        """
        self.num_instants = int(df.shape[0])
        self.num_features = int(len(features_names))

    def _check_model(self) -> None:
        """
        Assesses the consistency of model specifications

        :return: None
        """

        tf.debugging.assert_equal(self.num_instants, self.features.shape[0])
        tf.debugging.assert_equal(self.num_features, self.features.shape[2])

    def _get_state_space_model(self, sigma2, tau2):
        """
        Constructs the distribution for the observations, given fixed parameter values

        :param sigma2: (tensor) scalar measurement error precision
        :param tau2: (tensor) scalar evolution error precision
        :return: normal distribution object
        """
        return tfd.LinearGaussianStateSpaceModel(
            num_timesteps=self.num_instants,
            transition_matrix=tf.linalg.LinearOperatorIdentity(self.num_features),
            transition_noise=tfd.MultivariateNormalDiag(
                scale_diag= tf.ones([self.num_features]) / tf.sqrt(tau2)
            ),
            observation_matrix=lambda t: tf.linalg.LinearOperatorFullMatrix(self.features[t, :, :]),
            observation_noise=tfd.MultivariateNormalDiag(
                loc=0, scale_diag=tf.math.reciprocal(tf.sqrt(sigma2)) * tf.ones([1])),
            initial_state_prior=tfd.MultivariateNormalDiag(
                scale_diag=self.latent_process_initial_scale * tf.ones([self.num_features])
            ),
            # experimental_parallelize=True,  # raises all sorts of warnings
            # name="fitted_obs"
        )

    def _get_prior_distribution(self, name: str):

        prior_distr = {
            "sigma2": tfd.Gamma(concentration=self.sigma2_concentration,
                                rate=self.sigma2_rate, name="sigma2"),
            "tau2": tfd.Gamma(concentration=self.tau2_concentration,
                              rate=self.tau2_rate, name="tau2"),
        }
        return prior_distr[name]

    def _prior_generator(self) -> None:
        """
        Generates the prior distributions for the unknowns in the model

        :return: None (this function yields generators)
        """

        # prior for measurement error precision (sigma^2)
        sigma2 = yield Root(self._get_prior_distribution("sigma2"))

        # prior for evolution error precision (sigma^2)
        tau2 = yield Root(self._get_prior_distribution("tau2"))

        yield self._get_state_space_model(sigma2, tau2)

    def _get_first_guesses(
            self,
            user_guesses: dict
    ) -> Tuple[tf.Tensor, tf.Tensor]:

        """
        Generates first guesses for model parameters, based on the prior distribution or user input

        :param user_guesses: dictionary with user-defined first-guesses
        :return: tuple of 1st guesses (sigma2, tau2) to initialize optimizer
        """

        s2, t2, _ = self.get_sample_from_prior()

        def get_user_value(parameter_name, default_value):
            return (
                user_guesses[parameter_name]
                if parameter_name in user_guesses
                else default_value
            )

        if type(user_guesses) is dict:
            s2 = get_user_value("sigma2", s2)
            t2 = get_user_value("tau2", t2)
        return s2, t2

    def _get_surrogate_moments(self, user_guesses: dict) -> dict:
        """
        Produces mean and variance estimates for the Variational Bayesian approximate distribution

        :param user_guesses: user-defined first-guesses for the optimizer to run
        :return: dictionary of means and variances
        """

        s2, t2 = self._get_first_guesses(user_guesses)

        def get_posterior_mean(param_name, initial_value):
            return tf.Variable(initial_value, name=param_name + "_mean")

        def get_posterior_scale(param_name, initial_value):
            return tfp.util.TransformedVariable(
                0.1467744 * tf.ones_like(initial_value),
                bijector=tfb.Softplus(),
                name=param_name + "_scale",
            )

        moments = {}
        for name, ini_value in zip(["sigma2", "tau2"], [s2, t2]):
            moments.update(
                {
                    name + "_mean": get_posterior_mean(name, ini_value),
                    name + "_scale": get_posterior_scale(name, ini_value),
                }
            )
        moments.update(
            {
                "sigma2_concentration": tfp.util.TransformedVariable(
                    s2, bijector=tfb.Softplus(), name="sigma2_concentration"
                ),
                "sigma2_rate": tfp.util.TransformedVariable(
                    self._to_float(1.0), bijector=tfb.Softplus(), name="sigma2_rate"
                ),
                "tau2_concentration": tfp.util.TransformedVariable(
                    t2, bijector=tfb.Softplus(), name="tau2_concentration"
                ),
                "tau2_rate": tfp.util.TransformedVariable(
                    self._to_float(1.0), bijector=tfb.Softplus(), name="tau2_rate"
                ),
            }
        )
        return moments

    def _get_surrogate_approximation(self, user_guesses: dict = None):
        """
        Generates the multivariate VB approximate distribution to the posterior

        :param user_guesses: user-defined first guesses to the model's unknowns
        :return: multivariate distribution object whose parameters can be optimized
        """

        moments = self._get_surrogate_moments(user_guesses)

        def get_surrogate_distribution(name):
            return tfd.Independent(
                tfd.Normal(loc=moments[name + "_mean"], scale=moments[name + "_scale"]),
                reinterpreted_batch_ndims=len(moments[name + "_mean"].shape),
                name="q_" + name,
            )

        def surrogate_generator():
            yield Root(
                tfd.Gamma(
                    concentration=moments["sigma2_concentration"],
                    rate=moments["sigma2_rate"],
                    name="q_sigma2",
                )
            )
            yield Root(
                tfd.Gamma(
                    concentration=moments["tau2_concentration"],
                    rate=moments["tau2_rate"],
                    name="q_tau2",
                )
            )

        q = tfd.JointDistributionCoroutine(surrogate_generator)
        return q

    def _set_prior_from_vb_fit(self, previous_vb_fit: tfd.JointDistributionCoroutine):
        """
        Sets the prior distribution function based on a Variational Bayesian posterior

        :param previous_vb_fit: object from a VB fit to a previous dataset
        :return: None (method _get_prior_distribution is modified)
        """

        trainable_vars = previous_vb_fit.trainable_variables
        num_vars = len(trainable_vars)
        var_param0 = {trainable_vars[i].name.split("_")[0]: trainable_vars[i].numpy()
                      for i in range(0, num_vars, 2)}
        var_param1 = {trainable_vars[i].name.split("_")[0]: trainable_vars[i].numpy()
                      for i in range(1, num_vars + 1, 2)}
        self._get_prior_distribution = lambda name: (
            # Prior for sigma2 and tau2
            tfd.Gamma(
                concentration=tfb.Softplus().forward(var_param0[name]),
                rate=tfb.Softplus().forward(var_param1[name]),
                name=name
            )
        )

    @staticmethod
    def _get_initial_values_from_vb_fit(
            previous_vb_fit: tfd.JointDistributionCoroutine
    ) -> dict:

        var_names = ["sigma2", "tau2"]
        vb_sample = previous_vb_fit.sample()
        ini = {v: x.numpy() for v, x in zip(var_names, vb_sample)}
        return ini

    @staticmethod
    def _variational_bayes(
            model, obs, q, num_steps=1000, learning_rate=0.05, jit_compile=False, seed=1
    ):
        """
        Minimizes the KL-divergence between the true posterior distribution of model parameters
        and the VB approximation

        :param model: object
        :param obs: tensor of observations
        :param q: VB approximate distribution object
        :param num_steps: number of optimizer (Adam) iterations
        :param learning_rate: optimizer learning rate
        :param jit_compile: use just-in-time compilation?
        :param seed: pseudo random number generator seed
        :return: tensor of losses (as a side effect, object `q` is modified)
        """
        losses = tfp.vi.fit_surrogate_posterior(
            lambda sigma2, tau2: model.log_prob((sigma2, tau2, obs)),
            surrogate_posterior=q,
            optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
            num_steps=num_steps,
            jit_compile=jit_compile,
            seed=seed,
        )
        return losses

    def _find_endpoint_indices(self, time_endpoints: List[str]) -> List[int]:
        if isinstance(time_endpoints[0], str):
            time_endpoints = [np.datetime64(x) for x in time_endpoints]
        nearest_idx = [np.argmin(abs(t - self.time)) for t in time_endpoints]
        return [nearest_idx[0], nearest_idx[1] + 1]

    @staticmethod
    def _to_int(x):
        return tf.cast(tf.constant(x), dtype=tf.int32)

    @staticmethod
    def _to_float(x):
        return tf.cast(tf.constant(x), dtype=tf.float32)

    @staticmethod
    def _to_bool(x):
        return tf.cast(tf.constant(x), dtype=tf.bool)



