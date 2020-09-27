import logging
from .reproduction_number import Change_point
import numpy as np
from scipy.stats import gamma, nbinom
from scipy.special import binom
import pandas as pd
import datetime
import tensorflow as tf
import pymc4 as pm
import arviz as az

log = logging.getLogger(__name__)


def test_data(**in_params):
    """
    Generates a test dataset with one change point
    2 countries
    4 age groups
    R_0=2.3
    and one changepoint to R_1=1.0

    there is some noise for each country and age group.

    Parameters
    ----------
    N:
    R_0:
    I_0:
    d_cp:
    t_max:
    num_nans:
    """

    # ------------------------------------------------------------------------------ #
    # Set params for the test dataset
    # ------------------------------------------------------------------------------ #
    params = {
        # population size per country and age group
        "N": np.array([[1e15, 1e15, 1e15, 1e15], [1e15, 1e15, 1e15, 1e15]]),
        # Reproduction number at t=0 per country and age group
        "R_0": np.array([[2.31, 2.32, 2.33, 2.34], [2.31, 2.32, 2.33, 2.34]]),
        # Initial infected
        "I_0": np.array([[10, 10, 10, 10], [10, 10, 10, 10]], dtype="float64"),
        # Change point date/index
        "d_cp": np.array([[15, 16, 18, 20], [15, 16, 18, 20]]),
        # Length of the change point
        "l_cp": 5.2,
        # Alpha value of the change point
        "alpha_cp": np.array([[0.73, 0.72, 0.74, 0.75], [0.73, 0.72, 0.74, 0.75]]),
        # Number of timesteps
        "t_max": 80,
        # Number of Nans before data (cuts t_max)
        "num_nans": 10,
        # Generation interval
        "g": 4.0,
    }

    # Check in_params
    for key, value in in_params.items():
        if key in params:
            params[key] = value
        else:
            log.warning(f"Input parameter '{key}' not known!")

    # ------------------------------------------------------------------------------ #
    # Start
    # ------------------------------------------------------------------------------ #
    # Create time array
    t = np.arange(start=0, stop=params["t_max"])
    t = np.stack([t] * 4)
    t = np.stack([t] * 2)

    # Create change point
    cp = Change_point(params["alpha_cp"], 1, params["l_cp"], params["d_cp"])

    # Calculate R_t
    R_0 = np.expand_dims(params["R_0"], axis=-1)
    alpha = np.expand_dims(params["alpha_cp"], axis=-1)
    R_t = R_0 * np.exp(-alpha * cp.get_gamma(t))

    def f(t, I_t, R_t, S_t, N):
        """
        Function for a simple SI model

        Parameters
        ----------
        t: number
            current timestep
        I_t: array
            I for every age group
        R_t: array 2d
            Reproduction matrix
        """
        f = S_t / N
        new = f * R_t * I_t
        return new, S_t - new

    # ------------------------------------------------------------------------------ #
    # Iteration
    # ------------------------------------------------------------------------------ #

    # Initial values
    S_t = params["N"]

    I_t = [params["I_0"]]
    for i in range(0, 15):
        I_t.insert(0, I_t[0] / (1 + params["R_0"] / params["g"]))

    # Gamma kernel
    gamma_pdf = gamma.pdf(np.arange(0, 15), params["g"])
    # Normalize
    gamma_pdf = gamma_pdf / np.linalg.norm(gamma_pdf, 1)

    # Convolution
    for t_i in range(15, params["t_max"] + 15):
        I_n = np.zeros(params["I_0"].shape)
        for tau in range(15):
            I_n += I_t[t_i - tau] * gamma_pdf[tau]

        # Calc new
        I_n, S_t = f(t_i, I_n, R_t[:, :, t_i - 15], S_t, params["N"])

        I_t.append(I_n)
    I_t = np.array(I_t[16:])
    # ------------------------------------------------------------------------------ #
    # Data preparation (pandas)
    # ------------------------------------------------------------------------------ #
    dates = pd.date_range(
        start=datetime.datetime(2020, 1, 3),
        end=datetime.datetime(2020, 1, 3)
        + datetime.timedelta(days=params["t_max"] - 1),
    )

    # New cases
    df_new_cases = pd.DataFrame()
    df_new_cases["date"] = dates
    for num in [0, 1, 2, 3]:
        df_new_cases[("country_1", f"age_group_{num}")] = I_t[:, 0, num]
    for num in [0, 1, 2, 3]:
        df_new_cases[("country_2", f"age_group_{num}")] = I_t[:, 1, num]
    df_new_cases = df_new_cases.set_index("date")
    df_new_cases.columns = pd.MultiIndex.from_tuples(
        df_new_cases.columns, names=["country", "age_group"]
    )
    # Add nans to the first 10 values
    date_nans = pd.date_range(
        start=datetime.datetime(2020, 1, 3) - datetime.timedelta(days=10),
        end=datetime.datetime(2020, 1, 3),
    )
    for d in date_nans:
        df_new_cases.loc[d] = list([np.nan] * 8)

    # R_t
    df_R_t = pd.DataFrame()
    df_R_t["date"] = dates
    for num in [0, 1, 2, 3]:
        df_R_t[("country_1", f"age_group_{num}")] = R_t[0, num, :]
    for num in [0, 1, 2, 3]:
        df_R_t[("country_2", f"age_group_{num}")] = R_t[1, num, :]
    df_R_t = df_R_t.set_index("date")
    df_R_t.columns = pd.MultiIndex.from_tuples(
        df_R_t.columns, names=["country", "age_group"]
    )

    # Interventions
    df_interv = pd.DataFrame()
    df_interv["date"] = dates
    for c in range(2):
        # iter over countries
        a = np.zeros(np.mean(params["d_cp"][c, :], dtype="int32"))
        b = np.ones(params["t_max"] - a.shape[0])
        interv = np.concatenate((a, b))
        df_interv[(f"country_{c+1}", "intervention_1")] = interv

    df_interv = df_interv.set_index("date")
    df_interv.columns = pd.MultiIndex.from_tuples(
        df_interv.columns, names=["country", "intervention"]
    )

    return df_new_cases.sort_index(), df_R_t, df_interv


def test_data2(modelParams, **in_params):
    # ------------------------------------------------------------------------------ #
    # Set params for the test dataset
    # ------------------------------------------------------------------------------ #
    len_gen_interv_kernel = 12
    num_interventions = 2
    params = {
        # population size per country and age group
        "N": np.array([[1e15, 1e15, 1e15, 1e15], [1e15, 1e15, 1e15, 1e15]]),
        # Reproduction number at t=0 per country and age group
        "R_0": np.array([[2.31, 2.32, 2.33, 2.34], [2.31, 2.32, 2.33, 2.34]]),
        # Initial infected
        "I_0_diff_base": np.array([[[0, 1, 0, -1], [1, -1, 0, 0]]]),
        "I_0_diff_add": np.zeros(
            (
                len_gen_interv_kernel - 1,
                modelParams.num_countries,
                modelParams.num_age_groups,
            )
        ),
        # Change point date/index
        "delta_d_i": np.zeros((modelParams.num_interventions, 1, 1)),
        "delta_d_c": np.zeros((1, modelParams.num_countries, 1)),
        "sigma_d_interv": 0.3,
        "sigma_d_country": 0.3,
        # Length of the change point
        "l_i_sign": 4
        * np.ones(
            (
                modelParams.num_interventions,
                modelParams.num_countries,
                modelParams.num_age_groups,
            )
        ),
        # Alpha value of the change point
        "alpha_i_c_a": np.array(
            [
                [[0.73, 0.72, 0.74, 0.75], [0.73, 0.72, 0.74, 0.75]],
                [[0.73, 0.52, 0.54, 0.55], [0.53, 0.52, 0.54, 0.55]],
            ]
        ),
        "C": np.array(
            [
                [1, 0.1, 0.1, 0.1],
                [0.1, 1, 0.1, 0.1],
                [0.1, 0.1, 1, 0.1],
                [0.1, 0.1, 0.1, 1],
            ]
        ),
        # Number of timesteps
        # Number of Nans before data (cuts t_max)
        # Generation interval
        "g": 4.0,
    }

    model_name = "test_model"

    trace = pm.sample_posterior_predictive(
        test_model(modelParams),
        az.from_dict(posterior={f"{model_name}/b": np.array([1.0])}),
        var_names=(f"{test_model}/like", "test_model/R_t"),
        use_auto_batching=False,
    )

    new_cases = trace.posterior_predictive["test_model/like"]


def save_data(path, **params):
    """
        Main entry point to generate test data. Passes params to generate function
        and adds random noise to new cases.

        Creates folder structure for two test countries and saves the generated
        data to csv files.
    """
    new_cases, R_t, interv = test_data(**params)

    new_cases = _random_noise(new_cases, 0.00001)

    # Save new_Cases
    new_cases.xs("country_1", 1).to_csv(
        path + "/test_country_1/new_cases.csv", date_format="%d.%m.%y"
    )
    new_cases.xs("country_2", 1).to_csv(
        path + "/test_country_2/new_cases.csv", date_format="%d.%m.%y"
    )

    # Save interventions
    interv.xs("country_1", 1).to_csv(
        path + "/test_country_1/interventions.csv", date_format="%d.%m.%y"
    )
    interv.xs("country_2", 1).to_csv(
        path + "/test_country_2/interventions.csv", date_format="%d.%m.%y"
    )

    # Save R_t
    R_t.xs("country_1", 1).to_csv(
        path + "/test_country_1/reproduction_number.csv", date_format="%d.%m.%y"
    )
    R_t.xs("country_2", 1).to_csv(
        path + "/test_country_2/reproduction_number.csv", date_format="%d.%m.%y"
    )


def _random_noise(df, noise_factor):
    r"""
    Generates random noise on an observable by a Negative Binomial :math:`NB`.
    References to the negative binomial can be found `here <https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/NCSS/Negative_Binomial_Regression.pdf>`_
    .

    .. math::
        O &\sim NB(\mu=datapoint,\alpha)
    
    We keep the alpha parameter low to obtain a small variance which should than always be approximately the size of the mean.

    Parameters
    ----------
    df : new_cases , pandas.DataFrame
        Observable on which we want to add the noise

    noise_factor: :math:`\alpha`
        Alpha factor for the random number generation

    Returns
    -------
    array : 1-dim
        observable with added noise
    """

    def convert(mu, alpha):
        r = 1 / alpha
        p = mu / (mu + r)
        return r, 1 - p

    # Apply noise on every column
    for column in df:
        # Get values
        array = df[column].to_numpy()

        for i in range(len(array)):
            if (array[i] == 0) or (np.isnan(array[i])):
                continue
            log.debug(f"Data {array[i]}")
            r, p = convert(array[i], noise_factor)
            log.info(f"n {r}, p {p}")
            mean, var = nbinom.stats(r, p, moments="mv")
            log.debug(f"mean {mean} var {var}")
            array[i] = nbinom.rvs(r, p)
            log.debug(f"Drawn {array[i]}")

        df[column] = array

    return df
