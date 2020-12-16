import sys


sys.path.append("../")

import covid19_npis
from covid19_npis.model import main_model
import numpy as np


# Load our data from csv files into our own custom data classes
# c1 = covid19_npis.data.Country("test-country-1", "../data/test_country_1")  # name
# c2 = covid19_npis.data.Country("test-country-2", "../data/test_country_2",)
countries = [
    "Germany",
    # "Belgium",
    # "Czechia",
    # "Denmark",
    # "Finland",
    # "Greece",
    # "Italy",
    # "Netherlands",
    # "Portugal",
    # "Romania",
    # "Spain",
    # "Sweden",
    "Switzerland",
]
c1, c2 = [
    covid19_npis.data.Country(
        f"../data/coverage_db/{country}",
    )
    for country in countries
]

# Construct our modelParams from the data.
modelParams = covid19_npis.ModelParams(countries=[c1, c2])

len_gen_interv_kernel = 12

params = {
    # population size per country and age group
    "N": np.array([[1e15, 1e15, 1e15, 1e15], [1e15, 1e15, 1e15, 1e15]]),
    # Reproduction number at t=0 per country and age group
    "R_0": np.array([3.3, 3.4]),
    # Initial infected
    "I_0_diff_base": 5 * np.array([[[1, 1, 1, 1], [1, 1, 1, 1]]]),
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
    "l_i_sign": 4 * np.ones((modelParams.num_interventions,)),
    # Alpha value of the change point
    "alpha_i_c_a": np.stack(
        [[[0.73, 0.72, 0.74, 0.75], [0.73, 0.72, 0.74, 0.75]]]
        * modelParams.num_interventions
    )
    * 0.7,
    "C": np.stack(
        np.array(
            [
                [
                    [1, 0.1, 0.1, 0.1],
                    [0.1, 1, 0.1, 0.1],
                    [0.1, 0.1, 1, 0.1],
                    [0.1, 0.1, 0.1, 1],
                ]
            ]
            * 2
        )
    ),
    # Number of timesteps
    # Number of Nans before data (cuts t_max)
    # Generation interval
    "g_mu": 4.0,
    "g_theta": 1.0,
    "mean_delay": np.array([[12.0], [12.5]]),
    "sigma": 0.1,
    "phi_tests_reported": np.ones((modelParams.num_countries,)),
    "phi_age": np.ones((modelParams.num_age_groups,)),
    "theta_delay": np.zeros((modelParams.num_countries,)),
    "delay": np.linspace(10, 14, modelParams.num_splines)
    * np.ones((modelParams.num_countries, modelParams.num_splines)),
    "mu_testing_state": np.array([0, 0, 0, 12]),
    "sigma_testing_state": np.einsum(
        "...ij,...j->...ij",
        np.linalg.cholesky(
            np.array(
                [
                    [1, 0.1, 0.1, 0.1],
                    [0.1, 1, 0.1, 0.1],
                    [0.1, 0.1, 1, 0.1],
                    [0.1, 0.1, 0.1, 1],
                ]
            )
        ),
        np.array([0.1, 0.1, 0.1, 0.1]),
    ),
    "Phi_IFR": np.array([[0.1, 0.01, 0.001, 0.0001], [0.1, 0.01, 0.001, 0.0001]]),
    "death_m": np.array([14.0, 14.0]),
    "death_theta": np.array([1.0, 1.0]),
}

(
    positive_tests,
    total_tests,
    new_E_t,
    reporting_delay_kernel,
    R_t,
    h_0_t,
    cases_delayed_deaths,
) = covid19_npis.test_data.data_generators.test_data_from_model(
    main_model,
    modelParams,
    params,
    [
        "positive_tests",
        "total_tests",
        "new_E_t",
        "reporting_delay_kernel",
        "R_t",
        "h_0_t",
        "cases_delayed_deaths",
    ],
)


# ------------------------------------------------------------------------------ #
# Plotting
# ------------------------------------------------------------------------------ #

import matplotlib.pyplot as plt

t = np.arange(0, modelParams.length_sim)

# Plotting dict
plots = {
    "Positive tests": positive_tests,
    "Total tests": total_tests,
    "Deaths": cases_delayed_deaths,
}
# For now dirty plot does not differentiate between countries!
for key, item in plots.items():
    ax = covid19_npis.plot.time_series._timeseries(
        x=t,
        y=item,
        ls="-",
        what="fcast",
    )
    ax.set_title(key)
    plt.tight_layout()
    fig = plt.gcf()
    fig.show()


# covid19_npis.test_data.data_generators.save_data(
#    "../data/test_data_from_model", new_cases_inferred, R_t, interv
# )
