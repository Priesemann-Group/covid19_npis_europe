# Append folder with covid19_npis package
import sys
import datetime
import pandas as pd
import numpy as np

sys.path.append("../")
sys.path.append("../covid19_inference/")
import covid19_npis
import covid19_inference as cov19inf


""" # Data retrieval / Convertions
"""
# Configs
data_begin = datetime.datetime(2020, 3, 1)
data_end = datetime.datetime(2020, 5, 6)
countries = ["Belgium", "Czechia", "Latvia", "Portugal", "Switzerland"]
age_groups = [
    "0-9",
    "10-19",
    "20-29",
    "30-39",
    "40-49",
    "50-59",
    "60-69",
    "70-79",
    "80-89",
    "90-99",
]

# Retriev data via covid19_inference toolbox
# and create a dictionary for the different retrieval classes and download the datasets
retrieval = {}
for country in countries:
    retrieval[country] = getattr(cov19inf.data_retrieval, country)(True)

# Get all age groups and append to dataframe
df = pd.DataFrame(index=pd.date_range(data_begin, data_end))
for country in countries:
    for age_group in age_groups:
        df = df.join(
            retrieval[country].get_new(
                data_begin=data_begin, data_end=data_end, age_group=age_group
            )
        )

# Reindex to use multiindex
df = df.reindex(
    columns=pd.MultiIndex.from_tuples(df.columns.values, names=["country", "age_group"])
)

# Convert every number to int
df = df.apply(lambda col: col.apply(lambda x: int(x) if pd.notnull(x) else x), axis=1)
# Replace every 0 with np.nan
df = df.replace(0, np.nan)

print(df)

""" # Model initilization
"""
import pymc4 as pm
import tensorflow as tf
import tensorflow_probability as tfp

""" 1. Implement equations 1-6 of manuscript
"""
@pm.model
def NewCasesModel(
    I_0,
    R,
    s_mu_input,
    mu_mu_input,
    s_theta_input,
    mu_theta_input
):
    r"""
        <Model description here>

        Parameters:
        -----------

        I_0:
            Initial number of infectious.

        R:
            Reproduction number matrix.

        s_mu_input: float
            s_mu is the scale of the distribution for mu_gen.

        mu_mu_input: float
            mu_mu is the mean of the distribution for mu_gen.

        s_theta_input: float
            s_theta is the scale of the distribution for theta_gen.

        mu_theta_input: float
            mu_theta is the mean of the distribution for theta_gen.

        Returns:
        --------

        Sample from distribution of new, daily cases

    """
    #new_I_t = S_t / N_pop[]

    # mean of generation interval distribution
    # k_mu is the shape of the distribution for mu_gen
    s_mu = s_mu_input#0.04
    mu_mu = mu_mu_input#4.8
    k_mu = mu_mu / s_mu
    mu_gen = yield pm.Gamma("mu_gen", k_mu, s_mu)

    # scale parameter of generation interval distribution
    # k_theta is the shape of the distribution for theta_gen
    s_theta = s_theta_input#0.1
    mu_theta = mu_theta_input#0.8
    k_theta = mu_theta / s_theta
    theta_gen = yield pm.Gamma("theta_gen", k_theta, s_theta)

    # shape parameter of generation interval distribution
    k_gen = mu_gen / theta_gen

    # generation interval distribution
    # Emil: How do I make this time dependent?
    g = yield pm.Gamma("g", k_gen, theta_gen)

@pm.model
def model(df):
    """ Create I_0 2d matrix withshape:
            10 [time]
            x
            number_of_countries*number_of_age_groups

        We need 10 here because we choose a length of 10 for the convolution at a later point.
    """
    I_0 = yield pm.HalfCauchy(
        loc=10,
        scale=150,
        name="I_0",
        batch_stack=[10, len(countries) * len(age_groups)],
    )

    """
        Create Reproduction number matrix (4,4)
    """
    R = yield pm.Normal(loc=1, scale=2.5, name="R", batch_stack=[4, 4])

    """
        Reshape R matrix to fit model i.e.
        shape:
            number_of_countries*number_of_age_groups
            x
            number_of_countries*number_of_age_groups
    """
    R_reshaped = tfp.distributions.BatchReshape(
        distribution=R,
        batch_shape=[
            len(countries) * len(age_groups),
            len(countries) * len(age_groups),
        ],
    )

    """
        Get RV for new_cases from SIR model

        it should have the shape:
            (data_end-data_begin).days+fcast
            x
            number_of_countries*number_of_age_groups

    """


    new_cases = yield NewCasesModel(I_0=I_0, R=R_reshaped) #TODO

    """
        Delay new cases via convolution
    """
    # 1. Delay RV

    delay = yield pm.Normal(loc=4, scale=3, name="D", batch_shape=1)  # TODO shape

    # 2. Do convolution https://www.tensorflow.org/probability/api_docs/python/tfp/experimental/nn/Convolution
    new_cases_delayed = yield tfp.experimental.nn.Convolution() #TODO



    #supprisingly df.to_numpy() gives us the right numpy array i.e. with shape [time,countries*age_groups]
    likelihood = pm.NegativeBinomial(new_cases, observed=df.to_numpy())
