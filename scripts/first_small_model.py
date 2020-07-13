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
    I_0, R, g,
):
    r"""
        The Gamma distribution g that models the generation
        interval is parametrized by its mean

        Parameters:
        -----------

        I_0:
            Initial number of infectious.
        R:
            Reproduction number matrix.
        g:
            Generation interval


        Returns:
        --------

        Sample from distribution of new, daily cases

    """

    """ TODO:
    def new_infectious_cases_next_day(S_t, Ĩ_t):
        
        #Calculate new newly infectious per day
        #Sebastian: This will probably not work like that. Someone else should look over
        #it since im not too sure how to do that.
        
        
        #New susceptible pool
        

        Ĩ_t_new = tf.tensordot(Ĩ_t, g)

        S_t_new = S_t - Ĩ_t_new  # eq 4

        return [S_t_new, Ĩ_t_new]

    S_t, Ĩ_t = tf.scan(
        fn=new_infectious_cases_next_day,
        elems=[],
        initializer=[S_0, I_0],  # S_0 should be population size i.e. N
    )
    """
    return Ĩ_t


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

        Sebastian: Not too sure if it works like that, we will see.
    """
    R_reshaped = tfp.distributions.BatchReshape(
        distribution=R,
        batch_shape=[
            len(countries) * len(age_groups),
            len(countries) * len(age_groups),
        ],
    )

    """
        Create generation interval RV
    """
    g = covid19_npis.model.disease_spread()

    """
        Get RV for new_cases from SIR model

        it should have the shape:
            (data_end-data_begin).days+fcast
            x
            number_of_countries*number_of_age_groups

    """
    new_cases = yield NewCasesModel(I_0=I_0, R=R_reshaped, g=g)  # TODO

    """
        Delay new cases via convolution
    """
    # 1. Delay RV

    delay = yield pm.Normal(loc=4, scale=3, name="D", batch_shape=1)  # TODO shape

    # 2. Do convolution https://www.tensorflow.org/probability/api_docs/python/tfp/experimental/nn/Convolution
    new_cases_delayed = yield tfp.experimental.nn.Convolution()  # TODO

    # supprisingly df.to_numpy() gives us the right numpy array i.e. with shape [time,countries*age_groups]
    likelihood = pm.NegativeBinomial(new_cases, observed=df.to_numpy())
