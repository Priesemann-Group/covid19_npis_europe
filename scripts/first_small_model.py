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
data_end = datetime.datetime(2020, 3, 20)
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

""" ## 
"""


""" # Model initilization
"""

""" # Sample
"""

""" # Plotting
"""
