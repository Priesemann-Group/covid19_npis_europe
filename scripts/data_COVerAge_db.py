import requests
import datetime
import os
import zipfile
import pandas as pd


def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, "wb") as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)


# Download new coverAge file if it does not exist
today = datetime.datetime.today()
file = f"../data/coverage_db_{today.strftime('%m_%d')}.zip"
if not os.path.isfile(file):
    download_url("https://osf.io/9dsfk/download", file)

    # Unzip files
    with zipfile.ZipFile(file, "r") as zip_ref:
        zip_ref.extractall("../data/unzip/")


# Open file csv
df = pd.read_csv("../data/unzip/Data/inputDB.csv", low_memory=False, header=1)
df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y")
# Set index of dataframe
df = df.set_index(
    ["Country", "Region", "Date", "Age", "AgeInt", "Measure", "Metric", "Short"]
)


def get_data(country, measure, data_begin, data_end):
    """
    Gets the dataframe tuple for the selected country

    Parameters
    ----------
    country: str
        Name for the country in the dataset
    measure: str
        Name of the measure i.e. "Cases", "Tests", "Deaths"
    """

    # Get country
    data = df.xs(country, level="Country")

    # Select all regions
    data = data.xs("All", level="Region")

    # Select measure cases
    measure = data.xs(measure, level="Measure")

    # Sum over all entries i.e m and f
    measure = measure.groupby(["Date", "Age", "AgeInt"]).sum()

    # Create age_string in format no1-no2 inclusive
    measure = measure.reset_index()
    measure["age_str"] = (
        measure["Age"].astype(str)
        + "-"
        + (measure["Age"].astype(float) + measure["AgeInt"].astype(float) - 1)
        .astype(int)
        .astype(str)
    )
    measure = measure.set_index(["Date", "age_str"])

    # Create new dataframe with agestrings as columns and date as index
    ret_measure = pd.DataFrame()
    for age_str in measure.index.get_level_values(level="age_str").unique():
        ret_measure[age_str] = measure.xs(age_str, level="age_str").reindex(
            pd.date_range(data_begin, data_end)
        )["Value"]

    return ret_measure


# For each country select data and save it

path = "../data/coverage_db/"
data_begin = datetime.datetime(2020, 3, 2)
data_end = datetime.datetime.today()
countries = [
    "Germany",
    "France",
    "Italy",
    # "UK",
    "Belgium",
    "Denmark",
    "Spain",
    "Sweden",
    "Switzerland",
    "Romania",
    "Portugal",
    "Norway",
    "Netherlands",
    "Greece",
    "Finland",
    "Czechia",
    "Bulgaria",
]


if not os.path.isdir(path):
    os.mkdir(path)


for country in countries:
    if not os.path.isdir(path + country):
        os.mkdir(path + country)

    get_data(country, "Cases", data_begin, data_end).to_csv(
        path + country + "/new_cases.csv"
    )
    get_data(country, "Deaths", data_begin, data_end).to_csv(
        path + country + "/deaths.csv"
    )
