import requests
import datetime
import os, sys
import zipfile
import pandas as pd
import sys
import logging
import json
from tqdm import tqdm


try:
    import covid19_inference as cov19
except:
    sys.path.append("../../covid_inference")
    import covid19_inference as cov19
logging.getLogger("numexpr").setLevel(logging.WARNING)
log = logging.getLogger("COVerAge DL")


def download_url(name, url, save_path, chunk_size=1024):
    r = requests.get(url, stream=True)
    pbar = tqdm(
        desc=f"{name} download",
        unit="B",
        total=int(r.headers["Content-Length"],),
        unit_scale=True,
    )

    with open(save_path, "wb") as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:  # filter out keep-alive new chunks
                fd.write(chunk)
                pbar.update(chunk_size)
    pbar.close()


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

    # Select male and female
    measure = measure.reset_index()
    if "b" in measure["Sex"]:
        measure = measure[measure["Sex"] == "b"]

    measure = measure.groupby(["Date", "Age", "AgeInt"]).sum()
    # Create age_string in format no1-no2 inclusive
    if country == "Czechia":
        measure = measure.reset_index()
        measure["age_str"] = measure["Age"].astype(str).astype(int).astype(str)
        measure = measure.set_index(["Date", "age_str"])
    else:
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
        ret_measure[age_str] = (
            measure.xs(age_str, level="age_str")
            .groupby("Date")
            .sum()
            .reindex(pd.date_range(data_begin, data_end))["Value"]
        )

    # Calculate daily cases
    ret_measure = ret_measure.diff()

    return ret_measure


def population(country):
    """
    Downloads population data to path and create population.csv for every country.
    Use local copy in data folder if it exists
    """

    data = pop.loc[pop["Location"] == country]
    data = data.loc[data["Time"] == 2019]
    data = data.set_index("AgeGrp")
    data = data["PopTotal"]
    data = data * 1000
    data.index.name = "age"
    data.astype("int64").to_csv(path + country + "/population.csv",)
    log.debug(f"Successfully created population file for {country}!")


def config(country):
    """
    Save config dict as json file
    """
    conf = {}
    conf["name"] = country
    conf["age_groups"] = {}
    if country == "Germany":
        conf["age_groups"]["age_group_0"] = [0, 34]
        conf["age_groups"]["age_group_1"] = [35, 59]
        conf["age_groups"]["age_group_2"] = [60, 79]
        conf["age_groups"]["age_group_3"] = [80, 100]
    elif country == "Greece":
        conf["age_groups"]["age_group_0"] = [0, 17]
        conf["age_groups"]["age_group_1"] = [18, 39]
        conf["age_groups"]["age_group_2"] = [40, 64]
        conf["age_groups"]["age_group_3"] = [65, 100]
    else:
        conf["age_groups"]["age_group_0"] = [0, 29]
        conf["age_groups"]["age_group_1"] = [30, 59]
        conf["age_groups"]["age_group_2"] = [60, 79]
        conf["age_groups"]["age_group_3"] = [80, 100]
    with open(path + country + "/config.json", "w") as outfile:
        json.dump(conf, outfile, indent=2)
    log.debug(f"Successfully created config file for {country}!")


def tests(country):
    # Tests
    name = country
    tests = owd.get_new(
        "tests", country=country, data_begin=data_begin, data_end=data_end
    )
    if country in ["Germany", "Netherlands", "Spain"]:
        # We map the weekly tests to the median each day
        tests = owd.get_total("tests", country=country).diff()
        # Get time diffs
        tests = pd.DataFrame(tests)
        tests["delta"] = tests.index
        tests["delta"] = tests["delta"] - tests["delta"].shift()
        tests = tests.dropna()

        tests = tests["total_tests"] / tests["delta"].apply(lambda x: x.days)
        dates = pd.date_range(data_begin - datetime.timedelta(days=14), data_end)
        tests = tests.reindex(dates)[data_begin:data_end].ffill()
        tests = tests.fillna(0.0)
        tests.index.name = "date"
    if country == "Belgium":
        tests = bel_data.get_new("tests", data_begin=data_begin, data_end=data_end)

        # Fill na if they do not fill
        tests = tests.reindex(pd.date_range(data_begin, data_end))
        tests.index.name = "date"
    tests.to_csv(path + name + "/tests.csv", date_format="%d.%m.%y")
    log.debug(f"Successfully created tests file for {country}!")


def interventions(country):
    """
    Gets interventions from oxford interventions tracker and saves to
    interventions.csv
    """
    if country == "Czechia":
        c_name_dl = "Czech Republic"
    else:
        c_name_dl = country

    interventions = pd.DataFrame()
    for policy in policies:
        interventions[policy] = ox.get_time_data(
            policy=policy, country=c_name_dl, data_begin=data_begin, data_end=data_end,
        )
    interventions.index = interventions.index.rename("date")
    interventions = interventions.ffill()  # Pad missing values with previous values
    interventions.to_csv(path + country + "/interventions.csv", date_format="%d.%m.%y")
    log.debug(f"Successfully created interventions file for {country}!")


def align_age_groups(country):
    def helper_cases(cases):
        df = pd.DataFrame()
        if country == "Germany":
            df["age_group_0"] = cases["0-4"] + cases["5-14"] + cases["15-34"]
            df["age_group_1"] = cases["35-59"]
            df["age_group_2"] = cases["60-79"]
            df["age_group_3"] = cases["80-104"]
        elif country in ["Switzerland", "Romania", "Portugal", "Finland"]:
            df["age_group_0"] = cases["0-9"] + cases["10-19"] + cases["20-29"]
            df["age_group_1"] = cases["30-39"] + cases["40-49"] + cases["50-59"]
            df["age_group_2"] = cases["60-69"] + cases["70-79"]
            df["age_group_3"] = cases["80-104"]
        elif country == "Greece":
            df["age_group_0"] = cases["0-17"]
            df["age_group_1"] = cases["18-39"]
            df["age_group_2"] = cases["40-64"]
            df["age_group_3"] = cases["65-104"]
        elif country == "Czechia":
            df["age_group_0"] = cases[[str(x) for x in range(0, 29)]].sum(axis=1)
            df["age_group_1"] = cases[[str(x) for x in range(30, 59)]].sum(axis=1)
            df["age_group_2"] = cases[[str(x) for x in range(60, 79)]].sum(axis=1)
            df["age_group_3"] = cases[
                [str(x) for x in range(80, 120) if str(x) in cases.keys()]
            ].sum(axis=1)
        elif country == "Bulgaria":
            df["age_group_0"] = cases["0-19"] + cases["20-29"]
            df["age_group_1"] = cases["30-39"] + cases["40-49"] + cases["50-59"]
            df["age_group_2"] = cases["60-69"] + cases["70-79"]
            df["age_group_3"] = cases["80-89"] + cases["90-104"]
        else:
            df["age_group_0"] = cases["0-9"] + cases["10-19"] + cases["20-29"]
            df["age_group_1"] = cases["30-39"] + cases["40-49"] + cases["50-59"]
            df["age_group_2"] = cases["60-69"] + cases["70-79"]
            df["age_group_3"] = cases["80-89"] + cases["90-104"]
        return df

    def helper_deaths(deaths):
        df = pd.DataFrame()
        if country == "Germany":
            df["age_group_0"] = cases["0-4"] + cases["5-14"] + cases["15-34"]
            df["age_group_1"] = cases["35-59"]
            df["age_group_2"] = cases["60-79"]
            df["age_group_3"] = cases["80-104"]
        elif country == "Belgium":
            df["age_group_0"] = deaths["0-24"]
            df["age_group_1"] = deaths["25-44"] + deaths["45-64"]
            df["age_group_2"] = deaths["65-74"] + deaths["75-84"]
            df["age_group_3"] = deaths["85-104"]
        elif country in ["Switzerland", "Romania", "Portugal", "Finland"]:
            df["age_group_0"] = cases["0-9"] + cases["10-19"] + cases["20-29"]
            df["age_group_1"] = cases["30-39"] + cases["40-49"] + cases["50-59"]
            df["age_group_2"] = cases["60-69"] + cases["70-79"]
            df["age_group_3"] = cases["80-104"]
        elif country == "Greece":
            df["age_group_0"] = cases["0-17"]
            df["age_group_1"] = cases["18-39"]
            df["age_group_2"] = cases["40-64"]
            df["age_group_3"] = cases["65-104"]
        elif country == "Czechia":
            df["age_group_0"] = cases[[str(x) for x in range(0, 29)]].sum(axis=1)
            df["age_group_1"] = cases[[str(x) for x in range(30, 59)]].sum(axis=1)
            df["age_group_2"] = cases[[str(x) for x in range(60, 79)]].sum(axis=1)
            df["age_group_3"] = cases[
                [str(x) for x in range(80, 120) if str(x) in cases.keys()]
            ].sum(axis=1)
        elif country == "Bulgaria":
            df["age_group_0"] = cases["0-19"] + cases["20-29"]
            df["age_group_1"] = cases["30-39"] + cases["40-49"] + cases["50-59"]
            df["age_group_2"] = cases["60-69"] + cases["70-79"]
            df["age_group_3"] = cases["80-89"] + cases["90-104"]
        else:
            df["age_group_0"] = deaths["0-9"] + deaths["10-19"] + deaths["20-29"]
            df["age_group_1"] = deaths["30-39"] + deaths["40-49"] + deaths["50-59"]
            df["age_group_2"] = deaths["60-69"] + deaths["70-79"]
            df["age_group_3"] = deaths["80-89"] + deaths["90-104"]
        return df

    # Load new cases
    cases = pd.read_csv(path + country + "/new_cases.csv").set_index("Unnamed: 0")
    cases.index.name = "date"
    helper_cases(cases).to_csv(path + country + "/new_cases.csv")

    # deaths = pd.read_csv(path + country + "/deaths.csv").set_index("Unnamed: 0")
    # deaths.index.name = "date"

    # For now we just sum all deaths (lazy)
    # deaths.sum(axis=1).to_csv(path + country + "/deaths.csv")
    log.debug(f"Successfully aligned age_groups for {country}!")


def download_and_save_file(
    url, f_name, path="../../data/download/", timestamp=False, overwrite=False,
):
    """
    Downloads a file and saves it to a path
    """
    if not os.path.isdir(path):
        os.makedirs(path)

    if timestamp:
        today = datetime.datetime.today()
        f_name, extension = os.path.splitext(f_name)
        f_name = f_name + f"_{today.strftime('%m_%d')}{extension}"

    if not os.path.isfile(path + f_name) or overwrite:
        download_url(os.path.splitext(f_name)[0], url, path + f_name)
    else:
        log.debug(f"Found existing {path + f_name}, skipping download.")
    return path + f_name


def validate(country):
    """
    Read csv files and check for nans
    """
    files = [
        "/tests.csv",
        "/interventions.csv",
        "/deaths.csv",
        # "/new_cases.csv", we expect nans here
        "/population.csv",
    ]

    for file in files:
        df = pd.read_csv(path + country + file)
        if df.isnull().sum().sum() > 0:
            log.error(
                f"Found {df.isnull().sum().sum()} NaNs in '{path + country + file}'"
            )


if __name__ == "__main__":
    # ------------------------------------------------------------------------------ #
    # Download and open used files
    # ------------------------------------------------------------------------------ #
    # Download new coverAgeDB file
    f_path = download_and_save_file(
        url="https://osf.io/9dsfk/download/",
        f_name="CoverAgeDB.zip",
        path="../../data/raw/coverage_db/",
        timestamp=True,
    )
    # Unzip file and load it
    with zipfile.ZipFile(f_path, "r") as zip_ref:
        zip_ref.extractall("../../data/raw/coverage_db/")

    # Open file csv
    df = pd.read_csv(
        "../../data/raw/coverage_db/Data/inputDB.csv",
        low_memory=False,
        header=1,
        encoding="ISO-8859-1",
    )
    df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y")
    # Set index of dataframe
    df = df.set_index(
        [
            "Country",
            "Region",
            "Date",
            "Age",
            "AgeInt",
            "Sex",
            "Measure",
            "Metric",
            "Short",
        ]
    )

    # Download new population file
    f_path = download_and_save_file(
        url="https://population.un.org/wpp/Download/Files/1_Indicators%20(Standard)/CSV_FILES/WPP2019_PopulationBySingleAgeSex_1950-2019.csv",
        f_name="WPP2019_PopulationBySingleAgeSex_1950-2019.csv",
        path="../../data/raw/",
    )
    pop = pd.read_csv(f_path)

    # Download Our world in data cases,deaths and tests
    owd = cov19.data_retrieval.OWD(True)
    # Download Oxford tracker intervetions
    ox = cov19.data_retrieval.OxCGRT(True)
    # Download Belgium data (it is only patially available in the owd dataset)
    bel_data = cov19.data_retrieval.Belgium(True)

    # ------------------------------------------------------------------------------ #
    # Preprocess data and save it to path
    # ------------------------------------------------------------------------------ #
    # For each country select data and save it
    path = "../../data/coverage_db/"
    data_begin = datetime.datetime(2020, 3, 2)
    data_end = datetime.datetime(2020, 8, 15)
    countries = [
        "Germany",
        # "France",
        "Italy",
        # "UK",
        "Belgium",
        "Denmark",
        "Spain",
        "Sweden",
        "Switzerland",
        "Romania",
        "Portugal",
        # "Norway",
        "Netherlands",
        "Greece",
        "Finland",
        "Czechia",
        "Bulgaria",
    ]
    policies = [
        "C1_School closing",
        "C2_Workplace closing",
        "C3_Cancel public events",
        "C4_Restrictions on gatherings",
        "C6_Stay at home requirements",
    ]

    if not os.path.isdir(path):
        os.mkdir(path)

    countries_bar = tqdm(countries, desc="Creating data files")
    for country in countries_bar:
        countries_bar.set_description(f"Creating data files [{country[0:3]}]")
        if not os.path.isdir(path + country):
            os.mkdir(path + country)

        get_data(country, "Cases", data_begin, data_end).to_csv(
            path + country + "/new_cases.csv", date_format="%d.%m.%y",
        )
        log.debug(f"Successfully created new cases file for {country}!")

        owd.get_new(
            "deaths", country=country, data_begin=data_begin, data_end=data_end
        ).to_csv(
            path + country + "/deaths.csv", date_format="%d.%m.%y",
        )
        log.debug(f"Successfully created deaths file for {country}!")

        align_age_groups(country)

        interventions(country)

        tests(country)

        # Population
        population(country)

        # Config
        config(country)

    # Short validation of created data i.e. check for nans
    countries_bar = tqdm(countries, desc="Validation")
    for country in countries_bar:
        countries_bar.set_description(f"Validation [{country[0:3]}]")
        validate(country)
