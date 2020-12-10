"""
For each country we need to create a config folder with multiple Files:

- new_cases.csv
    - Time/Date column has to be named "date" or "time"
    - Age group columns have to be named consistent between different data and
    countries 

- interventions.csv
    - Time/Date column has to be named "date" or "time"
    - Different intervention as additional columns with intervention name as
    column name

- tests.csv
    - Time/Date column has to be named "date" or "time"
    - Daily performed tests column with name "tests"

- deaths.csv
    - Time/Date column has to be named "date" or "time"
    - Daily deaths column has to be named "deaths"
    - Optional: Daily deaths per age group same column names as in new_cases

- population.csv
    - Age column named "age"
    - Column Number of people per age named "PopTotal"

- config.json, dict:
    - name : "country_name"
    - age_groups : dict 
        - "column_name" : [age_lower, age_upper]
"""


import sys
import logging
import time
import os
import datetime
import pandas as pd
import json

sys.path.append("../../covid19_inference/")

from covid19_inference import data_retrieval

log = logging.getLogger(__name__)

# ------------------------------------------------------------------------------ #
# Config/Globals
# ------------------------------------------------------------------------------ #
countries = ["France", "Germany", "Belgium", "Portugal"]
path = "../../data"
begin = datetime.datetime(2020, 5, 17)
end = datetime.datetime.now() - datetime.timedelta(days=1)
_age_groups = ["age_group_0", "age_group_1", "age_group_2", "age_group_3"]
policies = [
    "C1_School closing",
    "C2_Workplace closing",
    "C3_Cancel public events",
    "C4_Restrictions on gatherings",
    "C5_Close public transport",
    "C6_Stay at home requirements",
    "C7_Restrictions on internal movement",
    "C8_International travel controls",
]
# ------------------------------------------------------------------------------ #
# Look into __main__ for the execution order


def interventions():
    """
    Gets interventions from oxford interventions tracker and saves to 
    interventions.csv
    """
    ox = data_retrieval.OxCGRT(True)  # interv

    for country in countries:
        interventions = pd.DataFrame()
        for policy in policies:
            interventions[policy] = ox.get_time_data(
                policy=policy, country=country, data_begin=begin, data_end=end,
            )
        interventions.index = interventions.index.rename("date")
        interventions = interventions.ffill()  # Pad missing values with previous values
        interventions.to_csv(
            path + f"/{country}/interventions.csv", date_format="%d.%m.%y"
        )
    log.info("Successfully created interventions files!")


def tests():
    """
    Gets number of tests from our world in data and saves
    them to tests.csv
    """
    owd = data_retrieval.OWD(True)  # tests

    for country in countries:
        tests = owd.get_new("tests", country=country, data_begin=begin, data_end=end)
        if country == "Germany":
            # We map the weekly tests to the median each day
            tests = owd.get_total("tests", country=country).diff()
            dates = pd.date_range(begin - datetime.timedelta(days=14), end)
            tests = tests.reindex(dates)[begin:end].ffill() / 7
            tests.index.name = "date"
        if country == "Belgium":
            tests = data_retrieval.Belgium(True).get_new(
                "tests", data_begin=begin, data_end=end
            )

        # Fill na if they do not fill
        tests = tests.reindex(pd.date_range(begin, end))
        tests.index.name = "date"

        tests.to_csv(path + f"/{country}/tests.csv", date_format="%d.%m.%y")
    log.info("Successfully created tests files!")


def new_cases():
    """
    Gets new cases/positive tests by age group for each country.
    This is kinda the hardest part, since we want to have similar age groups!
    Age groups: 0-29,30-59,60-79,80+
    """
    # cases France
    def france():
        France = data_retrieval.countries.France(True)
        age_groups = [
            "09",
            "19",
            "29",
            "39",
            "49",
            "59",
            "69",
            "79",
            "89",
            "90",
        ]  # available age_groups

        new_cases = pd.DataFrame()
        for age_group in age_groups:
            new_cases[age_group] = France.get_new(
                "confirmed", data_begin=begin, data_end=end, age_group=age_group
            )

        # We want to sum over 0-9,10-19 and 19-29 for the first age group:
        new_cases["age_group_0"] = new_cases["09"] + new_cases["19"] + new_cases["29"]
        new_cases = new_cases.drop(columns=["09", "19", "29"])
        # Do the same for agegroups 30-39,40-49,50-59
        new_cases["age_group_1"] = new_cases["39"] + new_cases["49"] + new_cases["59"]
        new_cases = new_cases.drop(columns=["39", "49", "59"])

        new_cases["age_group_2"] = new_cases["69"] + new_cases["79"]
        new_cases = new_cases.drop(columns=["69", "79"])

        new_cases["age_group_3"] = new_cases["89"] + new_cases["90"]
        new_cases = new_cases.drop(columns=["89", "90"])
        # Fill na if they do not fill
        new_cases = new_cases.reindex(pd.date_range(begin, end))
        new_cases.index.name = "date"
        new_cases.to_csv(path + f"/France/new_cases.csv", date_format="%d.%m.%y")

    # cases germany
    def germany():
        Germany = data_retrieval.RKI(True)
        age_groups = [
            "A00-A04",
            "A05-A14",
            "A15-A34",
            "A35-A59",
            "A60-A79",
            "A80+",
        ]  # available age_groups
        new_cases = pd.DataFrame()
        for age_group in age_groups:
            new_cases[age_group] = Germany.get_new(
                "confirmed", data_begin=begin, data_end=end, age_group=age_group
            )

        new_cases["age_group_0"] = (
            new_cases["A00-A04"] + new_cases["A05-A14"] + new_cases["A15-A34"]
        )
        new_cases = new_cases.drop(columns=["A00-A04", "A05-A14", "A15-A34"])

        new_cases["age_group_1"] = new_cases["A35-A59"]
        new_cases = new_cases.drop(columns="A35-A59")

        new_cases["age_group_2"] = new_cases["A60-A79"]
        new_cases = new_cases.drop(columns="A60-A79")

        new_cases["age_group_3"] = new_cases["A80+"]
        new_cases = new_cases.drop(columns="A80+")
        new_cases.index = new_cases.index.rename("date")
        new_cases.to_csv(path + f"/Germany/new_cases.csv", date_format="%d.%m.%y")

    def belgium():
        Belgium = data_retrieval.Belgium(True)
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
            "90+",
        ]
        new_cases = pd.DataFrame(index=pd.date_range(begin, end))
        for age_group in age_groups:
            new_cases[age_group] = Belgium.get_new(
                "confirmed", data_begin=begin, data_end=end, age_group=age_group
            )
        new_cases = new_cases.fillna(0)

        new_cases["age_group_0"] = (
            new_cases["0-9"] + new_cases["10-19"] + new_cases["20-29"]
        )
        new_cases = new_cases.drop(columns=["0-9", "10-19", "20-29"])

        new_cases["age_group_1"] = (
            new_cases["30-39"] + new_cases["40-49"] + new_cases["50-59"]
        )
        new_cases = new_cases.drop(columns=["30-39", "40-49", "50-59"])

        new_cases["age_group_2"] = new_cases["60-69"] + new_cases["70-79"]
        new_cases = new_cases.drop(columns=["60-69", "70-79"])

        new_cases["age_group_3"] = new_cases["80-89"] + new_cases["90+"]
        new_cases = new_cases.drop(columns=["80-89", "90+"])
        new_cases.index = new_cases.index.rename("date")
        new_cases.to_csv(path + f"/Belgium/new_cases.csv", date_format="%d.%m.%y")

    def portugal():
        Portugal = data_retrieval.Portugal(True)
        age_groups = [
            "0-9",
            "10-19",
            "20-29",
            "30-39",
            "40-49",
            "50-59",
            "60-69",
            "70-79",
            "80-100",
        ]
        new_cases = pd.DataFrame(index=pd.date_range(begin, end))
        for age_group in age_groups:
            new_cases[age_group] = Portugal.get_new(
                "confirmed", data_begin=begin, data_end=end, age_group=age_group
            )
        new_cases = new_cases.fillna(0)

        new_cases["age_group_0"] = (
            new_cases["0-9"] + new_cases["10-19"] + new_cases["20-29"]
        )
        new_cases = new_cases.drop(columns=["0-9", "10-19", "20-29"])

        new_cases["age_group_1"] = (
            new_cases["30-39"] + new_cases["40-49"] + new_cases["50-59"]
        )
        new_cases = new_cases.drop(columns=["30-39", "40-49", "50-59"])

        new_cases["age_group_2"] = new_cases["60-69"] + new_cases["70-79"]
        new_cases = new_cases.drop(columns=["60-69", "70-79"])

        new_cases["age_group_3"] = new_cases["80-100"]
        new_cases = new_cases.drop(columns=["80-100"])
        new_cases.index = new_cases.index.rename("date")
        new_cases.to_csv(path + f"/Portugal/new_cases.csv", date_format="%d.%m.%y")

    france()
    germany()
    belgium()
    portugal()
    log.info("Successfully created new_cases files!")


def deaths():
    """
    Get and save number of covid deaths. 
    """
    owd = data_retrieval.OWD(True)
    for country in countries:
        deaths = owd.get_new("deaths", country=country, data_begin=begin, data_end=end)
        if country == "Belgium":
            deaths = data_retrieval.Belgium(True).get_new(
                "deaths", data_begin=begin, data_end=end
            )
            deaths.index.name = "date"
            deaths.name = "new_deaths"

        deaths.to_csv(path + f"/{country}/deaths.csv", date_format="%d.%m.%y")


def population():
    """
    Downloads population data to path and create population.csv for every country.
    Use local copy in data folder if it exists
    """
    if not os.path.isfile(path + f"/WPP2019_PopulationBySingleAgeSex_1950-2019.csv"):
        # Download file if it does not exist yet
        import requests

        url = "https://population.un.org/wpp/Download/Files/1_Indicators%20(Standard)/CSV_FILES/WPP2019_PopulationBySingleAgeSex_1950-2019.csv"
        log.info("Beginning file download for population file, this will take a while!")
        myfile = requests.get(url)
        open(path + f"/WPP2019_PopulationBySingleAgeSex_1950-2019.csv", "wb").write(
            myfile.content
        )

    pop = pd.read_csv(path + f"/WPP2019_PopulationBySingleAgeSex_1950-2019.csv")
    for country in countries:
        data = pop.loc[pop["Location"] == country]
        data = data.loc[data["Time"] == 2019]
        data = data.set_index("AgeGrp")
        data = data["PopTotal"]
        data = data * 1000
        data.index.name = "age"
        data.astype("int64").to_csv(path + f"/{country}/population.csv",)
    log.info("Successfully created population files!")


def config():
    """
    Save config dict as json file
    """
    for country in countries:
        conf = {}
        conf["name"] = country
        conf["age_groups"] = {}
        if country == "Germany":
            conf["age_groups"]["age_group_0"] = [0, 34]
            conf["age_groups"]["age_group_1"] = [35, 59]
            conf["age_groups"]["age_group_2"] = [60, 79]
            conf["age_groups"]["age_group_3"] = [80, 100]
        elif country == "France":
            conf["age_groups"]["age_group_0"] = [0, 29]
            conf["age_groups"]["age_group_1"] = [30, 59]
            conf["age_groups"]["age_group_2"] = [60, 79]
            conf["age_groups"]["age_group_3"] = [80, 100]
        elif country == "Belgium":
            conf["age_groups"]["age_group_0"] = [0, 29]
            conf["age_groups"]["age_group_1"] = [30, 59]
            conf["age_groups"]["age_group_2"] = [60, 79]
            conf["age_groups"]["age_group_3"] = [80, 100]
        elif country == "Portugal":
            conf["age_groups"]["age_group_0"] = [0, 29]
            conf["age_groups"]["age_group_1"] = [30, 59]
            conf["age_groups"]["age_group_2"] = [60, 79]
            conf["age_groups"]["age_group_3"] = [80, 100]
        with open(path + f"/{country}/config.json", "w") as outfile:
            json.dump(conf, outfile, indent=2)
    log.info("Successfully created config files!")


def check_dates():

    # Get the length of the date column. Should be the same for all files!
    files = ["deaths.csv", "interventions.csv", "new_cases.csv", "tests.csv"]
    len_index = (end - begin).days + 1

    for country in countries:
        for f in files:
            if len(pd.read_csv(path + f"/{country}/{f}")["date"]) != len_index:
                log.error(
                    f"Date index of File {path}/{country}/{f} does not match other files!"
                )


if __name__ == "__main__":

    # Check if folders for countries exist if not, create them!
    for country in countries:
        # Check if folder exists and create
        if not os.path.exists(path + "/" + country):
            os.makedirs(path + "/" + country)
            print(f"Created folder {path}/{country}")

    interventions()
    tests()
    new_cases()
    deaths()
    population()
    config()

    check_dates()
