# ------------------------------------------------------------------------------ #
# Generates a dataset for each bundesland in germany, see main for configuration
#
# Runtime: 3min
# ------------------------------------------------------------------------------ #
import requests
import datetime
import os
import zipfile
import pandas as pd
import sys
import logging
import json
from tqdm.auto import tqdm
from data_COVerAge_db import download_and_save_file

sys.path.append("../covid_inference")
import covid19_inference as cov19

logging.getLogger("numexpr").setLevel(logging.WARNING)
log = logging.getLogger("Germany DL")


def config(ID, path):
    """
        Generate config for bundesland and saves it
    """
    config = {
        "name": ID,
        "age_groups": {
            "age_group_0": [0, 34],
            "age_group_1": [35, 59],
            "age_group_2": [60, 79],
            "age_group_3": [80, 100],
        },
    }
    with open(path + ID + "/config.json", "w") as outfile:
        json.dump(config, outfile, indent=2)
    log.debug(f"Successfully created config file for '{ID}'!")


def population(ID, path, pop_total):
    """
        Download population file if not present and saves, also does some weighting
        for each age group to increase the resolution
    """
    pop_raw = pd.read_csv(
        "../data/raw/12411-0017.csv",
        encoding="cp1252",
        header=5,
        delimiter=";",
        skipfooter=4,
        engine="python",
    )
    pop_raw.columns.values[0] = "date"
    pop_raw.columns.values[1] = "ID"
    pop_raw.columns.values[2] = "name"

    pop_raw.columns.values[3] = "0-2"
    pop_raw.columns.values[4] = "3-6"
    pop_raw.columns.values[5] = "6-9"
    pop_raw.columns.values[6] = "10-14"
    pop_raw.columns.values[7] = "15-17"
    pop_raw.columns.values[8] = "18-19"
    pop_raw.columns.values[9] = "20-24"
    pop_raw.columns.values[10] = "25-29"
    pop_raw.columns.values[11] = "30-34"
    pop_raw.columns.values[12] = "35-39"
    pop_raw.columns.values[13] = "40-44"
    pop_raw.columns.values[14] = "45-49"
    pop_raw.columns.values[15] = "50-54"
    pop_raw.columns.values[16] = "55-59"
    pop_raw.columns.values[17] = "60-64"
    pop_raw.columns.values[18] = "65-74"
    pop_raw.columns.values[19] = "75-100"

    pop_raw["ID"] = pop_raw["ID"].astype("str")

    # Get data by id
    pop = pop_raw[pop_raw["ID"] == ID]

    # Weight with total population distribution from wpp (to get every age)

    # We weight each age_group here
    pop_new = pd.DataFrame()

    for col in pop.columns:
        if col in ["date", "ID", "name", "Insgesamt"]:
            continue
        lower, upper = [int(i) for i in col.split("-")]
        # Get distribution for agegroup
        dist = (
            pop_total[lower : upper + 1] / pop_total[lower : upper + 1].sum()
        )  # +1 -> inklusive

        for i, val in dist.items():
            pop_new[i] = [float(val) * float(pop[col].values[0])]

    # Save file
    pop_new = pop_new.T
    pop_new.index.name = "age"
    pop_new.astype("int64").to_csv(path + ID + "/population.csv",)
    log.debug(f"Successfully created population file for {ID}!")
    return pop_new


def interventions(ID, data_begin, data_end, policies, path, ox):
    """
    For now asuming same interventions accross all region
    """
    interventions = pd.DataFrame()
    for policy in policies:
        interventions[policy] = ox.get_time_data(
            policy=policy, country="Germany", data_begin=data_begin, data_end=data_end,
        )
    interventions.index = interventions.index.rename("date")
    interventions = interventions.ffill()  # Pad missing values with previous values
    interventions.to_csv(path + ID + "/interventions.csv", date_format="%d.%m.%y")
    log.debug(f"Successfully created interventions file for {ID}!")


def deaths(ID, data_begin, data_end, path):
    """
        Datafile by Matthias linden
    """
    deaths = pd.read_csv("../data/raw/DeathsRKI.csv")
    deaths["TodesMeldedatum"] = pd.to_datetime(
        deaths["TodesMeldedatum"], format="%Y-%m-%d"
    )
    deaths["IdLandkreis"] = deaths["IdLandkreis"].astype("int64").astype("str")

    deaths = deaths.groupby(["IdLandkreis", "TodesMeldedatum", "Altersgruppe"])[
        "AnzahlTodesfall"
    ].sum()
    deaths = deaths.xs(key=str(ID), level="IdLandkreis", axis=0)
    deaths = deaths.unstack()

    if "A00-A04" not in deaths.columns:
        deaths["A00-A04"] = 0

    if "A05-A14" not in deaths.columns:
        deaths["A05-A14"] = 0

    if "A15-A34" not in deaths.columns:
        deaths["A15-A34"] = 0

    if "A35-A59" not in deaths.columns:
        deaths["A35-A59"] = 0

    if "A60-A79" not in deaths.columns:
        deaths["A60-A79"] = 0

    if "A80+" not in deaths.columns:
        deaths["A80+"] = 0

    if "unbekannt" not in deaths.columns:
        deaths["unbekannt"] = 0

    dates = pd.date_range(data_begin, data_end)
    deaths = deaths.reindex(dates, fill_value=0)
    deaths = deaths.fillna(0)
    deaths["age_group_0"] = deaths["A05-A14"] + deaths["A05-A14"] + deaths["A15-A34"]
    deaths = deaths.rename(
        columns={
            "A35-A59": "age_group_1",
            "A60-A79": "age_group_2",
            "A80+": "age_group_3",
        }
    )
    deaths = deaths.drop(columns=["A00-A04", "A05-A14", "A15-A34", "unbekannt"])
    deaths = deaths.sort_index(axis=1)
    deaths.index.name = "date"
    deaths.name = ID
    deaths = deaths[data_begin:data_end]
    deaths.to_csv(
        path + ID + "/deaths.csv", date_format="%d.%m.%y",
    )
    log.debug(f"Successfully created deaths file for {ID}!")


def cases(ID, data_begin, data_end, rki):
    data = rki.data
    data["IdLandkreis"] = data["IdLandkreis"].astype("str")
    data = data.groupby(["IdLandkreis", "date", "Altersgruppe"])["confirmed"].sum()
    data = data.xs(key=str(ID), level="IdLandkreis", axis=0)
    data = data.unstack()

    if "A00-A04" not in data.columns:
        data["A00-A04"] = 0

    if "A05-A14" not in data.columns:
        data["A05-A14"] = 0

    if "A15-A34" not in data.columns:
        data["A15-A34"] = 0

    if "A35-A59" not in data.columns:
        data["A35-A59"] = 0

    if "A60-A79" not in data.columns:
        data["A60-A79"] = 0

    if "A80+" not in data.columns:
        data["A80+"] = 0

    if "unbekannt" not in data.columns:
        data["unbekannt"] = 0

    dates = pd.date_range(data_begin, data_end)
    data = data.reindex(dates, fill_value=0)
    data = data.fillna(0)
    data["age_group_0"] = data["A05-A14"] + data["A05-A14"] + data["A15-A34"]
    data = data.rename(
        columns={
            "A35-A59": "age_group_1",
            "A60-A79": "age_group_2",
            "A80+": "age_group_3",
        }
    )
    data = data.drop(columns=["A00-A04", "A05-A14", "A15-A34", "unbekannt"])
    data = data.sort_index(axis=1)
    data.index.name = "date"
    data.name = ID
    data = data[data_begin:data_end]
    data.to_csv(
        path + ID + "/new_cases.csv", date_format="%d.%m.%y",
    )
    log.debug(f"Successfully created data file for {ID}!")


def get_all_ids():
    deaths = pd.read_csv("../data/raw/DeathsRKI.csv")
    deaths["IdLandkreis"] = deaths["IdLandkreis"].astype("int64").astype("str")
    return deaths["IdLandkreis"].unique()


if __name__ == "__main__":

    # ------------------------------------------------------------------------------ #
    # Configuration
    # ------------------------------------------------------------------------------ #
    path = "../data/Germany_regions/"
    data_begin = datetime.datetime(2020, 4, 20)
    data_end = datetime.datetime(2020, 12, 27)
    policies = [
        "C1_School closing",
        "C2_Workplace closing",
        "C3_Cancel public events",
        "C4_Restrictions on gatherings",
        "C6_Stay at home requirements",
    ]

    # Download Oxford tracker intervetions
    ox = cov19.data_retrieval.OxCGRT(True)
    # Download Rki data
    rki = cov19.data_retrieval.RKI(True)
    # Download new population file and format it
    f_path = download_and_save_file(
        url="https://population.un.org/wpp/Download/Files/1_Indicators%20(Standard)/CSV_FILES/WPP2019_PopulationBySingleAgeSex_1950-2019.csv",
        f_name="WPP2019_PopulationBySingleAgeSex_1950-2019.csv",
        path="../data/raw/",
    )
    pop_total = pd.read_csv(
        "../data/raw/WPP2019_PopulationBySingleAgeSex_1950-2019.csv"
    )
    pop_total = pop_total.loc[pop_total["Location"] == "Germany"]
    pop_total = pop_total.loc[pop_total["Time"] == 2019]
    pop_total = pop_total.set_index("AgeGrp")
    pop_total = pop_total["PopTotal"]
    pop_total = pop_total * 1000
    pop_total.index.name = "age"
    pop_total = pop_total.astype("float")

    # Somehow some IDs are not present in some datasets..data_end
    expect_ids = [
        "11004",
        "11007",
        "11008",
        "11011",
        "11002",
        "11012",
        "11005",
        "11003",
        "11006",
        "11009",
        "11001",
        "11010",
    ]

    # ------------------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------------------ #
    if not os.path.isdir(path):  # Check if path exists
        os.mkdir(path)

    IDs = get_all_ids()

    pbar = tqdm(IDs, desc="Creating data files", total=len(IDs), position=0)

    for ID in pbar:
        pbar.set_description(f"Creating data files [{ID}]")

        if ID in expect_ids:
            continue
        if not os.path.isdir(path + ID):
            os.mkdir(path + ID)

        try:
            config(ID=ID, path=path)

            deaths(ID=ID, data_begin=data_begin, data_end=data_end, path=path)
            interventions(
                ID=ID,
                data_begin=data_begin,
                data_end=data_end,
                policies=policies,
                path=path,
                ox=ox,
            )
            population(ID=ID, path=path, pop_total=pop_total)
            cases(ID=ID, data_begin=data_begin, data_end=data_end, rki=rki)
        except Exception as e:
            print(ID)
            expect_ids.append(ID)
