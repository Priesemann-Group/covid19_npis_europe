# ------------------------------------------------------------------------------ #
# Generates a dataset for each bundesland in germany, see main below for configuration
#
# Runtime: <1min
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
from data_Germany_Landkreise import config, deaths, interventions, population, cases

sys.path.append("../covid_inference")
import covid19_inference as cov19

logging.getLogger("numexpr").setLevel(logging.WARNING)
log = logging.getLogger("Germany DL")


if __name__ == "__main__":
    # ------------------------------------------------------------------------------ #
    # Configuration
    # ------------------------------------------------------------------------------ #
    path = "../data/Germany_bundesländer/"
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

    tuples = {
        "1": "Schleswig-Holstein",
        "2": "Freie und Hansestadt Hamburg",
        "3": "Niedersachsen",
        "4": "Freie Hansestadt Bremen",
        "5": "Nordrhein-Westfalen",
        "6": "Hessen",
        "7": "Rheinland-Pfalz",
        "8": "Baden-Württemberg",
        "9": "Freistaat Bayern",
        "10": "Saarland",
        "11": "Berlin",
        "12": "Brandenburg",
        "13": "Mecklenburg-Vorpommern",
        "14": "Freistaat Sachsen",
        "15": "Sachsen-Anhalt",
        "16": "Freistaat Thüringen",
    }

    # ------------------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------------------ #
    if not os.path.isdir(path):  # Check if path exists
        os.mkdir(path)

    pbar = tqdm(
        tuples.items(), desc="Creating data files", total=len(tuples), position=0
    )

    for ID, bundesland in pbar:
        if not os.path.isdir(path + ID):
            os.mkdir(path + ID)
        config(ID=ID, path=path, name=bundesland)  # by name not id
        deaths(ID=ID, data_begin=data_begin, data_end=data_end, path=path)
        interventions(
            ID=ID,
            data_begin=data_begin,
            data_end=data_end,
            policies=policies,
            path=path,
            ox=ox,
        )
        cases(ID=ID, data_begin=data_begin, data_end=data_end, rki=rki, path=path)
        population(ID=ID, path=path, pop_total=pop_total)
