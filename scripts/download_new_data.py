import sys
import logging
import time
import os
import datetime
import pandas as pd

sys.path.append("../covid19_inference/")

from covid19_inference import data_retrieval

log = logging.getLogger(__name__)

# Config ---------------
countries = ["France", "Germany"]
path = "../data"
begin = datetime.datetime(2020, 5, 17)
end = datetime.datetime.today()
age_groups = {
    "France": [9, 19, 29, 39, 49, 59, 69, 79, 89, 90],
    "Germany": ["A00-A04", "A05-A14", "A15-A34", "A35-A59", "A60-A79", "A80+"],
}
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

# -----------------------

# Check if folders for countries exist if not, create them!
for country in countries:
    # Check if folder exists and create
    if not os.path.exists(path + "/" + country):
        os.makedirs(path + "/" + country)
        print(f"Created folder {path}/{country}")


""" Interventions:
We want to start with the easy part ;)
"""
ox = data_retrieval.OxCGRT(True)  # interv


for country in countries:
    interventions = pd.DataFrame()
    for policy in policies:
        interventions[policy] = ox.get_time_data(
            policy=policy, country=country, data_begin=begin, data_end=end,
        )
    interventions.to_csv(path + f"/{country}/interventions.csv", date_format="%d.%m.%y")


""" Tests:
This should also be quite easy, but there are some problems
if the cases do not occure daily! That is for example in germany the case.
"""
owd = data_retrieval.OWD(True)  # tests

for country in countries:
    tests = owd.get_new("tests", country=country, data_begin=begin, data_end=end)
    if country == "Germany":
        # We map the weekly tests to the median each day
        tests = owd.get_total("tests", country=country).diff()
        dates = pd.date_range(begin - datetime.timedelta(days=14), end)
        tests = tests.reindex(dates)[begin:end].ffill() / 7

    tests.to_csv(path + f"/{country}/tests.csv", date_format="%d.%m.%y")


""" New Cases/positive tests:
This is kinda the hardest part, since we want to have simmilar age groups!
Agegroups: 0-29,30-59,60-79,80+
"""
# cases France
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
new_cases["0-29"] = new_cases["09"] + new_cases["19"] + new_cases["29"]
new_cases = new_cases.drop(columns=["09", "19", "29"])
# Do the same for agegroups 30-39,40-49,50-59
new_cases["30-59"] = new_cases["39"] + new_cases["49"] + new_cases["59"]
new_cases = new_cases.drop(columns=["39", "49", "59"])

new_cases["60-79"] = new_cases["69"] + new_cases["79"]
new_cases = new_cases.drop(columns=["69", "79"])

new_cases["80+"] = new_cases["89"] + new_cases["90"]
new_cases = new_cases.drop(columns=["89", "90"])

new_cases.to_csv(path + f"/France/new_cases.csv", date_format="%d.%m.%y")


# cases germany
Germany = data_retrieval.RKI(True)
age_groups = [
    "A00-A04",
    "A05-A14",
    "A15-A34",
    "A35-A59",
    "A60-A79",
    "A80+",
]  # available age_groups
for age_group in age_groups:
    new_cases[age_group] = Germany.get_new(
        "confirmed", data_begin=begin, data_end=end, age_group=age_group
    )

new_cases["0-34"] = new_cases["A00-A04"] + new_cases["A05-A14"] + new_cases["A15-A34"]
new_cases = new_cases.drop(columns=["A00-A04", "A05-A14", "A15-A34"])

new_cases["35-59"] = new_cases["A35-A59"]
new_cases = new_cases.drop(columns="A35-A59")

new_cases["60-79"] = new_cases["A60-A79"]
new_cases = new_cases.drop(columns="A60-A79")

new_cases["80+"] = new_cases["A80+"]
new_cases = new_cases.drop(columns="A80+")

new_cases.to_csv(path + f"/Germany/new_cases.csv", date_format="%d.%m.%y")
