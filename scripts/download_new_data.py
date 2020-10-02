import sys
import logging
import time
import os
import datetime
import pandas as pd

sys.path.append("../covid19_inference")

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


owd = data_retrieval.OWD(True)  # tests
ox = data_retrieval.OxCGRT(True)  # interv

France = data_retrieval.countries.France(True)  # cases France
Germany = data_retrieval.RKI(True)  # cases Germany

for country in countries:
    # Check if folder exists and create
    if not os.path.exists(path + "/" + country):
        os.makedirs(path + "/" + country)
        print(f"Created folder {path}/{country}")

    # Testings
    owd.get_new("tests", country=country, data_begin=begin, data_end=end).to_csv(
        path + f"/{country}/total_tests.csv", date_format="%d.%m.%y"
    )

    # New cases per agegroup
    new_cases = pd.DataFrame()
    for age_group in age_groups[country]:
        new_cases[age_group] = globals()[country].get_new(
            "confirmed", data_begin=begin, data_end=end, age_group=age_group
        )
    new_cases.to_csv(path + f"/{country}/new_cases.csv", date_format="%d.%m.%y")

    # Interventions
    interventions = pd.DataFrame()
    for policy in policies:
        interventions[policy] = ox.get_time_data(
            policy=policy, country=country, data_begin=begin, data_end=end,
        )
    interventions.to_csv(path + f"/{country}/interventions.csv", date_format="%d.%m.%y")
