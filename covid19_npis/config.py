import datetime
import numpy as np
import pandas as pd


class Config(object):
    """ 
        Configuration class should be used if one wants to use the,
        plotting and data converters. Contains names and shapes with labels
        for every used distribution in the model.

        Distribution configs can be changed by overwriting the defaults

        .. code-block:: 
            config = Config()
            config.distributions["I_0"]["name"] = "my new fancy name"


        Parameters
        ----------
        data: pd.DataFrame
            DataFrame with country/age group multicolumn and datetime index
    """

    def __init__(self, data):

        # Get number of age groups and countries from data
        num_countries = len(data.columns.levels[0])
        num_age_groups = len(data.columns.levels[1])

        # Configs for distribution
        self.distributions = self.__get_default_dist_config(
            num_countries, num_age_groups
        )
        # Config for input data
        self.data = self.__get_default_data_config(data)

        # Data object
        self.df = data

    def get_data(self):
        return self.df

    def __get_default_dist_config(self, num_countries, num_age_groups):
        """
            Get default values of config dicts.
        """
        distributions = {
            "I_0": {
                "name": "I_0",
                "long_name": "Initial infectious people",
                "shape": (num_countries, num_age_groups),
                "shape_label": ("country", "age_group"),
            },
            "R": {
                "name": "R",
                "long_name": "Reproduction number",
                "shape": (num_countries, num_age_groups),
                "shape_label": ("country", "age_group"),
            },
            "C": {
                "name": "C",
                "long_name": "Contact matrix",
                "shape": (num_countries, num_age_groups, num_age_groups),
                "shape_label": ("country", "age_group_i", "age_group_j"),
            },
            "sigma": {
                "name": "sigma",
                "long_name": "likelihood scale",
                "shape": (1),
                "shape_label": ("likelihood scale"),
            },
        }

        return distributions

    def __get_default_data_config(self, df):
        data = {  # Is set on init
            "begin": df.index.min(),
            "end": df.index.max(),
            "age_groups": [],
            "countries": [],
        }

        # Create countries lookup list dynamic from data dataframe
        for i in range(len(df.columns.levels[0])):
            data["countries"].append(df.columns.levels[0][i])

        # Create age group list dynamic from data dataframe
        for i in range(len(df.columns.levels[1])):
            data["age_groups"].append(df.columns.levels[1][i])

        return data
