import datetime
import numpy as np
import pandas as pd

import logging

log = logging.getLogger(__name__)


class ModelParams:
    """ 
        Class for all model parameters is used to among others add data and 
        distrbution names/dimensions to the model.
        Is also used by the plotting routines as references to distribution names etc.

        Distribution params can be changed by overwriting the defaults

        .. code-block::

            params = ModelParams()
            params.distributions["I_0"]["name"] = "my new fancy name"

        This class also contains the data used for fitting. `dataframe` is the original
        dataframe. `data_tensor` is a tensor in the correct shape (time x countries x age)
        with values replace by nans when no data is available.

        Parameters
        ----------
        data: pd.DataFrame
            DataFrame with country/age group multicolumn and datetime index
    """

    def __init__(
        self, data, min_offset_sim_data=20, minimal_daily_cases=40, dtype="float32"
    ):

        # Data object
        self._dtype = dtype
        self._min_offset_sim_data = min_offset_sim_data
        self._minimal_daily_cases = minimal_daily_cases
        self.dataframe = data

        # Configs for distribution
        self.distributions = self.__get_default_dist_config()

    @property
    def dataframe(self):
        return self._df

    @dataframe.setter
    def dataframe(self, df):
        # set dataframe
        self._df = df

        # Config for input data
        self._data_summary = self.__get_default_data_config(self.dataframe)

        # set data tensor, replaces values smaller than 40 by nans.
        data_tensor = (
            self.dataframe.to_numpy()
            .astype(self.dtype)
            .reshape((-1, len(self.countries), len(self.age_groups)))
        )
        i_data_begin_list = []
        for c in range(data_tensor.shape[1]):
            mask = np.sum(data_tensor[:, c, :], axis=-1) > self._minimal_daily_cases
            i_data_begin = np.min(np.nonzero(mask)[0])
            i_data_begin_list.append(i_data_begin)
        i_data_begin_list = np.array(i_data_begin_list)
        i_data_begin_list = np.maximum(i_data_begin_list, self._min_offset_sim_data)
        self._indices_begin_data = i_data_begin_list
        for i in i_data_begin_list:
            data_tensor[:i] = np.nan
        self._data_tensor = data_tensor

    @property
    def min_offset_sim_data(self):
        return self._min_offset_sim_data

    @property
    def data_tensor(self):
        """
        Tensor of input dataframe i.e. daily new cases for countries/regions
        and age groups.
        |shape| time, country, agegroup 
        """
        return self._data_tensor

    @property
    def dtype(self):
        return self._dtype

    @property
    def indices_begin_sim(self):
        return self._indices_begin_data

    @property
    def data_summary(self):
        return self._data_summary

    @property
    def age_groups(self):
        return self.data_summary["age_groups"]

    @property
    def countries(self):
        return self.data_summary["countries"]

    @property
    def num_age_groups(self):
        return len(self.data_summary["age_groups"])

    @property
    def num_countries(self):
        return len(self.data_summary["countries"])

    def get_distribution_by_name(self, name):
        for dist, value in self.distributions.items():
            if value["name"] == name:
                this_dist = dist
        return self.distributions[this_dist]

    def __get_default_dist_config(self):
        """
            Returns default distributions dictionaries.
        """
        distributions = {
            "I_0_diff_base": {
                "name": "I_0_diff_base",
                "long_name": "Initial infectious difference to inferred ones",
                "shape": (self.num_countries, self.num_age_groups),
                "shape_label": ("country", "age_group"),
                "math": "I_0",
            },
            "R": {
                "name": "R",
                "long_name": "Reproduction number",
                "shape": (self.num_countries, self.num_age_groups),
                "shape_label": ("country", "age_group"),
                "math": "R",
            },
            "C": {
                "name": "C",
                "long_name": "Contact matrix",
                "shape": (self.num_countries, self.num_age_groups, self.num_age_groups),
                "shape_label": ("country", "age_group_i", "age_group_j"),
                "math": "C",
            },
            "sigma": {
                "name": "sigma",
                "long_name": "likelihood scale",
                "shape": (1),
                "shape_label": ("likelihood scale"),
                "math": r"\sigma",
            },
            "g_mu": {
                "name": "g_mu",
                "long_name": "long_name g_mu",
                "shape": (1),
                "shape_label": ("g_mu"),
                "math": r"g_{\mu}",
            },
            "g_theta": {
                "name": "g_theta",
                "long_name": "long_name g_theta",
                "shape": (1),
                "shape_label": ("g_theta"),
                "math": r"g_{\theta}",
            },
            "new_cases": {
                "name": "new_cases",
                "long_name": "Daily new infectious cases",
                "shape": self.data_tensor.shape,
                "shape_label": ("time", "country", "age_group"),
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
