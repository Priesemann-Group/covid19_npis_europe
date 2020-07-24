import numpy as np
import pandas as pd
import datetime


def simple_new_I(factor):
    """
        Generates a simple test dataset with fixed 
        R_t = R

        Returns
        -------
        DataFrame
    """

    def f(t, I_t, R_t):
        """
        Function for a simple SI model

        Parameters
        ----------
        t: number
            current timestep
        I_t: array
            I for every age group
        R_t: array 2d
            Reproduction matrix
        """
        return R_t @ I_t

    # 4 Age groups
    R_0 = np.diag([1, 2, 3, 2]) * factor + np.random.random(size=[4, 4]) * 0.25
    I_0 = np.array([10, 10, 10, 10])

    t = np.arange(start=0, stop=50)
    dates = pd.date_range(
        start=datetime.datetime(2020, 1, 3),
        end=datetime.datetime(2020, 1, 3) + datetime.timedelta(len(t) - 1),
    )

    I_t = [I_0]
    R_t = [R_0] * len(t)

    # ------------------------------------------------------------------------------ #
    # Iteration
    # ------------------------------------------------------------------------------ #
    for i in range(len(t) - 1):  # -1 because we append to an array
        I_n = f(i, I_t[i], R_t[i])

        # Only append to I_t because R_t gets constructed beforehand
        I_t.append(I_n)

    # timestamp
    ts = str(datetime.datetime.now().timestamp())

    # Prep data for return
    I_t = np.array(I_t)
    df = pd.DataFrame()
    df["date"] = dates
    df[(ts, "a0-10")] = I_t[:, 0]
    df[(ts, "a10-20")] = I_t[:, 1]
    df[(ts, "a20-30")] = I_t[:, 2]
    df[(ts, "a30-99")] = I_t[:, 3]

    df = df.set_index("date")
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["country", "age_group"])

    return df
