import numpy as np
import pandas as pd
import datetime
from .reproduction_number import Change_point, get_R_t


def simple_new_I(factor):
    """
        Generates a simple test dataset with fixed 
        R_t = R

        Returns
        -------
        DataFrame
    """
    N = 1e12

    def f(t, I_t, R_t, S_t):
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
        f = S_t / N

        new = f * R_t @ I_t
        return new, S_t - new

    # 4 Age groups
    R_0 = np.diag([1, 1, 1.05, 1.03]) * factor + np.random.random(size=[4, 4]) * 0.05
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
    S_t = N
    for i in range(len(t) - 1):  # -1 because we append to an array
        I_n, S_t = f(i, I_t[i], R_t[i], S_t)

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


def simple_new_I_with_R_t(factor):
    """
    Create a simple test dataset with time dependent reproduction number
    """

    N = 1e12
    R_0 = np.array([1.2, 1.3, 1.1, 1.3]) * factor
    I_0 = np.array([10, 10, 10, 10])
    cp1_1 = Change_point(alpha=0.1, gamma_max=0.5, length=5, begin=8)
    cp1_2 = Change_point(alpha=0.2, gamma_max=-0.5, length=2, begin=30)
    cp2 = Change_point(alpha=0.2, gamma_max=1, length=2, begin=25)
    cps = [cp1_1, cp1_2, cp2]

    # time
    t = np.arange(start=0, stop=50)
    dates = pd.date_range(
        start=datetime.datetime(2020, 1, 3),
        end=datetime.datetime(2020, 1, 3) + datetime.timedelta(len(t) - 1),
    )

    # Create r_t from cps
    R_t = get_R_t(t, R_0, cps)
    # Create r_t matrix
    for i in range(t.max()):
        if i == 0:
            R_m = np.array([np.diag(R_t[i])])
        else:
            R_m = np.concatenate((R_m, [np.diag(R_t[i])]), axis=0)

    def f(t, I_t, R_t, S_t):
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
        f = S_t / N

        new = f * R_t @ I_t
        return new, S_t - new

    # ------------------------------------------------------------------------------ #
    # Iteration
    # ------------------------------------------------------------------------------ #
    S_t = N
    I_t = [I_0]
    for i in range(len(t) - 1):  # -1 because we append to an array
        I_n, S_t = f(i, I_t[i], R_m[i], S_t)

        # Only append to I_t because R_t gets constructed beforehand
        I_t.append(I_n)

    # ------------------------------------------------------------------------------ #
    # Data preparation (pandas)
    # ------------------------------------------------------------------------------ #
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
