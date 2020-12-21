# ------------------------------------------------------------------------------ #
# Plot a give trace as command line parameter
# Use `python ./plot_trace.py -h` for usage!
# Runtime dependent on number of distributions/timeseries in trace!
#
# @Author:        Sebastian B. Mohr
# @Created:       2020-12-18 14:40:45
# @Last Modified: 2020-12-21 15:23:31
# ------------------------------------------------------------------------------ #

# Get trace fp
import argparse, os, textwrap, sys, logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pymc4 as pm
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

try:
    import covid19_npis
except Exception as e:
    sys.path.append("../")
    import covid19_npis

log = logging.getLogger(__name__)


# ------------------------------------------------------------------------------ #
# Default/console arguments
# ------------------------------------------------------------------------------ #
def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(f"Please create the folder '{string}'!")


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=textwrap.dedent(
        """\
        Plots trace from our model trace
        --------------------------------
            Create plot(s) for distributions and
            timeseries and saves them as pdf into
            a folder.
        """
    ),
)
parser.add_argument(
    "file", metavar="f", type=str, help="Pickled trace to plot.",
)
# Output folder
parser.add_argument(
    "-o",
    "--output",
    dest="folder",
    type=dir_path,
    default="./figures",
    help="Output folder for the figures.",
)
parser.add_argument(
    "-c",
    "--chains_seperated",
    dest="chains_seperated",
    type=bool,
    default=False,
    help="Plot the chains with seperated  colors (only working for ts yet)",
)
# Optional distributions list
parser.add_argument(
    "-d",
    "--distributions",
    dest="distributions",
    nargs="+",
    help="Distributions variables in trace to plot (default: all saved as 'determinstics')",
    metavar="DIST",
)
# Optional timeseries list
parser.add_argument(
    "-t",
    "--timeseries",
    dest="timeseries",
    nargs="+",
    help="Timeseries variables in trace to plot (default: all saved as 'determinstics')",
    metavar="TS",
)
if len(sys.argv) < 2:
    log.error("Supply trace file!")
    parser.print_help()
    sys.exit(0)

args = parser.parse_args()


# ------------------------------------------------------------------------------ #
# Load pickled trace
# ------------------------------------------------------------------------------ #
fpath, name = os.path.split(args.file)
modelParams, trace = covid19_npis.utils.load_trace(name, fpath)

# Create model and sample state from modelParams
this_model = covid19_npis.model.main_model(modelParams)
_, sample_state = pm.evaluate_model_transformed(this_model, sample_shape=(3,))

# Make modelParams global.. dirty hack may change in future
modelParams._make_global()

# ------------------------------------------------------------------------------ #
# Get all distributions/timeseries
# ------------------------------------------------------------------------------ #
def check_for_dist_or_ts(this_model, sample_state_dict):
    ts = []
    dists = []
    for key, item in sample_state_dict.items():
        try:
            if "time" in item.shape_label:
                ts.append(key.replace(f"{this_model.name}/", ""))
            else:
                dists.append(key.replace(f"{this_model.name}/", ""))
        except:
            continue
    return ts, dists


# Get all default distributions and timesries
all_ts, all_dists = check_for_dist_or_ts(this_model, sample_state.deterministics)


if args.distributions is None:
    args.distributions = all_dists

if args.distributions == [""]:
    args.distributions = []

if args.timeseries is None:
    args.timeseries = all_ts

if args.timeseries == [""]:
    args.timeseries = []


# ------------------------------------------------------------------------------ #
# Plot given distributions/timeseries
# ------------------------------------------------------------------------------ #

# Progress bar
pbar = tqdm(total=len(args.distributions) + len(args.timeseries), desc="Creating plots")

# Distributions
dist_fig = {}
dist_axes = {}
for dist_name in args.distributions:
    pbar.set_description(f"Creating plots [{dist_name[0:3]}]")
    dist_fig[dist_name], dist_axes[dist_name] = covid19_npis.plot.distribution(
        trace, sample_state=sample_state, key=dist_name
    )
    # Save figure
    for i, fig in enumerate(dist_fig[dist_name]):
        if len(dist_fig[dist_name]) > 1:
            subname = f"_{i}"
        else:
            subname = ""
        fig.savefig(
            f"{args.folder}/dist_{dist_name}" + f"{subname}.pdf",
            dpi=300,
            transparent=True,
        )
        plt.close(fig)
    pbar.update(1)

# Timeseries
ts_fig = {}
ts_axes = {}
for ts_name in args.timeseries:
    pbar.set_description(f"Creating plots [{ts_name[0:3]}]")
    ts_fig[ts_name], ts_axes[ts_name] = covid19_npis.plot.timeseries(
        trace,
        sample_state=sample_state,
        key=ts_name,
        plot_chain_separated=args.chains_seperated,
    )
    # plot observed data into new_cases i.e. pos tests
    if ts_name == "new_E_t" or ts_name == "positive_tests":
        for i, c in enumerate(modelParams.data_summary["countries"]):
            for j, a in enumerate(modelParams.data_summary["age_groups"]):
                ts_axes[ts_name][j][i] = covid19_npis.plot.time_series._timeseries(
                    modelParams.pos_tests_dataframe.index[:],
                    modelParams.pos_tests_dataframe[c][a],
                    ax=ts_axes[ts_name][j][i],
                    alpha=0.5,
                    ls="-",
                )
    # plot observed data into total tests
    if ts_name == "total_tests_compact":
        for i, c in enumerate(modelParams.data_summary["countries"]):
            ts_axes[ts_name][i] = covid19_npis.plot.time_series._timeseries(
                modelParams.total_tests_dataframe.index[:],
                modelParams.total_tests_dataframe.xs(c, level="country", axis=1),
                ax=ts_axes[ts_name][i],
                alpha=0.5,
                ls="-",
            )

    # plot observed data into deaths
    if ts_name == "deaths_compact" or ts_name == "deaths":
        for i, c in enumerate(modelParams.data_summary["countries"]):
            ts_axes[ts_name][i] = covid19_npis.plot.time_series._timeseries(
                modelParams.deaths_dataframe.index[:],
                modelParams.deaths_dataframe.xs(c, level="country", axis=1),
                ax=ts_axes[ts_name][i],
                alpha=0.5,
                ls="-",
            )

    # Save figures
    for i, fig in enumerate(ts_fig[ts_name]):
        if len(ts_fig[ts_name]) > 1:
            subname = f"_{i}"
        else:
            subname = ""
        fig.savefig(
            f"{args.folder}/ts_{ts_name}" + f"{subname}.pdf", dpi=300, transparent=True
        )
        plt.close(fig)
    pbar.update(1)

pbar.close()
