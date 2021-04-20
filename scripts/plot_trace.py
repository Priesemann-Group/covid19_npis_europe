# ------------------------------------------------------------------------------ #
# Plots a give trace
# Uses command line parameters for more configurations!
# Use `python ./plot_trace.py -h` for usage!
# Runtime dependent on number of distributions/timeseries in trace!
#
# @Author:        Sebastian B. Mohr
# @Created:       2020-12-18 14:40:45
# @Last Modified: 2021-01-27 12:54:39
# ------------------------------------------------------------------------------ #

# Get trace fp
import argparse, os, textwrap, sys, logging

abspath = os.path.abspath(sys.argv[0])
dname = os.path.dirname(abspath)
os.chdir(dname)
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
    "file", metavar="f", type=str, help="Path to stored trace.",
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
    help="Plot the chains with separated  colors (only working for ts yet)",
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
# modelParams, trace = covid19_npis.utils.load_trace_zarr(args.file)
modelParams, trace = covid19_npis.utils.load_trace(args.file)
modelParams._R_interval_time = 5

# Create model and sample state from modelParams
this_model = covid19_npis.model.main_model(modelParams)
_, sample_state = pm.evaluate_model_transformed(this_model, sample_shape=(3,))

# Make modelParams global.. dirty hack may change in future
modelParams._make_global()

# ------------------------------------------------------------------------------ #
# Get all distributions/timeseries
# ------------------------------------------------------------------------------ #

skip_dist_ts = [
    "E_0_t",
    "new_E_t",
    "reporting_delay_kernel",
    "new_E_t_delayed",
    "total_tests",
    "deaths",
    "testing_delay",
]


def check_for_dist_or_ts(this_model, sample_state_dict):
    ts = []
    dists = []
    for key, item in sample_state_dict.items():
        if key.replace(f"{this_model.name}|", "") in skip_dist_ts:
            continue
        try:
            if "time" in item.shape_label:
                ts.append(key.replace(f"{this_model.name}|", ""))
            else:
                dists.append(key.replace(f"{this_model.name}|", ""))
        except:
            continue
    return ts, dists


# Get all default distributions and timesries
all_ts, all_dists = check_for_dist_or_ts(this_model, sample_state.deterministics)
log.info("Plotting may take some time! Go ahead and grab a coffee or two.")
print(
    r"""
        ..
      ..  ..
            ..
             ..
            ..
           ..
         ..
##       ..    ####
##.............##  ##
##.............##   ##
##.............## ##
##.............###
 ##...........##
  #############
  #############
#################"""
)
log.info(f"Timeseries plots: {all_ts}")
log.info(f"Distribution plots: {all_dists}")

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
pbar = tqdm(
    total=len(args.distributions) + len(args.timeseries),
    desc="Creating plots",
    position=0,
)
# Timeseries
ts_axes = {}
for ts_name in args.timeseries:
    pbar.set_description(f"Creating plots [{ts_name}]")

    # Set observed data for plotting
    if ts_name == "new_E_t" or ts_name == "positive_tests" or ts_name == "positive_tests_modulated":
        observed = modelParams.pos_tests_dataframe
    elif (ts_name == "total_tests_compact") and (
        modelParams.data_summary["files"]["/tests.csv"]
    ):
        observed = modelParams.total_tests_dataframe
    elif (ts_name == "deaths_compact" or ts_name == "deaths") and (
        modelParams.data_summary["files"]["/deaths.csv"]
    ):
        observed = modelParams.deaths_dataframe
    else:
        observed = None

    ts_axes[ts_name] = covid19_npis.plot.timeseries(
        trace,
        sample_state=sample_state,
        key=ts_name,
        plot_chain_separated=args.chains_seperated,
        observed=observed,
        dir_save=args.folder,
    )

    pbar.update(1)


# Distributions
dist_fig = {}
dist_axes = {}
for dist_name in args.distributions:
    pbar.set_description(f"Creating plots [{dist_name[0:3]}]")
    dist_axes[dist_name] = covid19_npis.plot.distribution(
        trace, sample_state=sample_state, key=dist_name, dir_save=args.folder,
    )
    pbar.update(1)


pbar.close()


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
