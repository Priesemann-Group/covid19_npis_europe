# ------------------------------------------------------------------------------ #
# Main entry point to run our model with data from German bundesländer.
# Uses roughtly the same structure as described on the getting started page.
#
# Runtime: ~8h?
#
# ------------------------------------------------------------------------------ #

import argparse, os, textwrap, sys, logging, datetime, time


# ------------------------------------------------------------------------------ #
# ARGUMENTS
# ------------------------------------------------------------------------------ #
parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=textwrap.dedent(
        """\
        Model with German bundesländer dataset
        --------------------------------------
            Runs our model with the german dataset, this dataset
            has to be generated first!

        """
    ),
)


def file_path(string):
    if os.path.isdir(os.path.dirname(string)):
        return string
    else:
        raise NotADirectoryError(
            f"Please create the folder '{os.path.dirname(string)}'!"
        )


# Output file
parser.add_argument(
    "-o",
    "--output",
    dest="fp_trace",
    type=file_path,
    default=f'./traces/{datetime.datetime.now().strftime("%y_%m_%d_%H")}',
    help="Filename for the created trace. (Default is timestamp)",
)

# Create plots?
parser.add_argument(
    "-p",
    "--plots",
    dest="plots",
    type=bool,
    default=True,
    help="Should the plots get generated from the trace automaticly? (Default is True)",
)


# Sampling
parser.add_argument(
    "--samples",
    dest="num_samples",
    default=60,
    help="Number of samples for the sampling (Default is 60)",
)

parser.add_argument(
    "--burn_in",
    dest="num_burn_in",
    default=100,
    help="Number of burn in steps for the sampling (Default is 100)",
)


# Debug
def log_level(string):
    levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
    if string.upper() in levels:
        return string.upper()
    else:
        raise NotALogLevel(
            f"Please choose one of the following Log levels: '{levels}'!"
        )


parser.add_argument(
    "--verbosity",
    type=log_level,
    default="INFO",
    help="Log level string (Default is 'INFO')",
)


args = parser.parse_args()
logging.getLogger("parso.python.diff").disabled = True

# Log setup
import pymc4 as pm

sys.path.append("../")
import covid19_npis


log = logging.getLogger()
log.setLevel(args.verbosity)  # Set log level
covid19_npis.utils.setup_colored_logs()


# ------------------------------------------------------------------------------ #
# 1. Create or generate a dataset
# ------------------------------------------------------------------------------ #
# A data set for this model run is created with the
# data_generators/Germany_bundesländer.py script!
# We just check if there is some data in the spezific folders here.

if not os.path.isdir("../data/Germany_bundesländer/"):
    raise DataError(
        f"Please generate the dataset before running the script'{os.path.dirname(string)}'!"
    )


# ------------------------------------------------------------------------------ #
# 2. Load dataset
# ------------------------------------------------------------------------------ #

modelParams = covid19_npis.ModelParams.from_folder("../data/Germany_bundesländer/")


# ------------------------------------------------------------------------------ #
# 3. Generate model with dataset
# ------------------------------------------------------------------------------ #

this_model = covid19_npis.model.main_model(modelParams)


# ------------------------------------------------------------------------------ #
# 4. Sampling
# ------------------------------------------------------------------------------ #

begin_time = time.time()
log.info("start")
num_chains = 4

trace_tuning, trace = pm.sample(
    this_model,
    num_samples=int(args.num_samples),
    num_samples_binning=10,
    burn_in_min=10,
    burn_in=int(args.num_burn_in),
    use_auto_batching=False,
    num_chains=num_chains,
    xla=False,
    initial_step_size=0.00001,
    ratio_tuning_epochs=1.3,
    max_tree_depth=4,
    decay_rate=0.75,
    target_accept_prob=0.75,
    # num_steps_between_results = 9,
    #    state=pm.evaluate_model_transformed(this_model)[1]
    # sampler_type="nuts",
)

end_time = time.time()
log.info("running time: {:.1f}s".format(end_time - begin_time))

# We also Sample the prior for the kde in the plots (optional)
"""trace_prior = pm.sample_prior_predictive(
    this_model, sample_shape=(1000,), use_auto_batching=False
)

# Save trace
name, fpath = covid19_npis.utils.save_trace(
    trace,
    modelParams,
    fpath=os.path.dirname(args.fp_trace),
    name=os.path.basename(args.fp_trace),
    trace_prior=trace_prior,
)


# ------------------------------------------------------------------------------ #
# 5. Plotting
# ------------------------------------------------------------------------------ #

if args.plots:
    # Run plotting script
    path = os.path.abspath(f"{fpath}/{name}")
    os.system(f"python plot_trace.py {path}")
else:
    log.info("Plotting skipped!")
"""
