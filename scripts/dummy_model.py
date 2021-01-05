import sys
import logging
import time
import itertools
import os


"""
Some runtime optimizations for CPU (using tur nodes)
os.environ["OMP_NUM_THREADS"] = 32
tf.config.threading.set_inter_op_parallelism_threads(32)
tf.config.threading.set_intra_op_parallelism_threads(16)
tf.config.set_soft_device_placement(enabled)
tf.config.optimizer.set_jit(
    True
)
"""


"""
SM: I dont know what this is doing, please explain :)

os.environ["KMP_BLOCKTIME"] = "1"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
"""
import pymc4 as pm
import tensorflow as tf
import numpy as np
import time
import os

"""
SM: I dont know what this is doing, please explain :)

old_opts = tf.config.optimizer.get_experimental_options()
print(old_opts)
tf.config.optimizer.set_experimental_options(
    {
        "autoparallel_optimizer": True,
        "layout_optimizer": True,
        "loop_optimizer": True,
        "dependency_optimizer": True,
        "shape_optimizer": True,
        "function_optimizer": True,
        "constant_folding_optimizer": True,
    }
)
print(tf.config.optimizer.get_experimental_options())
"""
sys.path.append("../")
import covid19_npis

""" # Debugging and other snippets
"""

# Logs setup
log = logging.getLogger()
# Needed to set logging level before importing other modules
log.setLevel(logging.DEBUG)
covid19_npis.utils.setup_colored_logs()
logging.getLogger("parso.python.diff").disabled = True
# Mute Tensorflow warnings ...
# logging.getLogger("tensorflow").setLevel(logging.ERROR)

# For eventual debugging:
tf.config.run_functions_eagerly(True)
# tf.debugging.enable_check_numerics(stack_height_limit=50, path_length_limit=50)

if tf.executing_eagerly():
    log.warning("Running in eager mode!")

# Force CPU
# covid19_npis.utils.force_cpu_for_tensorflow()
# covid19_npis.utils.split_cpu_in_logical_devices(32)


""" # 1. Data Retrieval
    Load data for different countries/regions, for now we have to define every
    country by hand maybe we want to automatize that at some point.

    TODO: maybe we want to outsource that to a different file at some point
"""

# Load our data from csv files into our own custom data classes

countries = [
    "Germany",
    "Belgium",
    "Czechia",
    "Denmark",
    "Finland",
    "Greece",
    # "Italy",
    # "Netherlands",
    "Portugal",
    # "Romania",
    # "Spain",
    "Sweden",
    "Switzerland",
]
c = [
    covid19_npis.data.Country(f"../data/coverage_db/{country}",)
    for country in countries
]

# Construct our modelParams from the data.
modelParams = covid19_npis.ModelParams(countries=c, minimal_daily_deaths=1)


# Define our model
this_model = covid19_npis.model.model.main_model(modelParams)

# Test shapes, should be all 3:
def print_dist_shapes(st):
    for name, dist in itertools.chain(
        st.discrete_distributions.items(), st.continuous_distributions.items(),
    ):
        if dist.log_prob(st.all_values[name]).shape != (3,):
            log.warning(dist.log_prob(st.all_values[name]).shape, name)
    for p in st.potentials:
        if p.value.shape != (3,):
            log.warning(p.value.shape, p.name)


_, sample_state = pm.evaluate_model_transformed(this_model, sample_shape=(3,))
print_dist_shapes(sample_state)

""" # 2. MCMC Sampling
"""

begin_time = time.time()
log.info("start")
trace = pm.sample(
    this_model,
    num_samples=5,
    burn_in=5,
    use_auto_batching=False,
    num_chains=2,
    xla=False,
    # sampler_type="nuts",
)

end_time = time.time()
log.info("running time: {:.1f}s".format(end_time - begin_time))


# We also Sample the prior for the kde in the plots (optional)
trace_prior = pm.sample_prior_predictive(
    this_model, sample_shape=(1000,), use_auto_batching=False
)

# Save our traces for the plotting script
name, fpath = covid19_npis.utils.save_trace(
    trace, modelParams, fpath="./traces", trace_prior=trace_prior
)


# Run plotting script
"""
os.system(f"plot_trace.py {fpath}/{name}")
"""
