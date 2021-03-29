import sys
import logging
import time
import itertools
import os
import datetime

import pandas as pd
import pymc4 as pm
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution, enable_eager_execution # tensorflow 2.+ has eager execution turned on by default and the supposed function 'tf.config.run_functions_eagerly()' seems ot have no effect. this one works, though
import numpy as np
import time
import os
import matplotlib.pyplot as plt

sys.path.append("../")
import covid19_npis

""" # Debugging and other snippets
"""

# Logs setup
log = logging.getLogger()
# Needed to set logging level before importing other modules
# log.setLevel(logging.DEBUG)
covid19_npis.utils.setup_colored_logs()
logging.getLogger("parso.python.diff").disabled = True
# Mute Tensorflow warnings ...
# logging.getLogger("tensorflow").setLevel(logging.ERROR)

# For eventual debugging:
# tf.config.run_functions_eagerly(True)
# disable_eager_execution()
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

def define_model():
    countries = [
        "Germany",
        "Germany_1age",
        "Belgium",
        # "Belgium_1age",
        # #    "Czechia",
        # "Denmark",
        # "Finland",
        # "Greece",
        # # # "Italy",
        # # # "Netherlands",
        "Portugal",
        # "Portugal_1age",
        # # # "Romania",
        # # # "Spain",
        # # "Sweden",
        # # "Switzerland",
    ]

    c = [
        covid19_npis.data.Country(f"../data/coverage_db/{country}",)
        for country in countries
    ]

    # Construct our modelParams from the data.
    modelParams = covid19_npis.ModelParams(countries=c, minimal_daily_deaths=1)

    # Define our model
    this_model = covid19_npis.model.model.main_model(modelParams)

    # if tf.executing_eagerly(): # does only work, when eager exec is turned on
    #     # Test shapes, should be all 3:
    #     def print_dist_shapes(st):
    #         for name, dist in itertools.chain(
    #             st.discrete_distributions.items(), st.continuous_distributions.items(),
    #         ):
    #             if dist.log_prob(st.all_values[name]).shape != (3,):
    #                 log.warning(
    #                     f"False shape: {dist.log_prob(st.all_values[name]).shape}, {name}"
    #                 )
    #         for p in st.potentials:
    #             if p.value.shape != (3,):
    #                 log.warning(f"False shape: {p.value.shape} {p.name}")
    #
    #
    #     _, sample_state = pm.evaluate_model_transformed(this_model, sample_shape=(3,))
    #     print_dist_shapes(sample_state)

    return c, modelParams, this_model

def run_model(this_model,modelParams,num_samples,num_chains):
    """ # 2. MCMC Sampling
    """

    begin_time = time.time()
    log.info("start")
    # num_chains = 6
    # num_samples = [200,1000]


    trace_tuning, trace = pm.sample(
        this_model,
        num_samples=num_samples[1],
        num_samples_binning=10,
        burn_in_min=10,
        burn_in=num_samples[0],
        use_auto_batching=False,
        num_chains=num_chains,
        xla=False,
        initial_step_size=0.00001,
        ratio_tuning_epochs=1.3,
        max_tree_depth=4,
        decay_rate=0.75,
        target_accept_prob=0.75,
        step_size_adaption_per_chain=False
        # num_steps_between_results = 9,
        #    state=pm.evaluate_model_transformed(this_model)[1]
        # sampler_type="nuts",
    )

    end_time = time.time()
    log.info("running time: {:.1f}s".format(end_time - begin_time))

    plt.figure()
    plt.plot(trace_tuning.sample_stats["step_size"][0])
    plt.figure()
    plt.plot(trace_tuning.sample_stats["lp"].T)
    plt.show(block=False)

    log.info("sampling prior")
    # We also Sample the prior for the kde in the plots (optional)
    trace_prior = pm.sample_prior_predictive(
        this_model, sample_shape=(500,), use_auto_batching=False
    )

    fpath = f'./traces/{datetime.datetime.now().strftime("%y_%m_%d_%H_%M")}'

    # Save our traces for the plotting script
    # store = covid19_npis.utils.save_trace_zarr(
    store = covid19_npis.utils.save_trace(
        trace, modelParams, store=fpath, trace_prior=trace_prior,
    )
    log.info(f"data saved to {fpath}")
    # os.system(f"python plot_trace.py {fpath}")
