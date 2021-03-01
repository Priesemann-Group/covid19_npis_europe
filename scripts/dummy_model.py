import sys
import logging
import time
import itertools
import os
import datetime
import functools


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
import tensorflow_probability as tfp
import numpy as np
import time
import os
import matplotlib.pyplot as plt

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
# log.setLevel(logging.DEBUG)
covid19_npis.utils.setup_colored_logs()
logging.getLogger("parso.python.diff").disabled = True
# Mute Tensorflow warnings ...
# logging.getLogger("tensorflow").setLevel(logging.ERROR)


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

# countries = [
#     "Germany",
#     "Belgium",
#     #    "Czechia",
#     "Denmark",
#     "Finland",
#     "Greece",
#     # "Italy",
#     # "Netherlands",
#     "Portugal",
#     # "Romania",
#     # "Spain",
#     "Sweden",
#     "Switzerland",
# ]

countries = [
    "Germany",
    # "Belgium",
    #    "Czechia",
    # "Denmark",
    # "Finland",
    # "Greece",
    # "Italy",
    # "Netherlands",
    "Portugal",
    # "Romania",
    # "Spain",
    # "Sweden",
    "Switzerland",
]

c = [
    covid19_npis.data.Country(f"../data/coverage_db/{country}",)
    for country in countries
]

# Construct our modelParams from the data.
modelParams = covid19_npis.ModelParams(countries=c, minimal_daily_deaths=1)
# modelParams = covid19_npis.ModelParams.from_folder("../data/Germany_bundesländer/")

# Define our model
this_model = covid19_npis.model.model.main_model(modelParams)

# Test shapes, should be all 3:
def print_dist_shapes(st):
    for name, dist in itertools.chain(
        st.discrete_distributions.items(), st.continuous_distributions.items(),
    ):
        if dist.log_prob(st.all_values[name]).shape != (3,):
            log.warning(
                f"False shape: {dist.log_prob(st.all_values[name]).shape}, {name}"
            )
    for p in st.potentials:
        if p.value.shape != (3,):
            log.warning(f"False shape: {p.value.shape} {p.name}")


_, sample_state = pm.evaluate_model_transformed(this_model, sample_shape=(3,))
print_dist_shapes(sample_state)

(
    posterior_approx,
    bijector,
    transformed_names,
) = covid19_npis.model.build_approximate_posterior(this_model)

sample_size = 50

(
    logpfn,
    init_random,
    _deterministics_callback,
    deterministic_names,
    state_,
) = pm.mcmc.samplers.build_logp_and_deterministic_functions(
    this_model, num_chains=sample_size, collect_reduced_log_prob=False
)


trace_loss = lambda traceable_quantities: tf.debugging.check_numerics(
    traceable_quantities.loss, f"loss not finite: {traceable_quantities.loss}"
)

# For eventual debugging:
# tf.config.run_functions_eagerly(True)
# tf.debugging.enable_check_numerics(stack_height_limit=50, path_length_limit=50)

begin = time.time()
posterior = tfp.vi.fit_surrogate_posterior(
    logpfn,
    posterior_approx,
    tf.optimizers.Adam(
        learning_rate=0.0001, epsilon=0.1, beta_1=0.9, beta_2=0.999, clipvalue=10.0
    ),
    4000,
    convergence_criterion=None,
    sample_size=sample_size,
    trainable_variables=None,
    # jit_compile=False,
    variational_loss_fn=functools.partial(
        tfp.vi.monte_carlo_variational_loss,
        discrepancy_fn=tfp.vi.kl_reverse,
        use_reparameterization=True,
    ),
    trace_fn=trace_loss,
)
print(f"Runtime: {time.time() - begin:.3f} s")


_, st = pm.evaluate_model_posterior_predictive(
    this_model, values=posterior_approx.sample(100)
)
var_names = list(st.all_values.keys()) + list(st.deterministics_values.keys())
samples = {
    k: (
        st.untransformed_values[k]
        if k in st.untransformed_values
        else (
            st.deterministics_values[k]
            if k in st.deterministics_values
            else st.transformed_values[k]
        )
    )
    for k in var_names
}


"""  # 2. MCMC Sampling
"""

begin_time = time.time()
log.info("start")
num_chains = 3

from tensorflow_probability import bijectors as tfb

init_state = posterior_approx.sample(num_chains)
init_state = [init_state[name] for name in transformed_names]
bijector_to_list = tfb.Restructure(
    [name for name in transformed_names], {name: name for name in transformed_names}
)
bijector_list = tfb.Chain([bijector_to_list, bijector, tfb.Invert(bijector_to_list)])


trace_tuning, trace = pm.sample(
    this_model,
    num_samples=500,
    num_samples_binning=10,
    burn_in_min=10,
    burn_in=500,
    use_auto_batching=False,
    num_chains=num_chains,
    xla=False,
    initial_step_size=0.001,
    ratio_tuning_epochs=1.3,
    max_tree_depth=5,
    decay_rate=0.75,
    target_accept_prob=0.75,
    step_size_adaption_per_chain=False,
    bijector=bijector_list,
    init_state=init_state
    # num_steps_between_results = 9,
    #    state=pm.evaluate_model_transformed(this_model)[1]
    # sampler_type="nuts",
)

end_time = time.time()
log.info("running time: {:.1f}s".format(end_time - begin_time))

# plt.figure()
# plt.plot(trace_tuning.sample_stats["step_size"][0])
# plt.figure()
# plt.plot(trace_tuning.sample_stats["lp"].T)
# plt.show()


# We also Sample the prior for the kde in the plots (optional)
trace_prior = pm.sample_prior_predictive(
    this_model, sample_shape=(500,), use_auto_batching=False
)

fpath = f'./traces/{datetime.datetime.now().strftime("%y_%m_%d_%H")}'

# Save our traces for the plotting script
store = covid19_npis.utils.save_trace_zarr(
    trace, modelParams, store=fpath, trace_prior=trace_prior,
)


os.system(f"python plot_trace.py {fpath}")
