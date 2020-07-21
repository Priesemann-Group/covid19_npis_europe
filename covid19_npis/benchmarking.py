from pymc4.inference.sampling import build_logp_and_deterministic_functions, vectorize_logp_function, tile_init
import tensorflow as tf
from timeit import timeit
from timerit import Timerit
import tensorflow_probability as tfp

def benchmark(model, num_chains=(2,20), use_auto_batching=False, only_xla=True, iters=5, n_evals=100):
    data_dicts = []
    for nchains in num_chains:
        (
            logpfn,
            init,
            _deterministics_callback,
            deterministic_names,
            state_,
        ) = build_logp_and_deterministic_functions(
            model,
            num_chains=nchains,
            collect_reduced_log_prob=use_auto_batching,
        )

        init_state = list(init.values())
        init_keys = list(init.keys())
        if use_auto_batching:
            parallel_logpfn = vectorize_logp_function(logpfn)
            deterministics_callback = vectorize_logp_function(_deterministics_callback)
            init_state = tile_init(init_state, nchains)
        else:
            parallel_logpfn = logpfn
            deterministics_callback = _deterministics_callback
            init_state = tile_init(init_state, nchains)

        #@tf.function(autograph=False, experimental_compile=False)
        def run_chains(init=init_state):

            #@tf.function(autograph=False, experimental_compile=xla)
            def body(i, x):
                def randomize(x):
                    return x + tf.random.uniform(x.shape)*1e-3

                init_rand = list(map(randomize, init))

                return i + 1, x + parallel_logpfn(*init_rand) + tf.cast(i, 'float32')

            i = tf.constant(0)
            c = lambda i, _: tf.less(i, n_evals)

            _, result = tf.while_loop(c, body, [i, tf.zeros(nchains)], parallel_iterations=1)

            return result

        if only_xla:
            config = tfp.debugging.benchmarking.BenchmarkTfFunctionConfig(
                strategies=('function/xla',),
                hardware=('cpu', 'gpu')
            )
        else:
            config = tfp.debugging.benchmarking.default_benchmark_config()


        data_dicts.append(tfp.debugging.benchmarking.benchmark_tf_function(run_chains, iters=iters,
                                                                           print_intermediates=True,
                                                                           config=config))
    format_header = ""
    format_str = ""

    for key in data_dicts[0][0].keys():
        format_header += r"{:<20} "
        if '_time' in key:
            format_str += r"{:<20.3f} "
        else:
            format_str += r"{:<20} "

    print(format_header.format(*list(data_dicts[0][0].keys())))

    for nchains in range(len(num_chains)):
        print("Num  chains: {}".format(num_chains[nchains]))
        for vals in data_dicts[nchains]:
            print(format_str.format(*list(vals.values())))

    return data_dicts