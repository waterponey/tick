"""
==========================
Fit Hawkes on finance data
==========================

This example fit hawkes kernels on finance data provided by `tick-datasets`_. 
repository.

.._ tick-datasets: https://github.com/X-DataInitiative/tick-datasets

Note that pickling part is only here to avoid training this model each time 
we build the doc
"""
import os
import pickle

import numpy as np

from tick.dataset import fetch_hawkes_bund_data
from tick.inference import HawkesConditionalLaw
from tick.plot import plot_hawkes_kernel_norms

# If learner has been saved we load it directly
cache_path = "../doc/_build/plot_directive/cache_hawkes_bund_kernels.pkl"
if os.path.exists(cache_path):
    hawkes_learner = pickle.load(open(cache_path, "rb"))

else:
    timestamps_list = fetch_hawkes_bund_data()

    kernel_discretization = np.hstack((0, np.logspace(-5, 0, 50)))
    hawkes_learner = HawkesConditionalLaw(
        claw_method="log", delta_lag=0.1, min_lag=5e-4, max_lag=500,
        quad_method="log", n_quad=50, min_support=1e-4, max_support=1,
        n_threads=4)

    hawkes_learner.fit(timestamps_list)

plot_hawkes_kernel_norms(hawkes_learner,
                         node_names=["P_u", "P_d", "T_a", "T_b"])

# we save learner if we train it for the first time
if not os.path.exists(cache_path):
    pickle.dump(hawkes_learner, open(cache_path, 'wb'))
