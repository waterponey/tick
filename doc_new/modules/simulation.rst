

.. _simulation:

==========================================
:mod:`tick.simulation`: simulation toolbox
==========================================

tick provides several classes to simulate datasets.
This is particularly useful to test optimization algorithms, and to
compare the statistical properties of inference methods.

For now, tick gives simulation classes for Generalized Linear
Models, Cox regression, Poisson Processes with any intensity and
Hawkes processes. Utilities for simulation of model coefficients
(with sparsity, etc.) and utilities for features matrix simulation
are provided as well.

Contents
========

.. toctree::
   :maxdepth: 2

* :ref:`simulation-tools`
* :ref:`simulation-linear-model`
* :ref:`simulation-survival`
* :ref:`simulation-point-process`


.. _simulation-tools:

1. Simulation tools
===================

We gather in this section tools for the simulation of datasets : simulation of
model weights, simulation of a features matrix, kernels for the simulation of
Hawkes processes and time functions for the simulation of Poisson processes.

1.1 Simulation of model weights
-------------------------------

Here are functions for the simulation of model weights.

.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: function.rst

   simulation.weights_sparse_exp
   simulation.weights_sparse_gauss

**Example**

.. plot:: modules/code_samples/simulation_weights.py
    :include-source:

1.2 Simulation of a features matrix
-----------------------------------

Here are functions for the simulation of a features matrix: each simulated
vector or features is distributed as a centered Gaussian vector with
a particular covariance matrix (uniform symmetrized or toeplitz).

.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: function.rst

   simulation.features_normal_cov_uniform
   simulation.features_normal_cov_toeplitz

**Example**

.. todo::

    Insert a sample code here

1.3 Kernels for Hawkes process simulation
-----------------------------------------

Kernels are useful objects for the custom simulation of Hawkes processes.
They can be used together with the class ????

.. todo::

    Insert a sample code here

.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: class.rst

   simulation.HawkesKernelExp
   simulation.HawkesKernelSumExp
   simulation.HawkesKernelPowerLaw
   simulation.HawkesKernelTimeFunc


1.4 Time function
-----------------

A class of time function is provided for the simulation of an inhomogeneous
Poisson process.
It is then intended to be used with the class ???

.. todo::

    Insert a sample code here


.. _simulation-linear-model:

2. Linear model simulation
==========================

Simulation of several linear models can be done using the following classes.
All simulation classes simulates a features matrix :math:`\boldsymbol X` with rows :math:`x_i`
and a labels vector :math:`y` with coordinates :math:`y_i` for :math:`i=1, \ldots, n`, that
are i.i.d realizations of a random vector :math:`X` and a scalar random variable :math:`Y`.

The conditional distribution of :math:`Y | X` is :math:`\mathbb P(Y=y | X=x)`,
where :math:`\mathbb P` depends on the considered model.

=====================================  =============================================  ============================
Model                                  Distribution :math:`\mathbb P(Y=y | X=x)`      Class
=====================================  =============================================  ============================
Linear regression                      :math:`\text{Normal}(w^\top x + b, \sigma^2)`  :class:`tick.simulation.SimuLinReg`
Logistic regression                    :math:`\text{Binomial}(w^\top x + b)`          :class:`tick.simulation.SimuLogReg`
Poisson regression (identity link)     :math:`\text{Poisson}(w^\top x + b)`           :class:`tick.simulation.SimuPoisReg` with ``link="identity"``
Poisson regression (exponential link)  :math:`\text{Poisson}(e^{w^\top x + b})`       :class:`tick.simulation.SimuPoisReg` with ``link="exponential"``
=====================================  =============================================  ============================

**Example**

.. plot:: modules/code_samples/simulation_linear_model.py
    :include-source:

.. todo::

    DONNER UN PEU PLUS DE DETAILS SUR LES OPTIONS DE CES CLASSES


.. _simulation-survival:

3. Survival analysis simulation
===============================

We provide a class for the simulation of a Cox regression model with right-censoring.
This generates data in the form of i.i.d triplets :math:`(x_i, t_i, c_i)`
for :math:`i=1, \ldots, n`, where :math:`x_i \in \mathbb R^d` is a features vector,
:math:`t_i \in \mathbb R_+` is the survival time and :math:`c_i \in \{ 0, 1 \}` is the
indicator of right censoring.
Note that :math:`c_i = 1` means that :math:`t_i` is a failure time
while :math:`c_i = 0` means that :math:`t_i` is a censoring time.

.. todo::

    DESCRIBE PRECISELY THE MODEL

For now, the following class is available


===================================  ===================================
Model                                Class
===================================  ===================================
Cox regression with right-censoring  :class:`tick.simulation.SimuCoxReg`
===================================  ===================================

**Example**

.. plot:: modules/code_samples/simulation_coxreg.py
    :include-source:


.. _simulation-point-process:

4. Point process simulation
===========================

Tick has a particular focus on inference for point processes.
It therefore proposes as well tools for their simulation: for now, inhomogeneous
Poisson processes and Hawkes processes.

4.1 Poisson processes
---------------------

The following classes are available for

.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: function.rst

   simulation.SimuPoissonProcess
   simulation.SimuInhomogeneousPoisson


.. todo::

    COMPLETER ET BLABLATER

4.2 Hawkes processes
--------------------


.. todo::

    COMPLETER ET BLABLATER


.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: function.rst

   simulation.SimuHawkes
   simulation.SimuHawkesExpKernels
   simulation.SimuHawkesSumExpKernels
