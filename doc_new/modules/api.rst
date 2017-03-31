
.. _api:

=============
API Reference
=============

This is the full class and function references of tick. Please look at
the modules documentation cited below for more examples and use cases,
since direct class and function API is not enough for understanding their uses.


.. _api-base:

:mod:`tick.base`: Base classes and tools
========================================

This module contains all base classes and functions of tick.
The objects in this module are useful for development only, and are not
intended for end-users.

.. automodule:: tick.base
   :no-members:
   :no-inherited-members:

Classes
-------
.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: class.rst

   base.Base
   base.TimeFunction

Functions
---------
.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: function.rst

   base.actual_kwargs



.. _api-inference:

:mod:`tick.inference`: Inference classes
========================================

This module contains all classes giving inference tools, intended for end-users.

**User guide:** See the :ref:`inference` section for further details.

.. automodule:: tick.inference
   :no-members:
   :no-inherited-members:

Base classes
------------
.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: class.rst

   inference.base.LearnerOptim
   inference.base.LearnerGLM

Classes
-------
.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: class.rst

   inference.LogisticRegression
   inference.CoxRegression
   inference.HawkesExpKern
   inference.HawkesSumExpKern


.. _api-optim-model:

:mod:`tick.optim.model`: Models classes
=======================================

This module contains classes giving computational informations about the models available
in tick.

**User guide:** See the :ref:`optim-model` section for further details.

.. automodule:: tick.optim.model
   :no-members:
   :no-inherited-members:

Base classes
------------
.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: class.rst

   optim.model.base.Model
   optim.model.base.ModelFirstOrder
   optim.model.base.ModelGeneralizedLinear
   optim.model.base.ModelGeneralizedLinearWithIntercepts
   optim.model.base.ModelLabelsFeatures
   optim.model.base.ModelLipschitz
   optim.model.base.ModelSecondOrder
   optim.model.base.ModelSelfConcordant

Classes
-------
.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: class.rst

   optim.model.ModelLinReg
   optim.model.ModelLinRegWithIntercepts
   optim.model.ModelLogReg
   optim.model.ModelPoisReg
   optim.model.ModelCoxRegPartialLik
   optim.model.ModelHawkesFixedExpKernLeastSq
   optim.model.ModelHawkesFixedExpKernLogLik
   optim.model.ModelHawkesFixedSumExpKernLeastSq


:mod:`tick.optim.prox`: Proximal operators classes
==================================================

This module contains all the proximal operators available in tick.

**User guide:** See the :ref:`optim-prox` section for further details.

Base classes
------------
.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: class.rst

   optim.prox.base.Prox

Classes
-------
.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: class.rst

   optim.prox.ProxZero
   optim.prox.ProxL1
   optim.prox.ProxL1w
   optim.prox.ProxElasticNet
   optim.prox.ProxL2Sq
   optim.prox.ProxMulti
   optim.prox.ProxNuclear
   optim.prox.ProxPositive
   optim.prox.ProxSortedL1
   optim.prox.ProxTV


.. _api-optim-solver:

:mod:`tick.optim.solver`: Solver classes
========================================

This module contains all the solvers available in tick.

**User guide:** See the :ref:`optim-solver` section for further details.

Base classes
------------
.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: class.rst

   optim.solver.base.Solver
   optim.solver.base.SolverFirstOrder
   optim.solver.base.SolverSto
   optim.solver.base.SolverFirstOrderSto
   optim.history.History

Classes
-------
.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: class.rst

   optim.solver.GD
   optim.solver.AGD
   optim.solver.BFGS
   optim.solver.GFB
   optim.solver.SCPG
   optim.solver.SGD
   optim.solver.SDCA
   optim.solver.SVRG


.. _api-plot:

:mod:`tick.plot`: Plotting utilities
====================================

This module contains some utilities functions for plotting

**User guide:** See the :ref:`plot` section for further details.

Functions
---------
.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: function.rst

   plot.stem
   plot.stems
   plot.plot_history


.. _api-preprocessing:

:mod:`tick.preprocessing`: Preprocessing utilities
==================================================

This module contains some utilities functions for preprocessing of data.

**User guide:** See the :ref:`preprocessing` section for further details.

Classes
-------
.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: class.rst

   preprocessing.FeaturesBinarizer


.. _api-simulation:

:mod:`tick.simulation`: Simulation classes and fuctions
=======================================================

This module contains all simulation tools available in tick.

**User guide:** See the :ref:`simulation` section for further details.

Base classes
------------
.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: class.rst

   simulation.base.Simu
   simulation.base.SimuWithFeatures
   simulation.base.SimuPointProcess

Classes
-------
.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: class.rst

   simulation.SimuLinReg
   simulation.SimuLogReg
   simulation.SimuPoisReg
   simulation.SimuCoxReg
   simulation.SimuPoissonProcess
   simulation.SimuInhomogeneousPoisson
   simulation.SimuHawkes
   simulation.SimuHawkesExpKernels
   simulation.SimuHawkesSumExpKernels
   simulation.HawkesKernelExp
   simulation.HawkesKernelSumExp
   simulation.HawkesKernelPowerLaw
   simulation.HawkesKernelTimeFunc

Functions
---------
.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: function.rst

   simulation.features_normal_cov_uniform
   simulation.features_normal_cov_toeplitz
   simulation.weights_sparse_exp
   simulation.weights_sparse_gauss
