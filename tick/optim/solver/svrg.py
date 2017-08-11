# License: BSD 3 clause

from tick.optim.solver.base import SolverFirstOrderSto
from tick.optim.solver.build.solver import SVRG as _SVRG

__author__ = "Stephane Gaiffas"

# TODO: preparer methodes pour set et get attributes

variance_reduction_methods_mapper = {
    'last': _SVRG.VarianceReductionMethod_Last,
    'avg': _SVRG.VarianceReductionMethod_Average,
    'rand': _SVRG.VarianceReductionMethod_Random
}


# TODO: get_history doc

# TODO: objective doc

# TODO: set_model, set_prox doc

# TODO: solve


class SVRG(SolverFirstOrderSto):
    """
    Stochastic Variance Reduced Gradient for the minimization of objectives of
    the form

    .. math::
        \\frac 1n \\sum_{i=1}^n f_i(w) + g(w),

    where the functions :math:`f_i` have smooth gradients and :math:`g` is
    prox-capable. Function :math:`f = \\frac 1n \\sum_{i=1}^n f_i` corresponds to the
    ``model.loss`` function of the model passed with ``set_model`` and
    :math:`g` corresponds to the ``prox.value`` function of the prox passed
    with the ``set_prox`` method.
    One iteration of ``SVRG`` corresponds to the following iteration applied
    ``epoch_size`` times:

    .. math::
        w \\gets \\mathrm{prox}_{\\eta g} \\big(w - \\eta (\\nabla f_i(w) -
        \\nabla f_i(\\bar{w}) + \\nabla f(\\bar{w}) \\big)

    where :math:`i` is sampled at random (strategy depends on ``rand_type``) at
    each iteration, and where :math:`\\bar w` and :math:`\\nabla f(\\bar w)`
    are updated at the beginning of each epoch, with a strategy that depend on
    the ``variance_reduction`` parameter. The step-size :math:`\\eta` can be
    tuned with ``step``, the seed of the random number generator for generation
    of samples :math:`i` can be seeded with ``seed``. The iterations stop
    whenever tolerance ``tol`` is achieved, or ``max_iter`` epochs (namely
    ``tol``:math:`\\times` ``epoch_size``) have been made.
    The obtained solution :math:`w` is returned by the ``solve`` method, and is
    also stored in the ``solution`` attribute of the solver.

    Parameters
    ----------
    step : `float`
        Step-size parameter, the most important parameter of the solver.
        Whenever possible, this can be automatically tuned as
        ``step = 1 / model.get_lip_max()``. Otherwise, use a try-an-improve
        approach

    epoch_size : `int`, default given by model
        Epoch size, namely how many iterations are made before updating the
        variance reducing term. By default, this is automatically tuned using
        information from the model object passed through ``set_model``

    rand_type : {'unif', 'perm'}, default='unif'
        How samples are randomly selected from the data

        * if ``'unif'`` samples are uniformly drawn among all possibilities
        * if ``'perm'`` a random permutation of all possibilities is
          generated and samples are sequentially taken from it. Once all of
          them have been taken, a new random permutation is generated

    tol : `float`, default=1e-10
        The tolerance of the solver (iterations stop when the stopping
        criterion is below it)

    max_iter : `int`, default=100
        Maximum number of iterations of the solver, namely maximum number of
        epochs (by default full pass over the data, unless ``epoch_size`` has
        been modified from default)

    verbose : `bool`, default=True
        If `True`, solver verboses history, otherwise nothing is displayed,
        but history is recorded anyway

    print_every : `int`, default=1
        Print history information every time the iteration number is a
        multiple of ``print_every``. Used only is ``verbose`` is True

    record_every : `int`, default=1
        Save history information every time the iteration number is a
        multiple of ``record_every``

    variance_reduction : {'last', 'avg', 'rand'}, default='last'
        Determine what is used as phase iterate for variance reduction.

        * 'last' : the phase iterate is the last iterate of the previous epoch
        * 'avg' : the phase iterate is the average over the iterates in the past
          epoch
        * 'rand': the phase iterate is a random iterate of the previous epoch

    seed : `int`, default=-1
        BLABLA

    Attributes
    ----------
    model : `Solver`
        The model used by the solver, passed with the ``set_model`` method

    prox : `Prox`
        Proximal operator used by the solver, passed with the ``set_prox``
        method

    solution : `numpy.array`, shape=(n_coeffs,)
        BLABLA

    time_elapsed : `float`
        BLABLA

    time_end : `str`
        BLABLA

    time_start : `str`
        BLABLA

    history : `dict`-like
        BLABLA

    References
    ----------
    * Blabla
    * Blabla


    """

    def __init__(self, step: float = None, epoch_size: int = None,
                 rand_type: str = "unif", tol: float = 1e-10,
                 max_iter: int = 100, verbose: bool = True,
                 print_every: int = 1, record_every: int = 1,
                 seed: int = -1, variance_reduction: str = "last"):

        SolverFirstOrderSto.__init__(self, step, epoch_size, rand_type,
                                     tol, max_iter, verbose,
                                     print_every, record_every, seed=seed)
        step = self.step
        if step is None:
            step = 0.

        epoch_size = self.epoch_size
        if epoch_size is None:
            epoch_size = 0

        # Construct the wrapped C++ SGD solver
        self._solver = _SVRG(epoch_size, self.tol,
                             self._rand_type, step, self.seed)

        self.variance_reduction = variance_reduction

    @property
    def variance_reduction(self):
        return next((k for k, v in variance_reduction_methods_mapper.items()
                     if v == self._solver.get_variance_reduction()), None)

    @variance_reduction.setter
    def variance_reduction(self, val: str):

        if val not in variance_reduction_methods_mapper:
            raise ValueError(
                'variance_reduction should be one of "{}", got "{}".'.format(
                    ', '.join(variance_reduction_methods_mapper.keys()),
                    val))

        self._solver.set_variance_reduction(
            variance_reduction_methods_mapper[val])
