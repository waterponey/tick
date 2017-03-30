
import numpy as np
import matplotlib.pyplot as plt

from tick.simulation import SimuLogReg, weights_sparse_gauss
from tick.optim.model import ModelLogReg

n_samples, n_features = 2000, 50
weights0 = weights_sparse_gauss(n_weights=n_features, nnz=10)
intercept0 = 1.
X, y = SimuLogReg(weights0, intercept=intercept0, seed=123,
                  n_samples=n_samples, verbose=False).simulate()

model = ModelLogReg(fit_intercept=True).fit(X, y)

coeffs0 = np.concatenate([weights0, [intercept0]])

plt.figure(figsize=(9, 3))
plt.subplot(1, 2, 1)
plt.stem(model.grad(coeffs0))
plt.title(r"$\nabla f(\mathrm{coeffs0})$", fontsize=16)
plt.subplot(1, 2, 2)
plt.stem(model.grad(np.ones(model.n_coeffs)))
plt.title(r"$\nabla f(\mathrm{coeffs1})$", fontsize=16)
