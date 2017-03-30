import numpy as np
import matplotlib.pyplot as plt
from mlpp.optim.prox import ProxL1, ProxElasticNet, ProxL2Sq, \
    ProxPositive, ProxSortedL1, ProxTV, ProxZero

x = np.random.randn(50)
a, b = x.min() - 1e-1, x.max() + 1e-1
s = 0.4

proxs = [
    ProxZero(),
    ProxPositive(),
    ProxL2Sq(strength=s),
    ProxL1(strength=s),
    ProxElasticNet(strength=s, ratio=0.5),
    ProxSortedL1(strength=s),
    ProxTV(strength=s)
]

plt.figure(figsize=(16, 8))
plt.subplot(2, 4, 1)
plt.stem(x)
plt.title("original vector", fontsize=16)
plt.xlim((-1, 51))
plt.ylim((a, b))

for i, prox in enumerate(proxs):
    plt.subplot(2, 4, i + 2)
    plt.stem(prox.call(x))
    plt.title(prox.name, fontsize=16)
    plt.xlim((-1, 51))
    plt.ylim((a, b))

plt.tight_layout()
