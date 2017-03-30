import numpy as np
import matplotlib.pyplot as plt
from tick.optim.prox import ProxL1

plt.figure(figsize=(20, 4))

x = 0.5 * np.random.randn(50)
a, b = x.min() - 1e-1, x.max() + 1e-1

proxs = [
    ProxL1(strength=0.),
    ProxL1(strength=3e-1),
    ProxL1(strength=3e-1, range=(10, 40)),
    ProxL1(strength=3e-1, positive=True),
    ProxL1(strength=3e-1, range=(10, 40), positive=True),
]

names = [
    "original vector",
    "prox",
    "prox with range=(10, 40)",
    "prox with positive=True",
    "range=(10, 40) and positive=True",
]

for i, (prox, name) in enumerate(zip(proxs, names)):
    plt.subplot(1, 5, i + 1)
    plt.stem(prox.call(x))
    plt.title(name)
    plt.xlim((-1, 51))
    plt.ylim((a, b))

plt.tight_layout()