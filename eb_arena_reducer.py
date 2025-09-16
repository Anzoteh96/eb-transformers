import re
import matplotlib.pyplot as plt
from collections import defaultdict

logfile = "eb_2025_09_12-19_47_I6x.log"
layer_grads = defaultdict(list)

with open(logfile, "r") as f:
    for line in f:
        m = re.match(r"Layer: (.*), Grad norm: ([0-9.eE+-]+)", line)
        if m:
            layer, grad = m.group(1), float(m.group(2))
            layer_grads[layer].append(grad)

layers = sorted(layer_grads.keys())
n_layers = len(layers)
fig, axs = plt.subplots(n_layers, 1, figsize=(8, 2 * n_layers), sharex=True)

if n_layers == 1:
    axs = [axs]

for i, layer in enumerate(layers):
    axs[i].plot(layer_grads[layer])
    axs[i].set_yscale("log")
    axs[i].set_ylabel("Grad norm")
    axs[i].set_title(layer)
axs[-1].set_xlabel("Step")
plt.tight_layout()
plt.show()