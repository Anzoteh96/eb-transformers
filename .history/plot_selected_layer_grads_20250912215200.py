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
# Select first, middle, and last 2 layers (adjust as needed)
selected = [layers[0], layers[len(layers)//2], layers[-2], layers[-1]]
fig, axs = plt.subplots(len(selected), 1, figsize=(8, 2 * len(selected)), sharex=True)
if len(selected) == 1:
    axs = [axs]
for i, layer in enumerate(selected):
    axs[i].plot(layer_grads[layer])
    axs[i].set_yscale("log")
    axs[i].set_ylabel("Grad norm")
    axs[i].set_title(layer)
axs[-1].set_xlabel("Step")
plt.tight_layout()
plt.show()
