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

# Plot
for layer, grads in layer_grads.items():
    plt.plot(grads, label=layer)
plt.yscale("log")
plt.xlabel("Step")
plt.ylabel("Grad norm")
plt.title("Per-layer Gradient Norms")
plt.legend(fontsize='small', ncol=2)
plt.tight_layout()
plt.show()