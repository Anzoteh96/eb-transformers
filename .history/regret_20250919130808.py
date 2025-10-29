import pickle
import matplotlib.pyplot as plt
import sys

if len(sys.argv) < 3:
    print("Usage: python regret.py <your_model_mse.pkl> <bayes_mse.pkl>")
    sys.exit(1)

model_file = sys.argv[1]
bayes_file = sys.argv[2]

with open(model_file, 'rb') as f:
    model_data = pickle.load(f)
with open(bayes_file, 'rb') as f:
    bayes_data = pickle.load(f)

# Compute regret for each batch/seed
regret = [m - b for m, b in zip(model_data['mses'], bayes_data['mses'])]
norm_regret = [m - b for m, b in zip(model_data['norm_mses'], bayes_data['norm_mses'])]

print("Regret (MSE):", regret)
print("Regret (Normalized MSE):", norm_regret)

plt.plot(regret, label='Regret (MSE)')
plt.plot(norm_regret, label='Regret (Normalized MSE)')
plt.xlabel('Seed/Batch Index')
plt.ylabel('Regret')
plt.title('Model Regret vs Bayes')
plt.legend()
plt.show()