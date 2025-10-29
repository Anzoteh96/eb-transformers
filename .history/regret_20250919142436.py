import pickle
import matplotlib.pyplot as plt
import glob
import os
import sys

if len(sys.argv) < 3:
    print("Usage: python regret.py <models_dir> <bayes_mse.pkl>")
    sys.exit(1)

models_dir = sys.argv[1]
bayes_file = sys.argv[2]

with open(bayes_file, 'rb') as f:
    bayes_data = pickle.load(f)

# Find all pickle files in the models_dir
model_files = glob.glob(os.path.join(models_dir, "*.pkl"))

for model_file in model_files:
    if os.path.isdir(model_file):
        continue
    print(f"\n=== {os.path.basename(model_file)} ===")
    with open(model_file, 'rb') as f:
        model_data = pickle.load(f)

    regret = [m - b for m, b in zip(model_data['mses'], bayes_data['mses'])]
    norm_regret = [m - b for m, b in zip(model_data['norm_mses'], bayes_data['norm_mses'])]

    print("Your Model MSE:", model_data['mses'])
    print("Bayes Model MSE:", bayes_data['mses'])
    print("Regret (MSE):", regret)
    print("Your Model Normalized MSE:", model_data['norm_mses'])
    print("Bayes Model Normalized MSE:", bayes_data['norm_mses'])
    print("Regret (Normalized MSE):", norm_regret)

    plt.plot(regret, label='Regret (MSE)')
    plt.plot(norm_regret, label='Regret (Normalized MSE)')
    plt.xlabel('Seed/Batch Index')
    plt.ylabel('Regret')
    plt.title(f'Regret: {os.path.basename(model_file)} vs Bayes')
    plt.legend()
    plt.show()