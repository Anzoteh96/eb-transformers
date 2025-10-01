import pickle
import matplotlib.pyplot as plt

with open('/Users/nicolascannella/Repos/eb-transformers/mse_outeb_2025_09_18-21_43_J8x.pkl_0_2__0.0', 'rb') as f:
    data = pickle.load(f)

print("Keys in output:", data.keys())
print("MSEs:", data['mses'])
print("Normalized MSEs:", data['norm_mses'])
print("Runtime (seconds):", data['time'])

plt.plot(data['mses'], label='MSE')
plt.plot(data['norm_mses'], label='Normalized MSE')
plt.xlabel('Seed/Batch Index')
plt.ylabel('MSE')
plt.title('Model MSEs') 
plt.legend()
plt.show()