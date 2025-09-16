import pickle
import matplotlib.pyplot as plt

with open('./mse_out_debug/eb_2025_09_16-14_17_iHQ.pkl_0_10__0.0', 'rb') as f:
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