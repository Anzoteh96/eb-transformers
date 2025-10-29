import pickle

filename = 'mse_outeb_2025_09_18-21_43_04v.pkl_0_2__0.0'  # Replace with your actual file path

with open(filename, 'rb') as f:
    data = pickle.load(f)

print("Type of object:", type(data))
if isinstance(data, dict):
    print("Keys:", data.keys())
    for k, v in data.items():
        print(f"{k}: type={type(v)}; sample={str(v)[:100]}")
else:
    print("Object contents:", str(data)[:500])