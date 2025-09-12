import pickle
import sys

# Usage: python read_mse_output.py <output_file>
if len(sys.argv) < 2:
    print("Usage: python read_mse_output.py <output_file>")
    sys.exit(1)

output_file = sys.argv[1]

with open(output_file, 'rb') as f:
    data = pickle.load(f)

print("Keys in output:", data.keys())
print("MSEs:", data.get('mses'))
print("Normalized MSEs:", data.get('norm_mses'))
print("Runtime:", data.get('time'))
print("Model name:", data.get('mdl_name'))
print("Args:", data.get('args'))
