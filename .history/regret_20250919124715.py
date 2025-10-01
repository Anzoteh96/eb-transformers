import pickle
import pandas as pd
import torch
from reducer_helper import mse_to_regret

# Load your MSE pickle file
data = torch.load('/Users/nicolascannella/Downloads/2kwarmup_debug_files/eb_2025_09_18-21_43_wrq.pkl', map_location=torch.device('cpu'), weights_only=False)

# Suppose your data contains MSEs for different models/baselines
# Example: data = {'mses': [...], 'norm_mses': [...], 'args': ..., ...}

# Convert to DataFrame (customize columns as needed)
mses_df = pd.DataFrame({
    'Model': ['your_model'] * len(data['mses']),
    'MSE': data['mses'],
    'Norm_MSE': data['norm_mses'],
    # Add other columns if needed
})

# Compute regret (assuming 'Model' is your args_col and 'bayes' is your baseline key)
regret_df = mse_to_regret(mses_df, args_col='Model', bayes_key='bayes')

print(regret_df)