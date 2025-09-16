import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt

# Usage: python inspect_prior_thetas.py <path_to_pickle>

def try_torch_load(pkl_path):
    try:
        import torch
        return torch.load(pkl_path, map_location="cpu")
    except Exception as e:
        print(f"torch.load failed: {e}\nFalling back to pickle.load...")
        return None


def cpu_pickle_load(pkl_path):
    import torch
    class CPU_Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
            return super().find_class(module, name)
    import io
    with open(pkl_path, 'rb') as f:
        return CPU_Unpickler(f).load()

def main(pkl_path):
    data = None
    # Try torch.load first (for PyTorch checkpoints)
    data = try_torch_load(pkl_path)
    if data is None:
        try:
            data = cpu_pickle_load(pkl_path)
        except Exception as e:
            print(f"cpu_pickle_load failed: {e}")
            print("Could not load the pickle file. Exiting.")
            sys.exit(1)

    print("Top-level keys/objects in pickle:", type(data), getattr(data, 'keys', lambda: [])())

    # Try to find GoofyPrior or thetas
    thetas = None
    if isinstance(data, dict):
        # Look for likely keys
        for k in data:
            if 'prior' in k.lower() or 'goofy' in k.lower():
                print(f"Found key '{k}':", type(data[k]))
                # Try to get thetas attribute or value
                obj = data[k]
                if hasattr(obj, 'thetas'):
                    thetas = getattr(obj, 'thetas')
                elif isinstance(obj, dict) and 'thetas' in obj:
                    thetas = obj['thetas']
                elif isinstance(obj, np.ndarray):
                    thetas = obj
        # If not found, try to look for 'thetas' at top level
        if thetas is None and 'thetas' in data:
            thetas = data['thetas']

    if thetas is not None:
        print("Thetas found! Shape:", np.shape(thetas))
        print("First few values:", np.array(thetas).flatten()[:10])
        plt.hist(np.array(thetas).flatten(), bins=50)
        plt.title("GoofyPrior Thetas Distribution")
        plt.xlabel("Theta value")
        plt.ylabel("Count")
        plt.show()
    else:
        print("Could not find thetas in the pickle file. Please inspect the printed keys and adapt the script if needed.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_prior_thetas.py <path_to_pickle>")
        sys.exit(1)
    main(sys.argv[1])
