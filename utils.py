import numpy as np
import torch
import pickle
import io
from eb_transformer import EBTransformer

# This class allows us to unpickle Torch objects on the CPU. 
class TorchCpuUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

# This function loads a model dict, either on CPU or CUDA.
def load_model_dict(filename, device):
    with open(filename, 'rb') as file:
        if device == 'cpu':
            return TorchCpuUnpickler(file).load()
        elif device == 'cuda':
            return pickle.load(file)
        else:
            raise ValueError('Unknown device')

def convert_tensor_to_bincount(arr: torch.Tensor):
    """
        Args: 
            arr: a Tensor of size B x N x D
        Returns:
            arr_truncate: tensor of B x C x 1, where C = max(arr) + 1
            weights: a Tensor of size B x C, where C = max(arr) + 1
    """
    # We want to do batched bincount
    B, N, D = arr.shape 
    device = arr.device
    assert D == 1, "Only support D=1 for now"
    arr_flat = arr.squeeze(-1)  # B x N
    #print("1")
    max_val = int(arr.max().item()) + 1
    # from IPython import embed; embed()
    #print(arr.shape)
    weights_zeros = torch.zeros((B, max_val), dtype=torch.long).to(device)
    weights = weights_zeros.scatter_add_(1, arr_flat.long(), torch.ones_like(arr_flat, dtype=torch.long))
    arr_truncate = torch.arange(max_val).reshape(1, -1, 1).repeat(repeats = (B, 1, 1)).float().to(device)
    return arr_truncate, weights

# Here, we want to take a transformer and "convert" it to inputs weighted by frequency. 
def model_input_bincounts(model, inputs: torch.Tensor):
    """
        Args: 
            model: a transformer model that takes input of size B x N x D
            arr: a Tensor of size B x N x D
        Returns:
            output: the output of the model on the weighted inputs
    """
    B, N, D = inputs.shape
    assert D == 1, "Only support D=1 for now"
    inputs_truncate, weights = convert_tensor_to_bincount(inputs)  # B x C x 1, B x C
    # Next, we want to expand weights in a way amenable to the model. 

    weights_attnmask = weights.unsqueeze(-2).expand(-1, weights.shape[-1], -1) # B x C x C
    outputs_truncate = model(inputs_truncate, weights=weights_attnmask) # B x C x output_dim
    outputs = outputs_truncate[torch.arange(B).unsqueeze(1), inputs.reshape(B, N).long()]

    return outputs

# Test out and see if the setup above makes sense. 
if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    model_all = load_model_dict("eb_hscan_selectv3/T24r_2w.pkl", device)
    model = model_all["model"]
    model.args.device = device
    labels = torch.rand(40, 1000, 1) * 100
    inputs = torch.poisson(labels).to(device)
    #outputs0 = model(inputs.float())
    outputs1 = model_input_bincounts(model, inputs.float())
    #print(torch.allclose(outputs0, outputs1, atol=1e-5))
