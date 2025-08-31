import torch
import torch.nn.functional as F

def pairwise_average_similarity_matrix(tensors):
    """
    Given a list of tensors [T_0, T_1, ..., T_n], each of shape (batch, seq_len, d),
    computes an (n+1) x (n+1) matrix where entry (i, j) is the average cosine similarity
    between flattened T_i and T_j across the batch.
    """
    # Flatten each tensor to (batch, -1)
    flat = [t.reshape(t.shape[0], -1) for t in tensors]
    n = len(flat)
    sim_matrix = torch.zeros((n, n), device=flat[0].device)
    for i in range(n):
        for j in range(n):
            # Compute cosine similarity for each batch element, then average
            sim = F.cosine_similarity(flat[i], flat[j], dim=1)  # shape: (batch,)
            sim_matrix[i, j] = sim.mean()
    return sim_matrix