import torch
import torch.nn.functional as F


def pairwise_average_similarity_matrix(tensors):
    """
    Compute the pairwise average cosine similarity along the last dimension
    for a list of tensors with the same shape.

    Args:
        tensors (list of torch.Tensor): A list of tensors of the same shape.

    Returns:
        torch.Tensor: A 2D tensor of shape (len(tensors), len(tensors)) containing
                      the pairwise average cosine similarities.
    """
    num_tensors = len(tensors)
    # Initialize a 2D tensor to store pairwise average similarities
    similarity_matrix = torch.zeros((num_tensors, num_tensors))

    # Step 1: Normalize each tensor along the last dimension
    normalized_tensors = [F.normalize(tensor, p=2, dim=-1) for tensor in tensors]

    # Step 2: Compute pairwise average similarities
    for i in range(num_tensors):
        for j in range(i, num_tensors):
            # Compute cosine similarity along the last dimension
            cosine_similarity = (normalized_tensors[i] * normalized_tensors[j]).sum(
                dim=-1
            )
            average_similarity = cosine_similarity.mean().item()

            # Populate the similarity matrix (it's symmetric)
            similarity_matrix[i, j] = average_similarity
            similarity_matrix[j, i] = average_similarity

    return similarity_matrix
