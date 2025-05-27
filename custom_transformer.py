import torch
from torch import Tensor
import torch.nn as nn
import math 
import torch.nn.functional as F
from typing import Callable, Optional, TYPE_CHECKING, Union

# Can we define like a ReLU version of Attn function? 

def linear_attention(query, key, value, attn_mask=None, dropout_p=0.0, phi = None, need_weights=True):
    """
        Args:
            query: B1, B2, M, D1 
            key: B1, B2, N, D1
            value: B1, B2, N, D2
            phi: function
        returns:
            attn_output: B1, B2, M, D2
    """
    if phi == None:
        phi = lambda x: x # This is what we ended up using. 
    N = key.shape[-2]
    # from IPython import embed; embed() 
    S = torch.matmul(phi(key).transpose(-2, -1), value) / N # B1, B2, D1, D2
    attn_output = torch.matmul(phi(query), S) # B1, B2, M, D2
    attn_output = torch.dropout(attn_output, dropout_p, train=True)
    if need_weights:
        return attn_output, S
    else:
        return attn_output, None
    return attn_output

# To do so, we need to first do the following ReLU operation. 
def my_scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False, activation_func = "softmax", need_weights=True) -> torch.Tensor:
    """
        Args:
            query: B1, B2, M, D1 
            key: B1, B2, N, D1
            value: B1, B2, N, D2
        returns:
            attn_output: B1, B2, M, D2
    """
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)
    
    B1, B2, M, D1 = query.shape 
    B1, B2, N, _ = key.shape # Last dim should be N. 
    # Simplified version: can we just add the things together? 

    attn_weight = torch.compile(torch.bmm)(query.reshape(B1 * B2, M, D1), (key.transpose(-2, -1)).reshape(B1 * B2, D1, N)) * scale_factor
    attn_weight = attn_weight.reshape(B1, B2, M, N)
    attn_weight += attn_bias
    assert activation_func in ["softmax", "relu", "sigmoid"]
    
    if activation_func == "softmax":
        attn_weight = torch.softmax(attn_weight, dim=-1)
    elif activation_func == "sigmoid":
        attn_weight = F.sigmoid(attn_weight) / M
    else:
        attn_weight = F.relu(attn_weight)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    # attn_weight: B1, B2, M, N
    # value: B1, B2, N, D2
    # S(t) version is just multiplying 
    if need_weights:
        return attn_weight @ value, attn_weight 
    else:
        return attn_weight @ value, None

class MyMultiHeadAttention(nn.Module):
    """
    Computes multi-head attention. Supports nested or padded tensors.

    Args:
        E_q (int): Size of embedding dim for query
        E_k (int): Size of embedding dim for key
        E_v (int): Size of embedding dim for value
        E_total (int): Total embedding dim of combined heads post input projection. Each head
            has dim E_total // nheads
        nheads (int): Number of heads
        dropout (float, optional): Dropout probability. Default: 0.0
        bias (bool, optional): Whether to add bias to input projection. Default: True
    """

    def __init__(
        self,
        E_q: int,
        E_k: int,
        E_v: int,
        E_total: int,
        nheads: int,
        dropout: float = 0.0,
        activation: str = "softmax", # The default. 
        bias=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.nheads = nheads
        self.dropout = dropout
        self._qkv_same_embed_dim = E_q == E_k and E_q == E_v
        if self._qkv_same_embed_dim:
            self.packed_proj = nn.Linear(E_q, E_total * 3, bias=bias, **factory_kwargs)
        else:
            self.q_proj = nn.Linear(E_q, E_total, bias=bias, **factory_kwargs)
            self.k_proj = nn.Linear(E_k, E_total, bias=bias, **factory_kwargs)
            self.v_proj = nn.Linear(E_v, E_total, bias=bias, **factory_kwargs)
        E_out = E_q
        self.out_proj = nn.Linear(E_total, E_out, bias=bias, **factory_kwargs)
        assert E_total % nheads == 0, "Embedding dim is not divisible by nheads"
        self.E_head = E_total // nheads
        self.bias = bias
        self.activation = activation

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask=None,
        is_causal=False,
        need_weights = True
    ) -> torch.Tensor:
        """
        Forward pass; runs the following process:
            1. Apply input projection
            2. Split heads and prepare for SDPA
            3. Run SDPA
            4. Apply output projection

        Args:
            query (torch.Tensor): query of shape (``N``, ``L_q``, ``E_qk``)
            key (torch.Tensor): key of shape (``N``, ``L_kv``, ``E_qk``)
            value (torch.Tensor): value of shape (``N``, ``L_kv``, ``E_v``)
            attn_mask (torch.Tensor, optional): attention mask of shape (``N``, ``L_q``, ``L_kv``) to pass to SDPA. Default: None
            is_causal (bool, optional): Whether to apply causal mask. Default: False

        Returns:
            attn_output (torch.Tensor): output of shape (N, L_t, E_q)
        """
        scale_factor = 1 / math.sqrt(query.size(-1))
        # Step 1. Apply input projection
        if self._qkv_same_embed_dim:
            if query is key and key is value:
                result = self.packed_proj(query)
                query, key, value = torch.chunk(result, 3, dim=-1)
            else:
                q_weight, k_weight, v_weight = torch.chunk(
                    self.packed_proj.weight, 3, dim=0
                )
                if self.bias:
                    q_bias, k_bias, v_bias = torch.chunk(
                        self.packed_proj.bias, 3, dim=0
                    )
                else:
                    q_bias, k_bias, v_bias = None, None, None
                query, key, value = (
                    F.linear(query, q_weight, q_bias),
                    F.linear(key, k_weight, k_bias),
                    F.linear(value, v_weight, v_bias),
                )

        else:
            query = self.q_proj(query)
            key = self.k_proj(key)
            value = self.v_proj(value)

        # Step 2. Split heads and prepare for SDPA
        # reshape query, key, value to separate by head
        # (N, L_t, E_total) -> (N, L_t, nheads, E_head) -> (N, nheads, L_t, E_head)
        query = query.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        key = key.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        value = value.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)

        # Step 3. Run my_SDPA
        # Optionally, apply the "linear" version. 
        # (N, nheads, L_t, E_head)
        if self.activation in ["linear", "linear_relu", "linear_sigmoid", "linear_gelu"]:
            if self.activation == "linear":
                phi = lambda x: x 
            elif self.activation == "linear_relu":
                phi = F.relu
            elif self.activation == "linear_sigmoid":
                phi = F.sigmoid
            elif self.activation == "linear_gelu":
                phi = F.gelu
            attn_output, attn_weight = linear_attention(
                query, key, value, attn_mask=attn_mask, dropout_p=self.dropout, phi = phi, need_weights = need_weights
            )
        else:
            attn_output, attn_weight = my_scaled_dot_product_attention(
                query, key, value, dropout_p=self.dropout, is_causal=is_causal, activation_func = self.activation, need_weights = need_weights
          )
        
        
        # (N, nheads, L_t, E_head) -> (N, L_t, nheads, E_head) -> (N, L_t, E_total)
        attn_output = attn_output.transpose(1, 2).flatten(-2)

        # Step 4. Apply output projection
        # (N, L_t, E_total) -> (N, L_t, E_out)
        attn_output = self.out_proj(attn_output)

        if need_weights:
            return attn_output, attn_weight 
        else:
            return attn_output, None