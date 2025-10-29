import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

TensorOrBoolMask = Optional[torch.Tensor]

class TempMHA(nn.Module):
    """
    Multi-head attention with a temperature parameter.
    Mirrors the PyTorch tutorial structure: q_proj, k_proj, v_proj, out_proj.

    - Temperature T scales the attention logits as softmax((QK^T)/(sqrt(d_k)*T)).
    - All trainable weights are identical in shape/semantics to nn.MultiheadAttention
      (split into q/k/v projections here), so we can load state_dict from an MHA.
    """
    def __init__(
        self,
        embed_dim: int, # dmodel
        num_heads: int, # head
        dropout: float = 0.0,
        batch_first: bool = False, # Use case: true. 
        bias: bool = True,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        # Tutorial-style projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias, **factory_kwargs)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True, **factory_kwargs)

    def _shape(self, x: torch.Tensor, B: int, L: int) -> torch.Tensor:
        # (B,L,E) -> (B, num_heads, L, head_dim)
        return x.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        *,
        temperature: float = 1.0,
        attn_mask: TensorOrBoolMask = None,          # bool or additive mask (broadcastable to (B*num_heads, Lq, Lk))
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, Lk) bool
        is_causal: Optional[bool] = None,
        need_weights: bool = False,                  # returns average over heads if True
        average_attn_weights: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Shapes:
          if batch_first=True:  query: (B, Lq, E), key/value: (B, Lk, Ek/Ev)
          else:                 query: (Lq, B, E), key/value: (Lk, B, Ek/Ev)
        """
        if key is None:  key = query
        if value is None: value = key

        if not self.batch_first:
            # (L,B,E) -> (B,L,E)
            query, key, value = query.transpose(0,1), key.transpose(0,1), value.transpose(0,1)

        B, Lq, _ = query.shape
        _, Lk, _ = key.shape

        # Linear projections
        q = self.q_proj(query)   # (B,Lq,E)
        k = self.k_proj(key)     # (B,Lk,E)
        v = self.v_proj(value)   # (B,Lk,E)

        # Reshape to (B,H,L,hd)
        q = self._shape(q, B, Lq)
        k = self._shape(k, B, Lk)
        v = self._shape(v, B, Lk)

        # Convert key_padding_mask to additive mask for SDPA if provided
        # SDPA supports boolean attn_mask; combine masks if both provided.
        sdpa_mask = attn_mask
        if key_padding_mask is not None:
            # key_padding_mask: (B, Lk) True = PAD (to be masked)
            # Expand to (B,1,1,Lk) then broadcast
            kpm = key_padding_mask[:, None, None, :]  # (B,1,1,Lk)
            if sdpa_mask is None:
                sdpa_mask = kpm
            else:
                # Combine: if either says "mask", we mask.
                # Convert existing mask to bool if necessary.
                if sdpa_mask.dtype != torch.bool:
                    sdpa_mask = sdpa_mask > 0
                sdpa_mask = sdpa_mask | kpm

        # SDPA expects (B*H, Lq, D) style; but it also supports (B,H,L,D) directly.
        # Use PyTorch's fused path with scale=1/temperature (on top of internal 1/sqrt(d)).
        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=sdpa_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=bool(is_causal) if is_causal is not None else False,
            scale=(1.0 / float(temperature))
        )  # (B,H,Lq,hd)

        # Merge heads
        attn = attn.transpose(1, 2).reshape(B, Lq, self.embed_dim)
        out = self.out_proj(attn)  # (B,Lq,E)

        if not self.batch_first:
            out = out.transpose(0,1)

        if need_weights:
            # If weights are needed, recompute attention weights cheaply:
            # Re-run the score computation in a memory-safe way to extract softmax(QK^T/ (sqrt(d)*T))
            # NOTE: This returns average over heads by default (like torch.nn.MultiheadAttention)
            # Shapes below assume batch_first
            qh = self._shape(self.q_proj(query if self.batch_first else query.transpose(0,1)), B, Lq)
            kh = self._shape(self.k_proj(key if self.batch_first else key.transpose(0,1)), B, Lk)
            # (B,H,Lq,hd) x (B,H,hd,Lk) -> (B,H,Lq,Lk)
            scores = torch.matmul(qh, kh.transpose(-2, -1)) / (self.head_dim ** 0.5)
            scores = scores / float(temperature)
            if sdpa_mask is not None:
                if sdpa_mask.dtype == torch.bool:
                    scores = scores.masked_fill(sdpa_mask, float("-inf"))
                else:
                    scores = scores + sdpa_mask
            weights = scores.softmax(dim=-1)  # (B,H,Lq,Lk)
            if average_attn_weights:
                weights = weights.mean(dim=1)  # (B,Lq,Lk)
                if not self.batch_first:
                    weights = weights.transpose(0,1)  # (Lq,B,Lk) to match nn.MHA API
            else:
                # return (B,H,Lq,Lk) or (Lq,B,H,Lk)
                if not self.batch_first:
                    weights = weights.transpose(0,2).transpose(1,2)  # (Lq,B,H,Lk)
            return out, weights

        return out, None

    # ---------- Interop utilities ----------

    @classmethod
    def from_mha(cls, mha: nn.MultiheadAttention, *, temperature: float = 1.0) -> "TempMHA":
        """
        Create TempMHA from an existing nn.MultiheadAttention and copy all weights.
        """
        # Infer dims
        embed_dim = mha.embed_dim
        num_heads = mha.num_heads
        dropout = mha.dropout
        batch_first = getattr(mha, "batch_first", False)
        kdim = getattr(mha, "kdim", None)
        vdim = getattr(mha, "vdim", None)
        bias = True  # nn.MHA always has bias for in_proj/out_proj if present

        mod = cls(embed_dim, num_heads, dropout=dropout, batch_first=batch_first,
                  bias=True, kdim=kdim, vdim=vdim,
                  device=next(mha.parameters()).device, dtype=next(mha.parameters()).dtype)

        # --- Copy weights ---
        # nn.MHA packs (Q,K,V) into a single (3E, E_in) matrix (and bias).
        # Map those into our separate Linear layers.
        with torch.no_grad():
            if mha._qkv_same_embed_dim:
                in_w = mha.in_proj_weight    # (3E, Ein)
                Ein = in_w.shape[1]
                E = embed_dim
                mod.q_proj.weight.copy_(in_w[0:E, :])
                mod.k_proj.weight.copy_(in_w[E:2*E, :])
                mod.v_proj.weight.copy_(in_w[2*E:3*E, :])

                if mha.in_proj_bias is not None:
                    in_b = mha.in_proj_bias   # (3E,)
                    mod.q_proj.bias.copy_(in_b[0:E])
                    mod.k_proj.bias.copy_(in_b[E:2*E])
                    mod.v_proj.bias.copy_(in_b[2*E:3*E])
                else:
                    nn.init.zeros_(mod.q_proj.bias); nn.init.zeros_(mod.k_proj.bias); nn.init.zeros_(mod.v_proj.bias)
            else:
                # separate q_proj_weight, k_proj_weight, v_proj_weight exist
                mod.q_proj.weight.copy_(mha.q_proj_weight)
                mod.k_proj.weight.copy_(mha.k_proj_weight)
                mod.v_proj.weight.copy_(mha.v_proj_weight)
                if mha.in_proj_bias is not None:
                    E = embed_dim
                    in_b = mha.in_proj_bias
                    mod.q_proj.bias.copy_(in_b[0:E])
                    mod.k_proj.bias.copy_(in_b[E:2*E])
                    mod.v_proj.bias.copy_(in_b[2*E:3*E])
                else:
                    nn.init.zeros_(mod.q_proj.bias); nn.init.zeros_(mod.k_proj.bias); nn.init.zeros_(mod.v_proj.bias)

            # out_proj
            mod.out_proj.weight.copy_(mha.out_proj.weight)
            if mha.out_proj.bias is not None:
                mod.out_proj.bias.copy_(mha.out_proj.bias)
            else:
                nn.init.zeros_(mod.out_proj.bias)

        # store a default temperature as attribute (optional; not a Parameter)
        mod.register_buffer("_default_temperature", torch.tensor(float(temperature)), persistent=False)
        return mod

    def to_mha(self) -> nn.MultiheadAttention:
        """
        Convert this TempMHA back into nn.MultiheadAttention (weights copied).
        """
        mha = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=self.dropout_p,
            batch_first=self.batch_first,
            bias=True,
            kdim=self.kdim,
            vdim=self.vdim,
            device=next(self.parameters()).device,
            dtype=next(self.parameters()).dtype,
        )
        with torch.no_grad():
            # Pack q/k/v into in_proj_weight/in_proj_bias
            E = self.embed_dim
            Ein_q = self.q_proj.weight.shape[1]
            # in_proj_weight: (3E, Ein)
            mha.in_proj_weight.copy_(
                torch.cat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], dim=0)
            )
            mha.in_proj_bias.copy_(
                torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias], dim=0)
            )
            mha.out_proj.weight.copy_(self.out_proj.weight)
            mha.out_proj.bias.copy_(self.out_proj.bias)
        return mha


def convert_model_mha_to_temp(
    model: nn.Module,
    *,
    default_temperature: float = 1.0
) -> nn.Module:
    """
    Recursively replace every nn.MultiheadAttention in `model` with TempMHA,
    copying all weights/config. Returns the modified model (in-place).
    """
    for name, module in list(model.named_children()):
        if isinstance(module, nn.MultiheadAttention):
            setattr(model, name, TempMHA.from_mha(module, temperature=default_temperature))
        else:
            convert_model_mha_to_temp(module, default_temperature=default_temperature)
    return model