import torch
import tqdm
from .custom_transformer import MyMultiHeadAttention
from linformer import LinformerSelfAttention
from .temp_mha import TempMHA
# import .avg_dot_product

class EBTransformer(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        factory_kwargs = {"device": args.device, "dtype": args.dtype}
        # We augment each input with constant 1 dimension. Otherwise, for dinput=1
        # layernorm completely kills the magnitude of the input.
        if "one_hot" in args and args.one_hot is not None and args.one_hot > 0:
            assert(args.dinput == 1), "One-hot encoding only works for dinput = 1"
            self.embed = torch.nn.Embedding(args.one_hot, args.dmodel, **factory_kwargs)
        else:
            self.embed = torch.nn.Linear(args.dinput + 1, args.dmodel, **factory_kwargs)
        # Standard transformers have layernorms without weight sharing.
        # So we want to incorporate that while making sure that previous transformers (layernorms w weight sharing) still works well.
        no_prenorm = "no_prenorm" in self.args and self.args.no_prenorm 
        no_postnorm = "no_postnorm" in self.args and self.args.no_postnorm

        if args.norm_share:
            if not no_prenorm:
                self.norm = torch.nn.LayerNorm(args.dmodel, **factory_kwargs)
            if not no_postnorm:
                self.norm2 = torch.nn.LayerNorm(args.dmodel, **factory_kwargs)
        else:
            num_norms = args.weight_share if args.weight_share > 0 else args.layers
            if not no_prenorm:
                self.norm = torch.nn.ModuleList(
                    [
                        torch.nn.LayerNorm(args.dmodel, **factory_kwargs)
                        for _ in range(num_norms)
                    ]
                )
            if not no_postnorm:
                self.norm2 = torch.nn.ModuleList(
                    [
                        torch.nn.LayerNorm(args.dmodel, **factory_kwargs)
                        for _ in range(num_norms)
                    ]
                )
        self.attn_only = ("attn_only" in args) and args.attn_only
        

        # Separating between weight sharing and no weight sharing.
        # Weight share = 0 means all layers are different. 
        self.num_different_weights = args.weight_share if args.weight_share > 0 else args.layers
        if args.weight_share == 0:
            args.weight_share = args.layers
        

        # If weight sharing is N > 0, then we have N of the different weights.
        # TODO: check how to do it for the case we want to have MLP for one of the two layers. 
        if not self.attn_only:
            # This is for the case we want to keep MLP. 
            self.linear1 = torch.nn.ModuleList(
                [
                    torch.nn.Linear(args.dmodel, 4 * args.dmodel, **factory_kwargs)
                    for _ in range(self.num_different_weights)
                ]
            )
            # self.dropout = torch.nn.Dropout(args.dropout);
            self.linear2 = torch.nn.ModuleList(
                [
                    torch.nn.Linear(4 * args.dmodel, args.dmodel, **factory_kwargs)
                    for _ in range(self.num_different_weights)
                ]
            )

            self.activation = (
                torch.nn.modules.GELU()
                if args.activation == "gelu"
                else torch.nn.modules.ReLU()
            )
        # Here we distinguish between weight tieing. 
        if "att_activ" not in args:
            args.att_activ = "softmax"
        activ_list = ["softmax", "relu", "sigmoid", "linformer", "linear", "linear_relu", 
                      "linear_sigmoid", "linear_sigmoid_normalize", "linear_gelu", "linear_softmax", "fla"]
        assert args.att_activ in activ_list, \
            "Activations for attention other than softmax, sigmoid, ReLU and linformer are not supported"
            
        if args.att_activ in ["relu", "sigmoid", "linear", "linear_relu", "linear_sigmoid", 
                              "linear_sigmoid_normalize", "linear_gelu", "linear_softmax"]:
            self.self_attn = torch.nn.ModuleList(
                [
                    MyMultiHeadAttention(
                        args.dmodel, args.dmodel, args.dmodel, args.dmodel, 
                        args.heads, activation = args.att_activ, **factory_kwargs,
                    )
                    for _ in range(self.num_different_weights)
                ]
            )
        elif args.att_activ == "linformer":
            self.self_attn = torch.nn.ModuleList(
                [
                    LinformerSelfAttention(
                        dim = args.dmodel,
                        seq_len = args.seqlen,
                        heads = args.heads,
                        k = args.dmodel,
                        one_kv_head = False,
                        share_kv = False
                    )
                    for _ in range(self.num_different_weights)
                ]
            ).to(args.device)
        elif args.att_activ == "fla": # Flash linear attention
            from fla.layers import MultiScaleRetention
            self.self_attn = torch.nn.ModuleList(
                [
                    MultiScaleRetention(hidden_size = args.dmodel,num_heads = args.heads)
                    for _ in range(self.num_different_weights)
                ]
            ).to(args.device)
        else:
            # For softmax we just use SDPA. 
            self.self_attn = torch.nn.ModuleList(
                [
                    torch.nn.MultiheadAttention(
                        args.dmodel,
                        args.heads,  # dropout=args.dropout,
                        batch_first=True,
                        **factory_kwargs,
                    )
                    for _ in range(self.num_different_weights)
                ]
            )
        

        # Enable causal processing
        if False:
            tmp = torch.full((args.seqlen, args.seqlen), 1, dtype=torch.uint8)
            self.mask = torch.triu(tmp, diagonal=1).to(
                args.device
            )  # replace diagonal and below with zeros
        else:
            self.mask = None

        if args.decoding_layer_norm:
            self.norm3 = torch.nn.LayerNorm(args.dmodel, **factory_kwargs)

        self.decoder = torch.nn.Linear(args.dmodel, args.dinput, **factory_kwargs)
        self._init_params()
        return None
    def _init_params(self):
        # Initialization trick from the "Attention is all you need" paper.
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
        return None

    def forward(self, inputs, weights = None, num_padding: int = 0, store_activations: bool = False):
        """
            Args:
                inputs: B x N x d_in
                weight: optional weights for each input (for weighted MSE loss and weighted attention), 
                # must be in the form B x N x N or N x N. 
                num_padding: number of padding tokens (if any)
                store_activations: whether to store activations for each layer (linear probe purpose)
        """
        # inputs is BxTxd_in. inputs_aux is BxTx(d_in+1).

        self.num_different_weights = self.args.weight_share if self.args.weight_share > 0 else self.args.layers
        # Can we do one-hot encoding?
        if "one_hot" in self.args and self.args.one_hot is not None and self.args.one_hot > 0:
            # Here, we encode everything as one-hot, where we assume clip all inputs to dinput - 1. 
            inputs_clamp = torch.clamp(inputs, min = 0, max = self.args.one_hot - 1)
            emb = self.embed(inputs_clamp.long()).reshape(inputs.shape[0], inputs.shape[1], -1)
        else:
            ones = torch.ones(inputs.shape[0], inputs.shape[1], 1).to(self.args.device)
            inputs_aux = torch.cat([inputs, ones], dim=2)
            if num_padding > 0: 
                pad_zero = torch.zeros(inputs_aux.shape[0], num_padding, inputs.shape[2]).to(self.args.device)
                pad_one = -torch.ones(inputs_aux.shape[0], num_padding, 1).to(self.args.device)
                pad_space = torch.cat([pad_zero, pad_one], dim = 2)
                inputs_aux = torch.cat([inputs_aux, pad_space], dim = 1)
            emb = self.embed(inputs_aux)
            # We need to add padding as "empty space", which will be the form (1, 0). 

        no_prenorm = "no_prenorm" in self.args and self.args.no_prenorm 
        no_postnorm = "no_postnorm" in self.args and self.args.no_postnorm

        # Next, we store activations. 
        if store_activations: 
            all_activations = {
                "activations": [emb],
                "mlp_step": [None],
                "attn_step": [None],
                "prenorm_step": [None],
                "postnorm_step": [None],
                "attn_pre_step": [None],
                #"out": [self.decoder(emb)],
            }

        for i in range(self.args.layers):
            idx = (
                int((i * self.num_different_weights) // self.args.layers)
            )
            if not no_prenorm:
                if self.args.norm_share:
                    tmp = self.norm(emb)
                else:
                    tmp = self.norm[idx](emb)
            else:
                tmp = emb
            if store_activations:
                all_activations["prenorm_step"].append(tmp)
            
            # Implement Self-Attention module
            if self.args.att_activ == "linformer":
                tmp = self.self_attn[idx](tmp)
            elif self.args.att_activ == "fla":
                tmp,_,_ = self.self_attn[idx](tmp)
            elif "temperature" in self.args and self.args.temperature is not None:
                tmp, _ = self.self_attn[idx](
                    tmp, tmp, tmp, attn_mask=self.mask, need_weights=False, temperature = self.args.temperature
                )
            else:
                if weights is not None and isinstance(self.self_attn[idx], torch.nn.MultiheadAttention):
                    assert self.mask is None, "Causal attention with weights not supported yet."
                    log_weights = torch.log(weights + 1e-20) # Should have size B x N x N
                    tmp, _ = self.self_attn[idx](
                        tmp, tmp, tmp, attn_mask=log_weights.repeat_interleave(self.args.heads, dim=0), need_weights=False
                    )
                else:
                    tmp, _ = self.self_attn[idx](
                        tmp, tmp, tmp, attn_mask=self.mask, need_weights=False
                    )
            if store_activations:
                all_activations["attn_pre_step"].append(tmp)
            
            # tmp = self.dropout(tmp);
            ## WARNING: do not use emb += .... since that op is inplace
            emb = emb + tmp * self.args.step
            # Implement FF module
            if not no_postnorm:
                if self.args.norm_share:
                    tmp = self.norm2(emb)
                else:
                    tmp = self.norm2[idx](emb)
            # Check if we need to incorporate MLP. 
            if store_activations:
                all_activations["postnorm_step"].append(tmp)
            
            attn_only = self.args.attn_only if "attn_only" in self.args else self.attn_only
            if not attn_only:
                tmp_ff1 = self.linear1[idx](tmp)
                tmp_ff2 = self.activation(tmp_ff1)
                tmp_ff3 = self.linear2[idx](tmp_ff2)
            else:
                tmp_ff3 = tmp
            emb = emb + tmp_ff3 * self.args.step

            if store_activations:
                all_activations["activations"].append(emb)
                all_activations["mlp_step"].append(tmp_ff3 * self.args.step)
                all_activations["attn_step"].append(tmp * self.args.step)

        if self.args.decoding_layer_norm:
            emb = self.norm3(emb)
        
        # We need to remove padding from the output.
        if num_padding > 0:
            emb = emb[:, :-num_padding, :]

        out = self.decoder(emb)
        # For Poisson channel, to ensure that we only predict nonnegative values, we add a final relu step. 
        # Note that ReLU can turn bad if your network start with all negative values (or get into that at one point),
        # let's do GELU instead.
        channel = "poisson" if "channel" not in self.args else self.args.channel
        if channel == "poisson":
            gelu = torch.nn.GELU()
            out = gelu(out)
        if store_activations:
            return out, all_activations 
        else:
            return out

    def eval_loss(self, inputs, labels, num_padding: int = 0):
        """
        Args:
            inputs: B x N x d_in
            labels: B x N x d_in
            num_padding: number of padding tokens (if any)
        """
        out = self.forward(inputs, num_padding = num_padding)
        loss = ((out - labels) ** 2).sum() / inputs.numel()
        return loss

    def get_activations(self, inputs):
        ones = torch.ones(inputs.shape[0], inputs.shape[1], 1).to(self.args.device)
        inputs_aux = torch.cat([inputs, ones], dim=2)
        emb = self.embed(inputs_aux)
        activations = []
        steps = []
        for i in range(self.args.layers):
            idx = (
                int(i * self.args.weight_share // self.args.layers)
                if self.args.weight_share > 0
                else i
            )
            if "norm_share" not in self.args or self.args.norm_share:
                tmp = self.norm(emb)
            else:
                tmp = self.norm[idx](emb)
            tmp, _ = self.self_attn[idx](
                tmp, tmp, tmp, attn_mask=self.mask, need_weights=False
            )

            emb = emb + tmp * self.args.step
            # Implement FF module
            if "norm_share" not in self.args or self.args.norm_share:
                tmp = self.norm2(emb)
            else:
                tmp = self.norm2[idx](emb)
            tmp_ff1 = self.linear1[idx](tmp)
            tmp_ff2 = self.activation(tmp_ff1)
            tmp_ff3 = self.linear2[idx](tmp_ff2)
            emb = emb + tmp_ff3 * self.args.step
            steps.append(tmp_ff3 * self.args.step)
            activations.append(emb)
        return {"activations": activations, "steps": steps}

    def get_layer_activation(self, inputs, layer):
        self.eval()
        _, activations = self.forward(inputs, num_padding = 0, store_activations = True)
        return {
            "activations": activations["activations"][layer],
            "mlp_step": activations["mlp_step"][layer],
            "attn_step": activations["attn_step"][layer],
            "attn_pre_step": activations["attn_pre_step"][layer],
            "prenorm_step": activations["prenorm_step"][layer],
            "postnorm_step": activations["postnorm_step"][layer]
        }
    
    # TODO: fix this. 
    def get_avg_dot_product_of_intermediates(self, inputs):
        """
        return an ordered list of n names of collected features,
        as well as an nxn tensor of their average pairwise cosine similaritis
        in the given batch.
        Order of names is deterministic
        """
        self.eval()
        # Add dummy dimension to help avoid killing input's magnitude by the LayerNorm.
        # inputs is BxTxd_in. inputs_aux is BxTx(d_in+1).
        ones = torch.ones(inputs.shape[0], inputs.shape[1], 1).to(self.args.device)
        inputs_aux = torch.cat([inputs, ones], dim=2)
        emb = self.embed(inputs_aux)
        names = ["activation_0"]
        tensors = [emb]
        for i in range(self.args.layers):
            idx = (
                int(i * self.args.weight_share // self.args.layers)
                if self.args.weight_share > 0
                else i
            )
            if "norm_share" not in self.args or self.args.norm_share:
                tmp = self.norm(emb)
            else:
                tmp = self.norm[idx](emb)
            tmp, _ = self.self_attn[idx](
                tmp, tmp, tmp, attn_mask=self.mask, need_weights=False
            )

            names.append(f"attn_{i + 1}")
            tensors.append(tmp * self.args.step)

            # tmp = self.dropout(tmp);
            ## WARNING: do not use emb += .... since that op is inplace
            emb = emb + tmp * self.args.step
            attn_step = tmp * self.args.step
            # Implement FF module
            if "norm_share" not in self.args or self.args.norm_share:
                tmp = self.norm2(emb)
            else:
                tmp = self.norm2[idx](emb)
            tmp_ff1 = self.linear1[idx](tmp)
            tmp_ff2 = self.activation(tmp_ff1)
            tmp_ff3 = self.linear2[idx](tmp_ff2)
            mlp_step = tmp_ff3 * self.args.step

            names.append(f"mlp_{i + 1}")
            tensors.append(tmp_ff3 * self.args.step)

            emb = emb + tmp_ff3 * self.args.step
            activation = emb
        if self.args.decoding_layer_norm:
            emb = self.norm3(emb)
        out = self.decoder(emb)

        names.append(f"out")
        tensors.append(out)

        similarity_matrix = avg_dot_product.pairwise_average_similarity_matrix(tensors)

        return similarity_matrix, names


class EBTransformerTruncate(EBTransformer):
    def __init__(self, base_model, device, num_layers_to_keep):
        self.args = base_model.args 
        self.args.device = device
        self.args.layers = num_layers_to_keep
        super(EBTransformerTruncate, self).__init__(self.args)

        # Next, we copy the weights from the base model, up until num_layers_to_keep.
        self.copy_weights(base_model)
        
        # Keep only the decoder's gradient.
        for param in self.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = True
    
    def copy_weights(self, base_model):
        self.embed.load_state_dict(base_model.embed.state_dict())
        if self.args.norm_share:
            if "no_prenorm" not in self.args or not self.args.no_prenorm:
                self.norm.load_state_dict(base_model.norm.state_dict())
            if "no_postnorm" not in self.args or not self.args.no_postnorm:
                self.norm2.load_state_dict(base_model.norm2.state_dict())
        else:
            if "no_prenorm" not in self.args or not self.args.no_prenorm:
                for i in range(self.num_different_weights):
                    self.norm[i].load_state_dict(base_model.norm[i].state_dict())
            if "no_postnorm" not in self.args or not self.args.no_postnorm:
                for i in range(self.num_different_weights):
                    self.norm2[i].load_state_dict(base_model.norm2[i].state_dict())
        if not self.attn_only:
            for i in range(self.num_different_weights):
                self.linear1[i].load_state_dict(base_model.linear1[i].state_dict())
                self.linear2[i].load_state_dict(base_model.linear2[i].state_dict())
        for i in range(self.num_different_weights):
            self.self_attn[i].load_state_dict(base_model.self_attn[i].state_dict())
        if self.args.decoding_layer_norm:
            self.norm3.load_state_dict(base_model.norm3.state_dict())
        self.decoder.load_state_dict(base_model.decoder.state_dict())
        return None
    