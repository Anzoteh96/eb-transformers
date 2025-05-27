import torch
import tqdm
from custom_transformer import MyMultiHeadAttention
from linformer import LinformerSelfAttention
# import avg_dot_product

class EBTransformer(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        factory_kwargs = {"device": args.device, "dtype": args.dtype}
        # We augment each input with constant 1 dimension. Otherwise, for dinput=1
        # layernorm completely kills the magnitude of the input.
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
        assert args.att_activ in ["softmax", "relu", "sigmoid", "linformer", "linear", "linear_relu", "linear_sigmoid", "linear_gelu"], \
            "Activations for attention other than softmax, sigmoid, ReLU and linformer are not supported"
            
        if args.att_activ in ["relu", "sigmoid", "linear", "linear_relu", "linear_sigmoid", "linear_gelu"]:
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
        return None

    def forward(self, inputs, num_padding: int = 0, store_activations: bool = False):
        """
            Args:
                inputs: B x N x d_in
                num_padding: number of padding tokens (if any)
                store_activations: whether to store activations for each layer (linear probe purpose)
        """
        # Add dummy dimension to help avoid killing input's magnitude by the LayerNorm.
        # inputs is BxTxd_in. inputs_aux is BxTx(d_in+1).
        
        self.num_different_weights = self.args.weight_share if self.args.weight_share > 0 else self.args.layers
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
            if self.args.att_activ == "linformer":
                tmp = self.self_attn[idx](tmp)
            else:
                tmp, _ = self.self_attn[idx](
                    tmp, tmp, tmp, attn_mask=self.mask, need_weights=False
                )
            
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
        # To ensure that we only prediction nonnegative values, we add a final relu step.
        # Note that ReLU can turn bad if your network start with all negative values (or get into that at one point),
        # let's do GELU instead.
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
        out = self.forward(inputs, num_padding)
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
        _, activations = self.forward(inputs, 0, True)
        return {
            "activations": activations["activations"][layer],
            "mlp_step": activations["mlp_step"][layer],
            "attn_step": activations["attn_step"][layer]
        }

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
