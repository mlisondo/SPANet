import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from spanet.options import Options
from spanet.network.jet_reconstruction.jet_scm_pipeline import JetSecondaryLoader
from spanet.dataset.types import Batch

tcompile = torch.compile

# --------------------------------------------------------------------------------------------------- MODULES FOR SET TRANSFORMER

# MULTIHEAD ATTENTION BLOCK
class MAB(nn.Module):
    def __init__(self, dim_Q : int, dim_K : int, dim_V : int, num_heads : int, ln : bool):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads

        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)

        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)

        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

# SELF ATTENTION BLOCK
class SAB(nn.Module):
    def __init__(self, dim_in : int, dim_out : int, num_heads : int, ln : bool):
        super().__init__()
        self.mab = MAB(
            dim_Q       = dim_in, 
            dim_K       = dim_in, 
            dim_V       = dim_out, 
            num_heads   = num_heads, 
            ln          = ln
        )

    def forward(self, X):
        return self.mab(X, X)

# INDUCED SELF ATTENTION BLOCK
class ISAB(nn.Module):
    def __init__(self, dim_in : int, dim_out : int, num_heads : int, num_inds : int, ln : bool):
        super().__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        
        self.mab0 = MAB(
            dim_Q       = dim_out, 
            dim_K       = dim_in, 
            dim_V       = dim_out, 
            num_heads   = num_heads, 
            ln          = ln
        )

        self.mab1 = MAB(
            dim_Q       = dim_in, 
            dim_K       = dim_out, 
            dim_V       = dim_out, 
            num_heads   = num_heads, 
            ln          = ln
        )

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

# POOLING BY MULTIHEAD ATTENTION
class PMA(nn.Module):
    def __init__(self, dim : int, num_heads : int, num_seeds : int, ln : bool):
        super().__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)

        self.mab = MAB(
            dim_Q       = dim, 
            dim_K       = dim, 
            dim_V       = dim, 
            num_heads   = num_heads, 
            ln          = ln
        )

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)

# --------------------------------------------------------------------------------------------------- SET TRANSFORMER

class SetTransformer(nn.Module):
    def __init__(self, dim_input : int, num_outputs : int, dim_output : int, num_inds : int, dim_hidden : int, num_heads : int, ln : bool, ISAB_layers : int, SAB_layers : int):
        super().__init__()
        self.in_proj = nn.Linear(dim_input, dim_hidden)

        isab_layer = []
        for _ in range(ISAB_layers):
            isab_layer.append(ISAB(
                dim_in      = dim_hidden, 
                dim_out     = dim_hidden, 
                num_heads   = num_heads, 
                num_inds    = num_inds, 
                ln          = ln
            )
            )
        self.isab = nn.Sequential(*isab_layer)

        self.pma = PMA(
            dim         = dim_hidden,
            num_heads   = num_heads,
            num_seeds   = num_outputs, 
            ln          = ln
        )

        sab_layer = []
        for _ in range(SAB_layers):
            sab_layer.append(SAB(
                dim_in      = dim_hidden,
                dim_out     = dim_hidden,
                num_heads   = num_heads,
                ln          = ln
            )
            )
        self.sab = nn.Sequential(*sab_layer)

        self.out = nn.Linear(dim_hidden, dim_output)

    def forward(self, X):
        # Encoder
        X = self.isab(self.in_proj(X))
        # Decoder
        X = self.sab(self.pma(X))
        return self.out(X)

# --------------------------------------------------------------------------------------------------- MOMENTUM PRIOR

class MomentumPrior(nn.Module):
    """
    Candidate-level set mixer for the momentum-only mode.
    Expects cand_tokens built by BranchSetEncoder(mode="mom"): (N, K, E)
    """
    def __init__(self, embed_dim: int, num_heads: int, SAB_layers: int, use_global_context: bool, ln: bool):
        super().__init__()

        self.sabs = nn.ModuleList([
            SAB(
                dim_in    = embed_dim,
                dim_out   = embed_dim,
                num_heads = num_heads,
                ln        = ln
            ) for _ in range(SAB_layers)
        ])

        self.use_global_context = use_global_context
        if use_global_context:
            self.global_pma = PMA(
                dim       = embed_dim,
                num_heads = num_heads,
                num_seeds = 1,
                ln        = ln
            )
        else:
            self.global_pma = None

        self.readout = nn.Linear(embed_dim, 1)

    def forward(self, cand_tokens):
        # cand_tokens: (N, K, E) from momentum-only BranchSetEncoder
        for sab in self.sabs:
            cand_tokens = sab(cand_tokens)

        if self.use_global_context:
            X = self.global_pma(cand_tokens).squeeze(1)  # (N, E)
            cand_tokens = cand_tokens + X.unsqueeze(1)
        else:
            X = None

        logits = self.readout(cand_tokens).squeeze(-1)   # (N, K)
        return logits, cand_tokens, X

# --------------------------------------------------------------------------------------------------- BRANCH ENCODER

class BranchSetEncoder(nn.Module):
    def __init__(self, jet_feat_dim : int, embed_dim : int, num_heads : int, num_inds : int, ISAB_layers : int, SAB_layers : int, mode : str, ln : bool):
        super().__init__()

        self.jet_set = SetTransformer(
            dim_input   = jet_feat_dim,     # = 5 for {pt, phi, mass, eta, btag}
            num_outputs = 1,
            dim_output  = embed_dim,
            num_inds    = num_inds,
            dim_hidden  = embed_dim,
            num_heads   = num_heads,
            ln          = ln,
            ISAB_layers = ISAB_layers,
            SAB_layers  = SAB_layers
        )

        self.mask_head  = nn.Linear(embed_dim, 1)

        self.branch_pma = PMA(
            dim         = embed_dim,
            num_heads   = num_heads,
            num_seeds   = 1,
            ln          = ln
        )

        self.mom_mask = torch.tensor(1, 1, 1, 1, [1, 1, 0, 1, 0], dtype=torch.float32)

    def forward(self, X):
        N, K, B, J, F = X.shape

        if mode == "mom":
            x *= self.mom_mask
        else:
            x = x

        # Per-branch encoding (treat each branch as a set of J jets)
        X_flat = X.reshape(N * K * B, J, F)
        bt = self.jet_set(X_flat).squeeze(1)             # (NKB, E)

        # Per-branch mask logits
        m = self.mask_head(bt).squeeze(-1)               # (NKB,)
        mask_logits = m.reshape(N, K, B)

        # Reshape tokens to (NK, B, E) and pool to candidate token with PMA
        bt = bt.reshape(N, K, B, -1)                        # (N, K, B, E)
        bt2 = bt.reshape(N * K, B, -1)                      # (NK, B, E)
        ct = self.branch_pma(bt2).squeeze(1)                # (NK, E)
        ct = ct.reshape(N, K, -1)                           # (N, K, E)

        return bt, ct, mask_logits

# --------------------------------------------------------------------------------------------------- CANDIDATE ENCODER

class CandidateSetEncoder(nn.Module):
    def __init__(self, embed_dim : int, num_heads : int, ISAB_layers : int, SAB_layers : int, use_global_context : bool, ln : bool):
        super().__init__()

        self.sabs = nn.ModuleList([SAB(
            dim_in      = embed_dim,
            dim_out     = embed_dim,
            num_heads   = num_heads,
            ln          = ln
            ) for _ in range(SAB_layers)])
        
        self.use_global_context = use_global_context

        if use_global_context:
            self.global_pma = PMA(
                dim         = embed_dim,
                num_heads   = num_heads,
                num_seeds   = 1,
                ln          = ln
            ) if use_global_context else None

        self.readout = nn.Linear(embed_dim, 1)

    def forward(self, cand_tokens):
        for sab in self.sabs:
            cand_tokens = sab(cand_tokens)

        if self.use_global_context:
            X = self.global_pma(cand_tokens).squeeze(1)
            cand_tokens += X.unsqueeze(1)
        else:
            X = None
        
        logits = self.readout(cand_tokens).squeeze(-1)
        return logits, cand_tokens, X

# --------------------------------------------------------------------------------------------------- SPANET-SCM

class SCM_Training_Val(JetSecondaryLoader):
    def __init__(self, options: Options, torch_script: bool = False):
        super(SCM_Training_Val, self).__init__(options, torch_script)
        self.options = options

        # Set Transformer config with sensible defaults if missing
        self.embed_dim = 
        self.num_heads = 
        self.ISAB_layers = 
        self.SAB_layers = 
        self.use_global_context = 

        self.jet_feat_dim = 


        B = self.options.branch_dim
        J = self.options.jet_max_dim
        Fdim = self.options.features_dim