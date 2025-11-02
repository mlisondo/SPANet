import torch
import torch.nn as nn
import torch.nn.functional as F
from spanet.options import Options
from spanet.network.jet_reconstruction.jet_scm_pipeline import JetSecondaryLoader
from spanet.dataset.types import Batch
from typing import Optional
from torch import Tensor

tcompile = torch.compile

def probe(o, name=None):
    obj = type(o)
    header = f"Object '{name}'"
    print(f"\n{header}: {obj.__module__}.{obj.__name__}")

    if hasattr(o, 'shape'):
        print(f"shape: {o.shape}")
    if hasattr(o, 'ndim'):
        print(f"ndim: {o.ndim}")
    if hasattr(o, 'dtype'):
        print(f"dtype: {o.dtype}")

    if hasattr(o, 'size') and not callable(o.size):
        print(f"size: {o.size}")

    try:
        print(f"len: {len(o)}")
    except Exception:
        pass

    try:
        if isinstance(o, (list, tuple)):
            for idx, item in enumerate(o):
                probe(item, f"{name}[{idx}]")
    except Exception:
        pass

    if isinstance(o, torch.Tensor):
        print(f"shape: {tuple(o.size())}")
        print(f"dtype: {o.dtype}")
        print(f"numel: {o.numel()}")

        print(f"shape: {tuple(o.size())}")
        print(f"dtype: {o.dtype}")
        print(f"numel: {o.numel()}")
        print(f"device: {o.device}")

# --------------------------------------------------------------------------------------------------- MODULES FOR SET TRANSFORMER

# MULTIHEAD ATTENTION BLOCK
class MAB(nn.Module):
    def __init__(self, dim_Q : int, dim_K : int, dim_V : int, num_heads : int, attn_drop : float = 0.0, ff_drop : float = 0.0, ln : bool = True, use_gate : bool = True):
        super().__init__()

        self.pre_ln = ln
        self.use_gate = use_gate

        self.q_in = nn.Identity() if dim_Q == dim_V else nn.Linear(dim_Q, dim_V) # mha does k and v porjections internally

        self.ln_q = nn.LayerNorm(dim_Q)
        self.ln_k = nn.LayerNorm(dim_K)
        self.ln_ff_in = nn.LayerNorm(dim_V)
        
        self.mha = nn.MultiheadAttention(
            embed_dim = dim_V,
            num_heads = num_heads,
            dropout = attn_drop,
            batch_first = True,
            kdim = dim_K,
            vdim = dim_K
        )

        self.ff = nn.Sequential(
            nn.Linear(dim_V, 4*dim_V), nn.GELU(), nn.Dropout(ff_drop),
            nn.Linear(4*dim_V, dim_V), nn.Dropout(ff_drop),
        )

        self.attn_gate = nn.Parameter(torch.tensor(1.0))

    def forward(self, Q, K, key_padding_mask : Optional[Tensor] = None) -> Tensor:

        Qn = self.ln_q(Q) if self.pre_ln else Q
        Kn = self.ln_k(K) if self.pre_ln else K

        Qq = self.q_in(Qn)

        # ------- key_padding_mask -------
        # muted, not deaf
        # Nobody can listen to that position, but that position can still listen to others
        # mask shape (batch size, #keys/values)

        # ------- attn_mask ------- DO NOT NEED ATTN_MASK
        # choose who's muted and/or deaf
        # Block a column -> that position is muted
        # Block a row -> that position is deaf
        # Block row + column -> fully isolated
        # mask shape (#queries, #keys/values) or (batch size * num_heads, #queries, #keys/values)

        attn_out, _ = self.mha(
            Qq, Kn, Kn, 
            key_padding_mask = key_padding_mask,
            need_weights = False
        )

        out = Qq + (self.attn_gate * attn_out if self.use_gate else attn_out)
        ln_out = self.ln_ff_in(out) if self.pre_ln else out

        return out + self.ff(ln_out)

# SELF ATTENTION BLOCK
class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, attn_drop = 0.0, ff_drop = 0.0, ln = True, use_gate = True):
        super().__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, attn_drop, ff_drop, ln, use_gate)

    def forward(self, X, key_padding_mask : Optional[Tensor] = None):
        return self.mab(X, X, key_padding_mask)

# INDUCED SELF ATTENTION BLOCK
class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, attn_drop = 0.0, ff_drop = 0.0, ln = True, use_gate = True):
        super().__init__()
        self.I = nn.Parameter(torch.empty(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in,  dim_out, num_heads, attn_drop, ff_drop, ln, use_gate)  # P <- X
        self.mab1 = MAB(dim_in,  dim_out, dim_out, num_heads, attn_drop, ff_drop, ln, use_gate)  # X <- P

    def forward(self, X, key_padding_mask : Optional[Tensor] = None):
        P = self.I.expand(X.size(0), -1, -1)
        H = self.mab0(P, X, key_padding_mask = key_padding_mask)  # seeds read from jets
        Y = self.mab1(X, H)                          # jets read from seeds
        return Y

# POOLING BY MULTIHEAD ATTENTION
class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, attn_drop = 0.0, ff_drop = 0.0, ln = True, use_gate = True):
        super().__init__()
        self.S = nn.Parameter(torch.empty(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, attn_drop, ff_drop, ln, use_gate)

    def forward(self, X, key_padding_mask : Optional[Tensor] = None):
        S = self.S.expand(X.size(0), -1, -1)
        return self.mab(S, X, key_padding_mask = key_padding_mask) # build per-batch seed queries -> feeds them and the set X into attention -> outputs an attention-pooled summary of X

# --------------------------------------------------------------------------------------------------- SET TRANSFORMER

class SetTransformer(nn.Module):
    def __init__(self,
    dim_input   : int,
    num_outputs : int,
    dim_output  : int,
    num_inds    : int,
    dim_hidden  : int,
    num_heads   : int,
    attn_drop   : float, 
    ff_drop     : float,
    ln          : bool,
    ISAB_layers : int,
    SAB_layers  : int
    ):
        super().__init__()
        self.in_proj = nn.Linear(dim_input, dim_hidden)

        isab_layer = []
        for _ in range(ISAB_layers):
            isab_layer.append(ISAB(dim_hidden, dim_hidden, num_heads, num_inds, attn_drop,ff_drop,ln))
        self.isab = nn.Sequential(*isab_layer)

        self.pma = PMA(dim_hidden, num_heads, num_outputs, attn_drop, ff_drop, ln)

        sab_layer = []
        for _ in range(SAB_layers):
            sab_layer.append(SAB(dim_hidden, dim_hidden, num_heads, attn_drop, ff_drop, ln))
        self.sab = nn.Sequential(*sab_layer)

        self.out = nn.Linear(dim_hidden, dim_output)

    def forward(self, X, key_padding_mask: Optional[Tensor] = None):
        X = self.in_proj(X)

        # Encoders
        for layer in self.isab:
            X = layer(X, key_padding_mask=key_padding_mask) # pad on K/V

        X = self.pma(X, key_padding_mask=key_padding_mask) # pool queries seeds over set X with pad
        # Decoder
        for layer in self.sab: # self-attn over seed
            X = layer(X)
        return self.out(X)

# --------------------------------------------------------------------------------------------------- BRANCH ENCODER

class BranchSetEncoder(nn.Module):
    def __init__(self,
    inclusive_feat_dim    : int,    prior_feat_dim    : int,
    inclusive_embed_dim   : int,    prior_embed_dim   : int,
    inclusive_num_inds    : int,    prior_num_inds    : int,
    inclusive_num_heads   : int,    prior_num_heads   : int,
    inclusive_attn_drop   : float,  prior_attn_drop   : float,
    inclusive_ff_drop     : float,  prior_ff_drop     : float,
    inclusive_ln          : bool,   prior_ln          : bool,
    inclusive_isab_layers : int,    prior_isab_layers : int,
    inclusive_sab_layers  : int,    prior_sab_layers  : int
    ):
        super().__init__()

        # ====================== INCLUSIVE ======================
        self.inclusive_jet_set = SetTransformer(
            dim_input   = inclusive_feat_dim,
            num_outputs = 1,
            dim_output  = inclusive_embed_dim,
            num_inds    = inclusive_num_inds,
            dim_hidden  = inclusive_embed_dim,
            num_heads   = inclusive_num_heads,
            attn_drop   = inclusive_attn_drop,
            ff_drop     = inclusive_ff_drop,
            ln          = inclusive_ln,
            ISAB_layers = inclusive_isab_layers,
            SAB_layers  = inclusive_sab_layers
        )

        self.inclusive_mask_head  = nn.Linear(inclusive_embed_dim, 1)

        self.inclusive_branch_pma = PMA(
            dim         = inclusive_embed_dim,
            num_heads   = inclusive_num_heads,
            num_seeds   = 1,
            attn_drop   = inclusive_attn_drop,
            ff_drop     = inclusive_ff_drop,
            ln          = inclusive_ln
        )

        # ====================== PRIOR ======================
        self.prior_jet_set = SetTransformer(
            dim_input   = prior_feat_dim,
            num_outputs = 1,
            dim_output  = prior_embed_dim,
            num_inds    = prior_num_inds,
            dim_hidden  = prior_embed_dim,
            num_heads   = prior_num_heads,
            attn_drop   = prior_attn_drop,
            ff_drop     = prior_ff_drop,
            ln          = prior_ln,
            ISAB_layers = prior_isab_layers,
            SAB_layers  = prior_sab_layers
        )

        self.prior_mask_head  = nn.Linear(prior_embed_dim, 1)

        self.prior_branch_pma = PMA(
            dim         = prior_embed_dim,
            num_heads   = prior_num_heads,
            num_seeds   = 1,
            attn_drop   = prior_attn_drop,
            ff_drop     = prior_ff_drop,
            ln          = prior_ln
        )
    
    def forward(self, inclusive_X, prior_X, 
    inclusive_jet_kpm : Optional[Tensor] = None, inclusive_branch_kpm  : Optional[Tensor] = None,
    prior_jet_kpm : Optional[Tensor] = None, prior_branch_kpm : Optional[Tensor] = None):
        # ====================== INCLUSIVE ======================
        E, K, B, J, inclusive_F = inclusive_X.shape
        inclusive_X_flat = inclusive_X.reshape(E * K * B, J, inclusive_F) # (E*K*B, J, inclusive_F)

        # Per-branch encoding
        inclusive_bt = self.inclusive_jet_set(inclusive_X_flat, key_padding_mask = inclusive_jet_kpm).squeeze(1)
        # self.inclusive_jet_set(...) -> (E*K*B, 1, inclusive_embed_dim); squeeze(1) -> (E*K*B, inclusive_embed_dim)

        # Per-branch mask logits
        inclusive_m = self.inclusive_mask_head(inclusive_bt).squeeze(-1)
        # Linear(inclusive_embed_dim -> 1) -> (E*K*B, 1); squeeze(-1) => (E*K*B,)
        inclusive_mask_logits = inclusive_m.reshape(E, K, B) # (E, K, B)

        # Reshape tokens to (EK, B, E) and pool to candidate token with PMA
        inclusive_bt = inclusive_bt.reshape(E, K, B, -1) # (E, K, B, inclusive_embed_dim)
        inclusive_bt2 = inclusive_bt.reshape(E * K, B, -1) # (E*K, B, inclusive_embed_dim)
        inclusive_ct = self.inclusive_branch_pma(inclusive_bt2, key_padding_mask = inclusive_branch_kpm).squeeze(1)
        # (E*K, num_seeds (1), inclusive_embed_dim); squeeze(1) -> (E*K, inclusive_embed_dim)
        inclusive_ct = inclusive_ct.reshape(E, K, -1) # (E, K, inclusive_embed_dim)

        # ====================== PRIOR ======================
        _, _, _, _, prior_F = prior_X.shape
        prior_X_flat = prior_X.reshape(E * K * B, J, prior_F)

        # Per-branch encoding
        prior_bt = self.prior_jet_set(prior_X_flat, key_padding_mask = prior_jet_kpm).squeeze(1)

        # Per-branch mask logits
        prior_m = self.prior_mask_head(prior_bt).squeeze(-1)
        prior_mask_logits = prior_m.reshape(E, K, B)

        # Reshape tokens to (EK, B, E) and pool to candidate token with PMA        
        prior_bt = prior_bt.reshape(E, K, B, -1)
        prior_bt2 = prior_bt.reshape(E * K, B, -1)
        prior_ct = self.prior_branch_pma(prior_bt2, key_padding_mask = prior_branch_kpm).squeeze(1)
        prior_ct = prior_ct.reshape(E, K, -1)

        return (inclusive_bt, inclusive_ct, inclusive_mask_logits, 
        prior_bt, prior_ct, prior_mask_logits)

# SHAPES :
    # inclusive_bt : (E, K, B, inclusive_embed_dim)
    # inclusive_ct : (E, K, inclusive_embed_dim)
    # inclusive_mask_logits : (E, K, B)

    # prior_bt : (E, K, B, prior_embed_dim)
    # prior_ct : (E, K, prior_embed_dim)
    # prior_mask_logits : (E, K, B)

# --------------------------------------------------------------------------------------------------- CANDIDATE ENCODER

class CandidateSetEncoder(nn.Module):
    def __init__(self,
    inclusive_embed_dim : int,            prior_embed_dim : int,
    inclusive_num_heads : int,            prior_num_heads : int,
    inclusive_ISAB_layers : int,          prior_ISAB_layers : int,
    inclusive_num_inds : int,             prior_num_inds : int,
    inclusive_SAB_layers : int,           prior_SAB_layers : int,
    inclusive_use_global_context : bool,  prior_use_global_context : bool,
    inclusive_attn_drop : float,          prior_attn_drop : float,
    inclusive_ff_drop : float,            prior_ff_drop : float,
    inclusive_ln : bool,                  prior_ln : bool,
    inclusive_num_seeds : int,            prior_num_seeds : int,
    use_cross_from_branches : bool,
    detach_bt : bool
    ):
        super().__init__()
        self.use_cross_from_branches = use_cross_from_branches
        self.detach_bt = detach_bt # candidate loss wont backprop through bt -> BranchSetEncoder learns only from its own loss

        # ====================== INCLUSIVE ======================
        self.inclusive_isabs = nn.ModuleList([
            ISAB(
                dim_in    = inclusive_embed_dim,
                dim_out   = inclusive_embed_dim,
                num_heads = inclusive_num_heads,
                num_inds  = inclusive_num_inds,
                attn_drop = inclusive_attn_drop,
                ff_drop   = inclusive_ff_drop,
                ln        = inclusive_ln
            ) for _ in range(inclusive_ISAB_layers)]
        )

        self.inclusive_sabs = nn.ModuleList([
            SAB(
                dim_in    = inclusive_embed_dim,
                dim_out   = inclusive_embed_dim,
                num_heads = inclusive_num_heads,
                attn_drop = inclusive_attn_drop,
                ff_drop   = inclusive_ff_drop,
                ln        = inclusive_ln
            ) for _ in range(inclusive_SAB_layers)]
        )

        self.inclusive_use_global_context = inclusive_use_global_context

        self.inclusive_global_pma = PMA(
                dim       = inclusive_embed_dim,
                num_heads = inclusive_num_heads,
                num_seeds = inclusive_num_seeds, # set to 1 by default
                attn_drop = inclusive_attn_drop, # im not sure if i should make a seperate one for this module
                ff_drop   = inclusive_ff_drop, # im not sure if i should make a seperate one for this module
                ln        = inclusive_ln
        )

        self.inclusive_xattn = MAB(
            dim_Q     = inclusive_embed_dim,
            dim_K     = inclusive_embed_dim,
            dim_V     = inclusive_embed_dim,
            num_heads = inclusive_num_heads,
            attn_drop = inclusive_attn_drop, # im not sure if i should make a seperate one for this module
            ff_drop   = inclusive_ff_drop, # im not sure if i should make a seperate one for this module
            ln        = inclusive_ln
        )

        self.inclusive_gate = nn.Parameter(torch.tensor(0.1)) # could potentially change to vector of size equal to the number of heads
        # to allow the model to weight different attention heads differently

        self.inclusive_readout = nn.Sequential(
            nn.LayerNorm(inclusive_embed_dim),
            nn.GELU(),
            nn.Linear(inclusive_embed_dim, 1)
        )

        # ====================== PRIOR ======================
        self.prior_isabs = nn.ModuleList([
            ISAB(
                dim_in    = prior_embed_dim,
                dim_out   = prior_embed_dim,
                num_heads = prior_num_heads,
                num_inds  = prior_num_inds, 
                attn_drop = prior_attn_drop, # im not sure if i should make a seperate one for this module
                ff_drop   = prior_ff_drop, # im not sure if i should make a seperate one for this module
                ln        = prior_ln
            ) for _ in range(prior_ISAB_layers)]
        )

        self.prior_sabs = nn.ModuleList([
            SAB(
                dim_in    = prior_embed_dim,
                dim_out   = prior_embed_dim,
                num_heads = prior_num_heads,
                attn_drop = prior_attn_drop, # im not sure if i should make a seperate one for this module
                ff_drop   = prior_ff_drop, # im not sure if i should make a seperate one for this module
                ln        = prior_ln
            ) for _ in range(prior_SAB_layers)]
        )

        self.prior_use_global_context = prior_use_global_context

        self.prior_global_pma = PMA(
                dim       = prior_embed_dim,
                num_heads = prior_num_heads,
                num_seeds = prior_num_seeds, # set to 1 by default
                attn_drop = prior_attn_drop, # im not sure if i should make a seperate one for this module
                ff_drop   = prior_ff_drop, # im not sure if i should make a seperate one for this module
                ln        = prior_ln
        )

        self.prior_xattn = MAB(
            dim_Q     = prior_embed_dim,
            dim_K     = prior_embed_dim,
            dim_V     = prior_embed_dim,
            num_heads = prior_num_heads,
            attn_drop = prior_attn_drop, # im not sure if i should make a seperate one for this module
            ff_drop   = prior_ff_drop, # im not sure if i should make a seperate one for this module
            ln        = prior_ln
        )

        self.prior_gate = nn.Parameter(torch.tensor(0.1)) # could potentially change to vector of size equal to the number of heads
        # to allow the model to weight different attention heads differently

        self.prior_readout = nn.Sequential(
            nn.LayerNorm(prior_embed_dim),
            nn.GELU(),
            nn.Linear(prior_embed_dim, 1)
        )
        

    def forward(self,
    inclusive_bt, prior_bt, # (E, K, B, *_embed_dim)
    inclusive_ct, prior_ct,  # (E, K, *_embed_dim)
    branch_kpm_inclusive: Optional[Tensor] = None,  # (E*K,B)
    branch_kpm_prior: Optional[Tensor] = None,
    candidate_kpm_inclusive: Optional[Tensor] = None,  # (E,K)
    candidate_kpm_prior: Optional[Tensor] = None,
    ):
        # ====================== INCLUSIVE ======================
        # E, K, B, J, inclusive_F = inclusive_X.shape
        # inclusive_X_flat = inclusive_X.reshape(E, K, B * J * inclusive_F)

        for inclusive_isab in self.inclusive_isabs:
            inclusive_ct = inclusive_isab(inclusive_ct, key_padding_mask = candidate_kpm_inclusive) # (E, K, inclusive_embed_dim)

        for inclusive_sab in self.inclusive_sabs:
            inclusive_ct = inclusive_sab(inclusive_ct, key_padding_mask = candidate_kpm_inclusive) # (E, K, inclusive_embed_dim)

        if self.use_cross_from_branches: # each candidate refine itself using only its own branches
            # let candidate token be enriched by looking at its own branch tokens
            E, K, B, inclusive_embed_dim = inclusive_bt.shape
            if self.detach_bt:
                inclusive_bt = inclusive_bt.detach()
            inclusive_bt = inclusive_bt.reshape(E * K, B, inclusive_embed_dim) # (EK, B, inclusive_embed_dim)
            inclusive_ct = inclusive_ct.reshape(E * K, inclusive_embed_dim).unsqueeze(1) # (EK, 1, inclusive_embed_dim)
            x_talk_inclusive = self.inclusive_xattn(inclusive_ct, inclusive_bt, key_padding_mask = branch_kpm_inclusive).squeeze(1) # (E*K, 1, inclusive_embed_dim) -> (E*K, inclusive_embed_dim)
            x_talk_inclusive = x_talk_inclusive.reshape(E, K, inclusive_embed_dim) # (E, K, inclusive_embed_dim)
            inclusive_ct = inclusive_ct.reshape(E, K, inclusive_embed_dim) # (E, K, inclusive_embed_dim)

            # inclusive_bt : torch.Tensor(E*K, B, inclusive_embed_dim)
            # inclusive_ct : torch.Tensor(E, K, inclusive_embed_dim)
            # x_talk_inclusive : torch.Tensor(E, K, inclusive_embed_dim)

            inclusive_ct = inclusive_ct + self.inclusive_gate * x_talk_inclusive
        
        if self.inclusive_use_global_context: # give every candidate the same event-level summary built from all candidates, then add it to each candidate
            global_inclusive = self.inclusive_global_pma(inclusive_ct, key_padding_mask = candidate_kpm_inclusive).squeeze(1) # (EK, 1, inclusive_embed_dim).squeeze -> (EK, inclusive_embed_dim)

            # Note: IF THIS LINE ERRORES OUT, ITS BECAUSE r IS SET TO SOMETHING GREATER THAN 1, CHECK options.py *_seeds_classifer
            inclusive_ct = inclusive_ct + global_inclusive.unsqueeze(1) # unsqueeze (EK, 1, D); broadcasts across K when added; (E, K, D) 
        else:
            global_inclusive = None

        inclusive_logits = self.inclusive_readout(inclusive_ct).squeeze(-1)

        if candidate_kpm_inclusive is not None:
            neg_inf = torch.finfo(inclusive_logits.dtype).min
            inclusive_logits = inclusive_logits.masked_fill(candidate_kpm_inclusive, neg_inf) # EXTRA PROTECTION against all masked

        # ====================== PRIOR ======================
        # _, _, _, _, prior_F = prior_X.shape
        # prior_X_flat = prior_X.reshape(E, K, B * J * prior_F)

        for prior_isab in self.prior_isabs:
            prior_ct = prior_isab(prior_ct, key_padding_mask = candidate_kpm_prior)

        for prior_sab in self.prior_sabs:
            prior_ct = prior_sab(prior_ct, key_padding_mask = candidate_kpm_prior)

        if self.use_cross_from_branches:
            # let candidate token be enriched by looking at its own branch tokens
            E, K, B, prior_embed_dim = prior_bt.shape
            if self.detach_bt:
                prior_bt = prior_bt.detach()
            prior_bt = prior_bt.reshape(E * K, B, prior_embed_dim) # (EK, B, prior_embed_dim)
            prior_ct = prior_ct.reshape(E * K, prior_embed_dim).unsqueeze(1) # (EK, 1, prior_embed_dim)
            x_talk_prior = self.prior_xattn(prior_ct, prior_bt, key_padding_mask=branch_kpm_prior).squeeze(1) # (E*K, 1, prior_embed_dim) -> (E*K, prior_embed_dim)
            # MAKE SURE THAT THE DIM MACTH
            x_talk_prior = x_talk_prior.reshape(E, K, prior_embed_dim) # (E, K, prior_embed_dim)
            prior_ct = prior_ct.reshape(E, K, prior_embed_dim)
            prior_ct = prior_ct + self.prior_gate * x_talk_prior
        
        if self.prior_use_global_context:
            global_prior = self.prior_global_pma(prior_ct, key_padding_mask = candidate_kpm_prior).squeeze(1)
            prior_ct = prior_ct + global_prior.unsqueeze(1)
        else:
            global_prior = None

        prior_logits = self.prior_readout(prior_ct).squeeze(-1)

        if candidate_kpm_prior is not None:
            neg_inf = torch.finfo(prior_logits.dtype).min
            prior_logits = prior_logits.masked_fill(candidate_kpm_prior, neg_inf) # EXTRA MEASURE !!

        return (inclusive_logits, inclusive_ct, global_inclusive,
        prior_logits, prior_ct, global_prior)

# --------------------------------------------------------------------------------------------------- DATA

# DATA SHAPES & ORDERING (from JetSecondaryLoader.topk_data)
    # Indices: 0-based jet indices; padded with -1 to length p_max (pads at the end).
    # Ordering: for matching only, per-branch jet indices are sorted ascending; truth branches are
    #           canonically permuted per event. Predictions remain unsorted.
    # Returns (in order):
    #   pred_truth:      (E, K, B) bool         # per-K branch match
    #   canon_masks:     (B, E) bool            # reconstructable branches after canonical permutation
    #   features_arr:    (E, K, B, p_max, F)    # features at raw predicted indices (no re-sorting)
    #   class_truth:     (E, K) bool            # hypothesis correct iff all reconstructable branches match and >= 1 branch exists
    #   canon_idx:       (B, E, p_max) long     # canonicalized truth indices; ascending within branch; -1 padded
    #   jet_preds_tensor:(E, K, B, p_max) long  # raw predicted indices
    #   jet_mult:        (E, Njets) bool        # how many jets were even available for assingment, max of NJets (= 10) per event
    #
    # Jet features (F=5) and preprocessing; jet_data has shape (E, Njets, 5) in this exact order:
    #   {pt, eta, phi, mass, btag}
    #   mass: log_normalize
    #   pt:   log_normalize
    #   eta:  normalize
    #   phi:  normalize
    #   btag: none

# --------------------------------------------------------------------------------------------------- SPANET-SCM

class SCM_Training_Val(JetSecondaryLoader):
    def __init__(self, options: Options, torch_script: bool = False):
        super(SCM_Training_Val, self).__init__(options, torch_script)
        self.options = options

        # New transformer config with sensible defaults if missing
        # masker
        self.i_dim_masker = options.i_dim_masker # both models expect the same dim
        self.p_dim_masker = options.p_dim_masker

        self.inc_heads_masker = options.inc_heads_masker
        self.p_heads_masker = options.p_heads_masker

        self.i_inds_masker = options.i_inds_masker
        self.p_inds_masker = options.p_inds_masker

        self.i_isab_masker = options.i_isab_masker
        self.p_isab_masker = options.p_isab_masker

        self.i_sab_masker = options.i_sab_masker
        self.p_sab_masker = options.p_sab_masker

        self.i_attn_masker = options.i_attn_masker
        self.p_attn_masker = options.p_attn_masker

        self.i_ff_masker = options.i_ff_masker
        self.p_ff_masker = options.p_ff_masker

        self.i_ln_masker = options.i_ln_masker
        self.p_ln_masker = options.p_ln_masker
        
        # classifier
        self.i_dim_classifer = options.i_dim_classifer
        self.p_dim_classifer = options.p_dim_classifer

        self.inc_heads_classifer = options.inc_heads_classifer
        self.p_heads_classifer = options.p_heads_classifer

        self.i_inds_classifer = options.i_inds_classifer
        self.p_inds_classifer = options.p_inds_classifer

        self.i_isab_classifer = options.i_isab_classifer
        self.p_isab_classifer = options.p_isab_classifer

        self.i_sab_classifer = options.i_sab_classifer
        self.p_sab_classifer = options.p_sab_classifer

        self.i_seeds_classifer = options.i_seeds_classifer
        self.p_seeds_classifer = options.p_seeds_classifer

        self.i_attn_classifer = options.i_attn_classifer
        self.p_attn_classifer = options.p_attn_classifer

        self.i_ff_classifer = options.i_ff_classifer
        self.p_ff_classifer = options.p_ff_classifer

        self.i_ln_classifer = options.i_ln_classifer
        self.p_ln_classifer = options.p_ln_classifer

        self.i_global_classifer = options.i_global_classifer
        self.p_global_classifer = options.p_global_classifer   

        self.use_x_branches = options.use_x_branches
        self.detach_branch = options.detach_branch

        # ================================ general ================================

        B    = self.options.branch_dim
        J    = self.options.jet_max_dim
        Fdim = self.options.features_dim   # jets: {pt, eta, phi, mass, btag} => usually 5

        self.prior_weight_start = options.prior_weight_start
        self.prior_weight_end = options.prior_weight_end
        self.prior_weight_epochs = options.prior_weight_epochs

        self.masker_weight_start = options.masker_weight_start
        self.masker_weight_end = options.masker_weight_end
        self.masker_weight_epochs = options.masker_weight_epochs

        # ================================ MASKER ================================
        self.masker = BranchSetEncoder(
            # features per jet
            inclusive_feat_dim    = Fdim,
            prior_feat_dim        = 3,

            # embedding sizes
            inclusive_embed_dim   = self.i_dim_masker,
            prior_embed_dim       = self.p_dim_masker,

            # ISAB per-branch jet-set encoder
            inclusive_num_inds    = self.i_inds_masker,
            prior_num_inds        = self.p_inds_masker,

            # attention heads
            inclusive_num_heads   = self.inc_heads_masker,
            prior_num_heads       = self.p_heads_masker,

            # dropouts
            inclusive_attn_drop   = self.i_attn_masker,
            prior_attn_drop       = self.p_attn_masker,
            inclusive_ff_drop     = self.i_ff_masker,
            prior_ff_drop         = self.p_ff_masker,

            # LayerNorm on/off
            inclusive_ln          = self.i_ln_masker,
            prior_ln              = self.p_ln_masker,

            # depth: jet-set encoder (ISAB) and post-PMA SAB
            inclusive_isab_layers = self.i_isab_masker,
            prior_isab_layers     = self.p_isab_masker,
            inclusive_sab_layers  = self.i_sab_masker,
            prior_sab_layers      = self.p_sab_masker,
        )

        # ============================== CLASSIFIER ==============================
        self.classifier = CandidateSetEncoder(
            # must match maskers per-candidate token dims
            inclusive_embed_dim       = self.i_dim_classifer, # NOTE:  # both models expect the same dim
            prior_embed_dim           = self.p_dim_classifer,

            # attention heads
            inclusive_num_heads       = self.inc_heads_classifer,
            prior_num_heads           = self.p_heads_classifer,

            # depth on candidate tokens
            inclusive_ISAB_layers     = self.i_isab_classifer,
            prior_ISAB_layers         = self.p_isab_classifer,
            inclusive_num_inds        = self.i_inds_classifer,
            prior_num_inds            = self.p_inds_classifer,
            inclusive_SAB_layers      = self.i_sab_classifer,
            prior_SAB_layers          = self.p_sab_classifer,

            # event-level global context via PMA (over K candidates)
            inclusive_use_global_context = self.i_global_classifer,
            prior_use_global_context     = self.p_global_classifer,

            # readout/regularization
            inclusive_attn_drop       = self.i_attn_classifer,
            prior_attn_drop           = self.p_attn_classifer,
            inclusive_ff_drop         = self.i_ff_classifer,
            prior_ff_drop             = self.p_ff_classifer,
            inclusive_ln              = self.i_ln_classifer,
            prior_ln                  = self.p_ln_classifer,

            # PMA seeds for global context
            inclusive_num_seeds       = self.i_seeds_classifer,
            prior_num_seeds           = self.p_seeds_classifer,

            # classifier-specific knobs
            use_cross_from_branches   = self.use_x_branches,
            detach_bt                 = self.detach_branch
        )

        for n, p in self.named_parameters():
            if not (n.startswith("classifier.") or n.startswith("masker.")):
                p.requires_grad_(False)

        # Compile
        self.classifier = tcompile(self.classifier, dynamic=True)
        self.masker     = tcompile(self.masker,    dynamic=True)

        # Imbalance / focal
        self.pos_weight_cap = 1000.0
        self.use_focal_masker = "use_focal_masker"
        self.focal_alpha_pos = 0.7
        self.focal_gamma = 2.0
    
    @staticmethod
    def focal_bce_with_logits(logits, targets, alpha_pos=0.25, gamma=2.0, reduction="mean"):
        p = torch.sigmoid(logits)
        pt = torch.where(targets.bool(), p, 1 - p)  # p_t
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        alpha_t = torch.where(
            targets.bool(),
            torch.as_tensor(alpha_pos, device=logits.device, dtype=logits.dtype),
            torch.as_tensor(1 - alpha_pos, device=logits.device, dtype=logits.dtype),
        )
        loss = alpha_t * (1 - pt).pow(gamma) * bce

        if reduction == "mean":
            return loss.mean()
        if reduction == "sum":
            return loss.sum()
        return loss

    @staticmethod
    def candidate_mute_mask(jet_preds_tensor): # for KTM
        # keep exactly the last member of each equivalence class
        # earlier duplicates are marked and dropped

        # potentially change logic to one similar to jet_scm_pipeline, or actually probably best to do it at that stage
        E, K, B, J = jet_preds_tensor.shape
        dev = jet_preds_tensor.device

        flat = jet_preds_tensor.reshape(E, K, B * J).to(torch.int64)

        FNV_OFFSET = torch.tensor(1469598103934665603, dtype=torch.int64, device=dev)
        FNV_PRIME  = torch.tensor(1099511628211, dtype=torch.int64, device=dev)

        h = FNV_OFFSET.expand(E, K).clone()
        for t in range(B * J):
            h = (h ^ flat[..., t]) * FNV_PRIME
        h = h ^ (h >> 32)

        h_sorted, perm = torch.sort(h, dim = 1, stable = True)
        flat_sorted = flat.gather(1, perm.unsqueeze(-1).expand_as(flat))

        same_hash = h_sorted[:, 1:] == h_sorted[:, :-1]
        eq_full = same_hash & flat_sorted[:, 1:, :].eq(flat_sorted[:, :-1, :]).all(dim = -1)

        dup_sorted = torch.zeros((E, K), dtype = torch.bool, device = dev)
        dup_sorted[:, :-1] = eq_full

        inv = torch.empty_like(perm)
        inv.scatter_(1, perm, torch.arange(K, device = dev).expand(E, K))
        dup = dup_sorted.gather(1, inv)

        return ~dup     # this returns a keep/valid mask, True -> candidate is valid and should be used. must be "NOT"ed if used as attn_mask

    @staticmethod
    def multi_positive_ce(class_logits : torch.Tensor, class_truth : torch.Tensor, valid_mask : torch.Tensor | None = None):
        if valid_mask is None:
            valid_mask = torch.ones_like(class_truth, dtype = torch.bool, device = class_truth.device)

        with torch.no_grad():
            pos_total = (class_truth.bool() & valid_mask).sum()
            neg_total = valid_mask.sum() - pos_total
            weight_scalar = (neg_total.to(torch.float32) / pos_total.clamp_min(1).to(torch.float32))
            K = class_logits.size(1)
            pos_weight = weight_scalar.expand(K).contiguous()
    
        weights = valid_mask.to(class_logits.dtype)
        pos_mask = class_truth.bool() & valid_mask
        has_truth = pos_mask.any(dim=1)
        num_pos = pos_mask.sum(dim=1)

        safe_logits = torch.where(valid_mask , class_logits, torch.zeros_like(class_logits))

        bce = F.binary_cross_entropy_with_logits(
            safe_logits,
            class_truth.to(class_logits.dtype),
            reduction = "none",
            pos_weight = pos_weight
        ) # (N, K)

        denom = weights.sum(dim = 1).clamp_min(1.0)
        loss_vec = (bce * weights).sum(dim = 1) / denom
        loss = loss_vec[has_truth].mean() if has_truth.any() else torch.zeros((), device = class_logits.device, dtype = loss_vec.dtype)

        with torch.no_grad():
            base = F.binary_cross_entropy_with_logits(
                torch.zeros_like(class_logits),
                class_truth.to(class_logits.dtype),
                reduction = "none",
                pos_weight = pos_weight
            )
            base_vec = (base * weights).sum(dim=1) / denom
            ce_baseline = base_vec[has_truth].mean() if has_truth.any() else base_vec.mean()

        return loss, has_truth, num_pos, ce_baseline

    def _compiled_core(self, features_arr, pred_truth, class_truth, cand_keep, cand_kpm : Optional[Tensor] = None, branch_kpm : Optional[Tensor] = None, jet_kpm : Optional[Tensor] = None):
        N, K, B, J, Fdim = features_arr.shape

        inclusive_ft = features_arr.clone()
        prior_ft = features_arr[..., :3].contiguous()

        (inclusive_bt, inclusive_ct, inclusive_mask_logits, 
        prior_bt, prior_ct, prior_mask_logits) = self.masker(
            inclusive_X = inclusive_ft, prior_X = prior_ft
        ) # forward takes
        #       necessary : inclusive_X, prior_X
        #       optional  : inclusive_jet_kpm, inclusive_branch_kpm, prior_jet_kpm, prior_branch_kpm

        (inclusive_logits, inclusive_ct, global_inclusive,
        prior_logits, prior_ct, global_prior) = self.classifier(
            inclusive_bt = inclusive_bt, prior_bt = prior_bt,
            inclusive_ct = inclusive_ct, prior_ct = prior_ct,
            candidate_kpm_inclusive = cand_kpm, candidate_kpm_prior = cand_kpm
        ) # forward takes
        #       necessary : inclusive_bt, prior_bt, inclusive_ct, prior_ct
        #       optional  : branch_kpm_inclusive, branch_kpm_prior, candidate_kpm_inclusive, candidate_kpm_prior

        inclusive_class_loss, has_truth, num_pos, ce_baseline_inclusive = self.multi_positive_ce(inclusive_logits, class_truth, valid_mask = cand_keep)

        prior_class_loss, _, _, ce_baseline_prior = self.multi_positive_ce(prior_logits, class_truth, valid_mask = cand_keep)

        rows = torch.arange(N, device = inclusive_logits.device)

        neg_inf = torch.finfo(inclusive_logits.dtype).min
        masked_inclusive = inclusive_logits.masked_fill(~cand_keep, neg_inf)
        pred_k = masked_inclusive.argmax(dim=1)

        top1_acc_truth = torch.tensor(0., device = inclusive_logits.device)
        num_pos_mean = num_pos.float().mean()
        if has_truth.any():
            top1_acc_truth = class_truth[rows[has_truth], pred_k[has_truth]].float().mean()
            num_pos_mean = num_pos[has_truth].float().mean()
        has_truth_frac = has_truth.float().mean()

        flat_truth = pred_truth.reshape(N * K, B)
        pos_rate = flat_truth.float().mean()

        # I DONT THINK I SHOULD RUN THIS LOSS DUE TO THE FACT THAT IT MIGHT AFFECT CLASSIFICATION METRICS.
        # the branch set encoder is currently trying to learn two things at once, how to properly summarize the data and then how to
        # tell if the branch is reconstructable or not. i might have to ignore reconstructability for now.
        branch_loss_inclusive = self.focal_bce_with_logits(
            inclusive_mask_logits.reshape(N*K, B), flat_truth,
            alpha_pos = self.focal_alpha_pos,
            gamma = self.focal_gamma,
            reduction = "mean"
        )
        branch_loss_prior = self.focal_bce_with_logits(
            prior_mask_logits.reshape(N*K, B), flat_truth,
            alpha_pos = self.focal_alpha_pos,
            gamma = self.focal_gamma,
            reduction = "mean"
        )

        return (
            inclusive_class_loss, prior_class_loss,
            top1_acc_truth, num_pos_mean, pos_rate,
            ce_baseline_inclusive, ce_baseline_prior, 
            branch_loss_inclusive, branch_loss_prior
        )
        
    _compiled_core = tcompile(_compiled_core, dynamic = True)

    def forward_scm(self, batch):
        (
            pred_truth, canon_masks, features_arr, 
            class_truth, canon_idx, jet_preds_tensor, 
            jet_mult
        ) = self.topk_data(batch)

        # candidate level kpm (attn_mask)
        cand_keep = self.candidate_mute_mask(jet_preds_tensor)  # (E, K)  True = keep
        cand_kpm = ~cand_keep   # key_padding_mask wants True = ignore

        # currently dont have a use branch_kpm, oops (same for jet_kpm)

        return self._compiled_core(features_arr, pred_truth, class_truth, cand_keep, cand_kpm = cand_kpm)

    def affine_map(self, start: float, end: float, total_epochs: int) -> float:
        if total_epochs <= 0:
            return float(end)
        t = min(1.0, max(0.0, self.current_epoch / float(total_epochs)))
        return float(start + (end - start) * t)

    def training_step(self, batch : Batch, batch_idx : int):
        prior_weight = self.affine_map(self.prior_weight_start, self.prior_weight_end, self.prior_weight_epochs)
        inclusive_weight = (1 - prior_weight)

        masker_weight = self.affine_map(self.masker_weight_start, self.masker_weight_end, self.masker_weight_epochs)

        (
            inclusive_class_loss, prior_class_loss,
            top1_acc_truth, num_pos_mean, pos_rate,
            ce_baseline_inclusive, ce_baseline_prior, 
            branch_loss_inclusive, branch_loss_prior
        ) = self.forward_scm(batch)

        total_loss_inclusive = inclusive_class_loss + (masker_weight * branch_loss_inclusive)
        total_loss_prior = prior_class_loss + (masker_weight * branch_loss_prior)
        abs_total_loss = (inclusive_weight * total_loss_inclusive) + (prior_weight * total_loss_prior)

        return abs_total_loss

    def validation_step(self, batch : Batch, batch_idx : int):
        prior_weight = self.affine_map(self.prior_weight_start, self.prior_weight_end, self.prior_weight_epochs)
        inclusive_weight = (1 - prior_weight)

        masker_weight = self.affine_map(self.masker_weight_start, self.masker_weight_end, self.masker_weight_epochs)

        (
            inclusive_class_loss, prior_class_loss,
            top1_acc_truth, num_pos_mean, pos_rate,
            ce_baseline_inclusive, ce_baseline_prior, 
            branch_loss_inclusive, branch_loss_prior
        ) = self.forward_scm(batch)

        total_loss_inclusive = inclusive_class_loss + (masker_weight * branch_loss_inclusive)
        total_loss_prior = prior_class_loss + (masker_weight * branch_loss_prior)
        abs_total_loss = (inclusive_weight * total_loss_inclusive) + (prior_weight * total_loss_prior)

        self.log('inclusive_class_loss', inclusive_class_loss, on_epoch = True, prog_bar = True)
        self.log('prior_class_loss', prior_class_loss, on_epoch = True, prog_bar = True)
        self.log('branch_loss_inclusive', branch_loss_inclusive, on_epoch = True, prog_bar = True)
        self.log('branch_loss_prior', branch_loss_prior, on_epoch = True, prog_bar = True)

        self.log('total_loss_inclusive', total_loss_inclusive, on_epoch = True, prog_bar = True)
        self.log('total_loss_prior', total_loss_prior, on_epoch = True, prog_bar = True)
        self.log('abs_total_loss', abs_total_loss, on_epoch = True, prog_bar = True)
        
        self.log('top1_acc_truth', top1_acc_truth, on_epoch = True, prog_bar = True)
        self.log('num_pos_mean', num_pos_mean, on_epoch = True, prog_bar = True)
        self.log('pos_rate', pos_rate, on_epoch = True, prog_bar = True)
        self.log('ce_baseline_inclusive', ce_baseline_inclusive, on_epoch = True, prog_bar = True)
        self.log('ce_baseline_prior', ce_baseline_prior, on_epoch = True, prog_bar = True)

        self.log('prior_weight', prior_weight, on_epoch = True, prog_bar = True)
        self.log('inclusive_weight', inclusive_weight, on_epoch = True, prog_bar = True)
        self.log('masker_weight', masker_weight, on_epoch = True, prog_bar = True)

        return {'abs_total_loss': abs_total_loss}