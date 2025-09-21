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
class MAB(nn.Module): # with pre/post LayerNorm, attention dropout
    def __init__(self, dim_Q : int, dim_K : int, dim_V : int, num_heads : int, pre_ln : bool, post_ln : bool, p_attn : float):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.pre_ln = pre_ln
        self.post_ln = post_ln

        # pre LayerNorm
        if pre_ln:
            self.ln_q = nn.LayerNorm(dim_Q)
            self.ln_k = nn.LayerNorm(dim_K)
            self.ln_mlp = nn.LayerNorm(dim_V)

        # projections
        self.proj_q = nn.Linear(dim_Q, dim_V)
        self.proj_k = nn.Linear(dim_K, dim_V)
        self.proj_v = nn.Linear(dim_K, dim_V)
        self.proj_out = nn.Linear(dim_V, dim_V)

        # Dropout
        self.attn_drop = nn.Dropout(p_attn)

        # post LayerNorm
        if post_ln:
            self.post_ln0 = nn.LayerNorm(dim_V)
            self.post_ln1 = nn.LayerNorm(dim_V)

        # learned gates
        self.attn_gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, Q, K, attn_mask: torch.Tensor = None):

        if self.pre_ln:
            Q_n = self.ln_q(Q)
            K_n = self.ln_k(K)
        else:
            Q_n, K_n = Q, K

        Qp = self.proj_q(Q_n)     # (B, Nq, dim_V)
        Kp = self.proj_k(K_n)     # (B, Nk, dim_V)
        Vp = self.proj_v(K_n)     # (B, Nk, dim_V)

        d_head = self.dim_V // self.num_heads
        Qh = torch.cat(Qp.split(d_head, dim=2), dim=0)   # (B*H, Nq, d_head)
        Kh = torch.cat(Kp.split(d_head, dim=2), dim=0)   # (B*H, Nk, d_head)
        Vh = torch.cat(Vp.split(d_head, dim=2), dim=0)   # (B*H, Nk, d_head)

        # scaled dot porduct
        pre_softmax_attn = Qh.bmm(Kh.transpose(1, 2)) / math.sqrt(d_head)

        # masking
        if attn_mask is not None:

        
        if self.use_dropout:
            attn = attn_drop(attn)

        # add residual
        out = torch.cat((Q_ + (A @ V_).split(Q.size(0), 0)), 2)

        # add linear norm
        if self.post_ln:
            out = self.post_ln0(out)
            out += F.relu(self.proj_out(out)) # small MLP
            out = self.post_ln1(out)
        else:
            out += F.relu(self.fc_o(out))

        return out

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
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X) # inducing points summarize the set X into m learned slots (H)
        return self.mab1(X, H) # each element of X attends to the induced summary H instead of all of X

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
        return self.mab(self.S.repeat(X.size(0), 1, 1), X) # build per-batch seed queries -> feeds them and the set X into attention -> outputs an attention-pooled summary of X

# --------------------------------------------------------------------------------------------------- SET TRANSFORMER

class SetTransformer(nn.Module):
    def __init__(self,
    dim_input   : int,
    num_outputs : int,
    dim_output  : int,
    num_inds    : int,
    dim_hidden  : int,
    num_heads   : int,
    ln          : bool,
    ISAB_layers : int,
    SAB_layers  : int
    ):
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

# --------------------------------------------------------------------------------------------------- BRANCH ENCODER

class BranchSetEncoder(nn.Module):
    def __init__(self,
    inclusive_feat_dim    : int,    prior_feat_dim    : int,
    inclusive_embed_dim   : int,    prior_embed_dim   : int,
    inclusive_num_inds    : int,    prior_num_inds    : int,
    inclusive_num_heads   : int,    prior_num_heads   : int,
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
            ln          = inclusive_ln,
            ISAB_layers = inclusive_isab_layers,
            SAB_layers  = inclusive_sab_layers
        )

        self.inclusive_mask_head  = nn.Linear(inclusive_embed_dim, 1)

        self.inclusive_branch_pma = PMA(
            dim         = inclusive_embed_dim,
            num_heads   = inclusive_num_heads,
            num_seeds   = 1,
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
            ln          = prior_ln,
            ISAB_layers = prior_isab_layers,
            SAB_layers  = prior_sab_layers
        )

        self.prior_mask_head  = nn.Linear(prior_embed_dim, 1)

        self.prior_branch_pma = PMA(
            dim         = prior_embed_dim,
            num_heads   = prior_num_heads,
            num_seeds   = 1,
            ln          = prior_ln
        )
    
    def forward(self, inclusive_X, prior_X):
        # ====================== INCLUSIVE ======================
        E, K, B, J, inclusive_F = inclusive_X.shape
        inclusive_X_flat = inclusive_X.reshape(E * K * B, J, inclusive_F) # (E*K*B, J, inclusive_F)

        # Per-branch encoding
        inclusive_bt = self.inclusive_jet_set(inclusive_X_flat).squeeze(1)
        # self.inclusive_jet_set(...) -> (E*K*B, 1, inclusive_embed_dim); squeeze(1) -> (E*K*B, inclusive_embed_dim)

        # Per-branch mask logits
        inclusive_m = self.inclusive_mask_head(inclusive_bt).squeeze(-1)
        # Linear(inclusive_embed_dim -> 1) -> (E*K*B, 1); squeeze(-1) => (E*K*B,)
        inclusive_mask_logits = inclusive_m.reshape(E, K, B) # (E, K, B)

        # Reshape tokens to (EK, B, E) and pool to candidate token with PMA
        inclusive_bt = inclusive_bt.reshape(E, K, B, -1) # (E, K, B, inclusive_embed_dim)
        inclusive_bt2 = inclusive_bt.reshape(E * K, B, -1) # (E*K, B, inclusive_embed_dim)
        inclusive_ct = self.inclusive_branch_pma(inclusive_bt2).squeeze(1)
        # (E*K, num_seeds (1), inclusive_embed_dim); squeeze(1) -> (E*K, inclusive_embed_dim)
        inclusive_ct = inclusive_ct.reshape(E, K, -1) # (E, K, inclusive_embed_dim)

        # ====================== INCLUSIVE ======================
        _, _, _, _, prior_F = prior_X.shape
        prior_X_flat = prior_X.reshape(E * K * B, J, prior_F)

        # Per-branch encoding
        prior_bt = self.prior_jet_set(prior_X_flat).squeeze(1)

        # Per-branch mask logits
        prior_m = self.prior_mask_head(prior_bt).squeeze(-1)
        prior_mask_logits = prior_m.reshape(E, K, B)

        # Reshape tokens to (EK, B, E) and pool to candidate token with PMA        
        prior_bt = prior_bt.reshape(E, K, B, -1)
        prior_bt2 = prior_bt.reshape(E * K, B, -1)
        prior_ct = self.prior_branch_pma(prior_bt2).squeeze(1)
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
                ln        = inclusive_ln
            ) for _ in range(inclusive_ISAB_layers)]
        )

        self.inclusive_sabs = nn.ModuleList([
            SAB(
                dim_in    = inclusive_embed_dim,
                dim_out   = inclusive_embed_dim,
                num_heads = inclusive_num_heads,
                ln        = inclusive_ln
            ) for _ in range(inclusive_SAB_layers)]
        )

        self.inclusive_use_global_context = inclusive_use_global_context

        self.inclusive_global_pma = PMA(
                dim       = inclusive_embed_dim,
                num_heads = inclusive_num_heads,
                num_seeds = inclusive_num_seeds, # set to 1 by default
                ln        = inclusive_ln
        )

        self.inclusive_xattn = MAB(
            dim_Q     = inclusive_embed_dim,
            dim_K     = inclusive_embed_dim,
            dim_V     = inclusive_embed_dim,
            num_heads = inclusive_num_heads,
            ln        = inclusive_ln
        )

        self.inclusive_gate = nn.Parameter(torch.tensor(0.0))

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
                ln        = prior_ln
            ) for _ in range(prior_ISAB_layers)]
        )

        self.prior_sabs = nn.ModuleList([
            SAB(
                dim_in    = prior_embed_dim,
                dim_out   = prior_embed_dim,
                num_heads = prior_num_heads,
                ln        = prior_ln
            ) for _ in range(prior_SAB_layers)]
        )

        self.prior_use_global_context = prior_use_global_context

        self.prior_global_pma = PMA(
                dim       = prior_embed_dim,
                num_heads = prior_num_heads,
                num_seeds = prior_num_seeds, # set to 1 by default
                ln        = prior_ln
        )

        self.prior_xattn = MAB(
            dim_Q     = prior_embed_dim,
            dim_K     = prior_embed_dim,
            dim_V     = prior_embed_dim,
            num_heads = prior_num_heads,
            ln        = prior_ln
        )

        self.prior_gate = nn.Parameter(torch.tensor(0.0))

        self.prior_readout = nn.Sequential(
            nn.LayerNorm(prior_embed_dim),
            nn.GELU(),
            nn.Linear(prior_embed_dim, 1)
        )
        

    def forward(self,
    inclusive_bt, prior_bt, # (E, K, B, *_embed_dim)
    inclusive_ct, prior_ct  # (E, K, *_embed_dim)
    ):
        # ====================== INCLUSIVE ======================
        # E, K, B, J, inclusive_F = inclusive_X.shape
        # inclusive_X_flat = inclusive_X.reshape(E, K, B * J * inclusive_F)

        for inclusive_isab in self.inclusive_isabs:
            inclusive_ct = inclusive_isab(inclusive_ct) # (E, K, inclusive_embed_dim)

        for inclusive_sab in self.inclusive_sabs:
            inclusive_ct = inclusive_sab(inclusive_ct) # (E, K, inclusive_embed_dim)

        if self.use_cross_from_branches: # each candidate refine itself using only its own branches
            # let candidate token be enriched by looking at its own branch tokens
            E, K, B, inclusive_embed_dim = inclusive_bt.shape
            if detach_bt:
                inclusive_bt = inclusive_bt.detach()
            inclusive_bt = inclusive_bt.reshape(E * K, B, inclusive_embed_dim) # (EK, B, inclusive_embed_dim)
            inclusive_ct = inclusive_ct.reshape(E * K, inclusive_embed_dim).unsqueeze(1) # (EK, 1, inclusive_embed_dim)
            x_talk_inclusive = inclusive_xattn(inclusive_ct, inclusive_bt).squeeze(1) # (E*K, 1, inclusive_embed_dim) -> (E*K, inclusive_embed_dim)
            # MAKE SURE THAT THE DIM MACTH
            x_talk_inclusive = x_talk_inclusive.reshape(E, K, inclusive_embed_dim) # (E, K, inclusive_embed_dim)
            inclusive_ct += inclusive_gate * x_talk_inclusive
        
        if self.inclusive_use_global_context: # give every candidate the same event-level summary built from all candidates, then add it to each candidate
            global_inclusive = self.inclusive_global_pma(inclusive_ct).squeeze(1) # (EK, 1, inclusive_embed_dim).squeeze -> (EK, inclusive_embed_dim)
            inclusive_ct += global_inclusive.unsqueeze(1) # unsqueeze (EK, 1, D); broadcasts across K when added; (E, K, D)
        else:
            global_inclusive = None

        inclusive_logits = self.inclusive_readout(inclusive_ct).squeeze(-1)

        # ====================== PRIOR ======================
        # _, _, _, _, prior_F = prior_X.shape
        # prior_X_flat = prior_X.reshape(E, K, B * J * prior_F)

        for prior_isab in self.prior_isabs:
            prior_ct = prior_isab(prior_ct)

        for prior_sab in self.prior_sabs:
            prior_ct = prior_sab(prior_ct)

        if self.use_cross_from_branches:
            # let candidate token be enriched by looking at its own branch tokens
            E, K, B, prior_embed_dim = prior_bt.shape
            if detach_bt:
                prior_bt = prior_bt.detach()
            prior_bt = prior_bt.reshape(E * K, B, prior_embed_dim) # (EK, B, prior_embed_dim)
            prior_ct = prior_ct.reshape(E * K, prior_embed_dim).unsqueeze(1) # (EK, 1, prior_embed_dim)
            x_talk_prior = prior_xattn(prior_ct, prior_bt).squeeze(1) # (E*K, 1, prior_embed_dim) -> (E*K, prior_embed_dim)
            # MAKE SURE THAT THE DIM MACTH
            x_talk_prior = x_talk_prior.reshape(E, K, prior_embed_dim) # (E, K, prior_embed_dim)
            prior_ct += prior_gate * x_talk_prior
        
        if self.prior_use_global_context:
            global_prior = self.prior_global_pma(prior_ct).squeeze(1)
            prior_ct += global_prior.unsqueeze(1)
        else:
            global_prior = None

        prior_logits = self.prior_readout(prior_ct).squeeze(-1)

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
#   class_truth:     (E, K) bool            # hypothesis correct iff all reconstructable branches match and ≥1 branch exists
#   canon_idx:       (B, E, p_max) long     # canonicalized truth indices; ascending within branch; -1 padded
#   jet_preds_tensor:(E, K, B, p_max) long  # raw predicted indices
#   jet_mult:        (E, Njets) bool        # per-jet validity
#
# Jet features (F=5) and preprocessing; jet_data has shape (E, Njets, 5) in this exact order:
#   {mass, pt, eta, phi, btag}
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



        # CANDIDATE ENCODER config with sensible defaults if missing
        self.class_embed_dim = options.class_embed_dim
        self.mask_embed_dim  = options.mask_embed_dim
        self.class_nhead     = options.class_nhead
        self.mask_nhead      = options.mask_nhead
        self.class_layers    = options.class_layers
        self.mask_layers     = options.mask_layers
        self.tr_dropout      = 0.1 # changed 0.0 -> 0.1
        self.mask_reduction  = "any"  # "any" | "mean" | "max"

        # BRANCH ENCODER config with sensible defaults if missing

        B = self.options.branch_dim
        J = self.options.jet_max_dim
        Fdim = self.options.features_dim

        # --- Transformer heads (no positional encodings) ---
        self.classifier = ClassifierTransformerHead(
            branch_dim=B, jets=J, feats=Fdim,
            class_embed_dim=self.class_embed_dim,
            nhead=self.class_nhead, num_layers=self.class_layers,
            dropout=self.tr_dropout,
        )
        self.masker = MaskerTransformerHead(
            feats=Fdim, mask_embed_dim=self.mask_embed_dim,
            nhead=self.mask_nhead, num_layers=self.mask_layers,
            dropout=self.tr_dropout,
        )

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

    @staticmethod
    def _dedup_valid_mask(jet_idx: torch.Tensor) -> torch.Tensor:

    @staticmethod
    def _multi_positive_ce(class_logits: torch.Tensor,
                           class_truth: torch.Tensor,
                           valid_mask: torch.Tensor | None = None):

    @staticmethod
    def listwise_softmax_ce(logits, truth, valid_mask):
    
    def _compiled_core(self, features_arr, pred_truth, class_truth, valid_mask):

    _compiled_core = tcompile(_compiled_core, dynamic=True)

    def forward_scm(self, batch):

    def training_step(self, batch: Batch, batch_idx: int):

    def validation_step(self, batch: Batch, batch_idx: int):