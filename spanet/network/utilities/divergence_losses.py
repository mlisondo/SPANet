import torch
from torch import Tensor
from torch.nn import functional as F

@torch.jit.script
def focal_loss(log_probability: Tensor, gamma: float):
    focal_scale = (1 - torch.exp(log_probability)) ** gamma
    return -log_probability * focal_scale

@torch.jit.script
def assignment_cross_entropy_loss(prediction: Tensor, target_data: Tensor, target_mask: Tensor, prediction_mask: Tensor, gamma: float) -> Tensor:
    batch_size, i, j, k = prediction.shape
    prediction = prediction.masked_fill(prediction_mask, 0.0)

    # Reshape target_data if necessary (assuming it's already [batch_size, 3])
    i_tgt, j_tgt, k_tgt = target_data[:, 0], target_data[:, 1], target_data[:, 2]
    i_tgt = torch.where(i_tgt == -1, torch.zeros_like(i_tgt), i_tgt)
    j_tgt = torch.where(j_tgt == -1, torch.zeros_like(j_tgt), j_tgt)
    k_tgt = torch.where(k_tgt == -1, torch.zeros_like(k_tgt), k_tgt)

    batch_range = torch.arange(batch_size)

    # Index predictions along each axis
    log_prob_i = prediction[batch_range, i_tgt, :, :]
    log_prob_j = prediction[batch_range, :, j_tgt, :]
    log_prob_k = prediction[batch_range, :, :, k_tgt]

    fli = focal_loss(log_prob_i, gamma)
    flj = focal_loss(log_prob_j, gamma)
    flk = focal_loss(log_prob_k, gamma)

    fli2 = torch.clone(fli)
    flj2 = torch.clone(flj)
    flk2 = torch.clone(flk)

    fli2[batch_range, :,k_tgt] = fli2[batch_range, :,k_tgt] + flk[batch_range, i_tgt,:]
    fli2[batch_range, j_tgt,:] = fli2[batch_range, j_tgt,:] + flj[batch_range, i_tgt,:]
    flj2[batch_range, i_tgt,:] = flj2[batch_range, i_tgt,:] + fli[batch_range, j_tgt,:]
    flj2[batch_range, :,k_tgt] = flj2[batch_range, :,k_tgt] + flk[batch_range, :,j_tgt]
    flk2[batch_range, i_tgt,:] = flk2[batch_range, i_tgt,:] + fli[batch_range, :,k_tgt]
    flk2[batch_range, :,j_tgt] = flk2[batch_range, :,j_tgt] + flj[batch_range, :,k_tgt]

    mean_fli2 = fli2.mean(dim=[1, 2])
    mean_flj2 = flj2.mean(dim=[1, 2])
    mean_flk2 = flk2.mean(dim=[1, 2])

    foc_loss = (mean_fli2 + mean_flj2 + mean_flk2) / 3
    
    return foc_loss


@torch.jit.script
def kl_divergence_old(p: Tensor, log_p: Tensor, log_q: Tensor) -> Tensor:
    sum_dim = [i for i in range(1, p.ndim)]
    return torch.sum(p * log_p - p * log_q, sum_dim)


@torch.jit.script
def kl_divergence(log_prediction: Tensor, log_target: Tensor) -> Tensor:
    sum_dim = [i for i in range(1, log_prediction.ndim)]
    return torch.nansum(F.kl_div(log_prediction, log_target, reduction='none', log_target=True), dim=sum_dim)


@torch.jit.script
def jensen_shannon_divergence(log_p: Tensor, log_q: Tensor) -> Tensor:
    sum_dim = [i for i in range(1, log_p.ndim)]

    # log_m = log( (exp(log_p) + exp(log_q)) / 2 )
    log_m = torch.logsumexp(torch.stack((log_p, log_q)), dim=0) - 0.69314718056

    # TODO play around with gradient
    # log_m = log_m.detach()
    log_p = log_p.detach()
    log_q = log_q.detach()

    kl_p = F.kl_div(log_m, log_p, reduction='none', log_target=True)
    kl_q = F.kl_div(log_m, log_q, reduction='none', log_target=True)

    return torch.nansum(kl_p + kl_q, dim=sum_dim) / 2.0
