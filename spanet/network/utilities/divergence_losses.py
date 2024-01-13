import torch
from torch import Tensor
from torch.nn import functional as F

@torch.jit.script
def focal_loss(log_probability: Tensor, gamma: float):
    focal_scale = (1 - torch.exp(log_probability)) ** gamma
    foc_loss = -log_probability * focal_scale
    return torch.mean(foc_loss, dim=1)

@torch.jit.script
def assignment_cross_entropy_loss(prediction: Tensor, target_data: Tensor, target_mask: Tensor, prediction_mask: Tensor, gamma: float) -> Tensor:
    batch_size, i, j, k = prediction.shape
    prediction = prediction.masked_fill(prediction_mask, 0.0)

    # Reshape target_data if necessary (assuming it's already [batch_size, 3])
    i_tgt, j_tgt, k_tgt = target_data[:, 0], target_data[:, 1], target_data[:, 2]

    # Index predictions along each axis
    log_prob_i1 = prediction[torch.arange(batch_size), i_tgt, :, :].reshape(batch_size, -1)
    log_prob_i2 = prediction[torch.arange(batch_size), j_tgt, :, :].reshape(batch_size, -1)
    log_prob_j1 = prediction[torch.arange(batch_size), :, j_tgt, :].reshape(batch_size, -1)
    log_prob_j2 = prediction[torch.arange(batch_size), :, i_tgt, :].reshape(batch_size, -1)
    log_prob_k = prediction[torch.arange(batch_size), :, :, k_tgt].reshape(batch_size, -1)

    foc_loss = torch.stack([focal_loss(log_prob_i1, gamma), \
                            focal_loss(log_prob_i2, gamma), \
                            focal_loss(log_prob_j1, gamma), \
                            focal_loss(log_prob_j2, gamma), \
                            focal_loss(log_prob_k, gamma)], dim=0).mean(dim=0)
    
    foc_loss = foc_loss.masked_fill(target_mask, 0.0)

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
