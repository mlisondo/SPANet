import torch
from torch import Tensor
from torch.nn import functional as F

@torch.jit.script
def focal_loss(log_probability: Tensor, weight_tensor: Tensor, gamma: float):
    probabilities = torch.exp(log_probability)

    zero_mask = log_probability == 0

    focal_loss_correct = -log_probability * torch.pow(1 - probabilities, gamma)

    focal_loss_incorrect = -torch.pow(probabilities, gamma) * torch.log(1 - probabilities)

    focal_scale = torch.where(
        zero_mask, 0,
        torch.where(weight_tensor == 0, focal_loss_incorrect, focal_loss_correct)
    )

    return focal_scale * (weight_tensor) ** 2

@torch.jit.script
def assignment_cross_entropy_loss(prediction: Tensor, target_data: Tensor, target_mask: Tensor, prediction_mask: Tensor, gamma: float) -> Tensor:
    batch_size, i, j, k = prediction.shape
    prediction = prediction.masked_fill(prediction_mask, 0.0)
    prediction = prediction.masked_fill(~target_mask.unsqueeze(1).unsqueeze(1).unsqueeze(1), 0.0)

    # Reshape target_data if necessary (assuming it's already [batch_size, 3])
    i_tgt, j_tgt, k_tgt = target_data[:, 0], target_data[:, 1], target_data[:, 2]
    i_tgt = torch.where(i_tgt == -1, torch.zeros_like(i_tgt), i_tgt)
    j_tgt = torch.where(j_tgt == -1, torch.zeros_like(j_tgt), j_tgt)
    k_tgt = torch.where(k_tgt == -1, torch.zeros_like(k_tgt), k_tgt)

    batch_range = torch.arange(batch_size)

    weight_tensor = torch.zeros_like(prediction)
    weight_tensor[batch_range, i_tgt, :, :] += 1
    weight_tensor[batch_range, j_tgt, :, :] += 1
    weight_tensor[batch_range, :, j_tgt, :] += 1
    weight_tensor[batch_range, :, i_tgt, :] += 1
    weight_tensor[batch_range, :, :, k_tgt] += 1

    fl = focal_loss(prediction, weight_tensor, gamma)  

    nz = fl != 0
    nz_count = torch.count_nonzero(nz, dim=[1, 2, 3])
    nz_sum = torch.sum(fl * nz, dim=[1, 2, 3])
    
    return nz_sum / nz_count.clamp(min=1)



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
