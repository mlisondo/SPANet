import torch
from torch import Tensor
from typing import List
from torch.nn import functional as F

@torch.jit.script
def unravel_index(indices: Tensor, shape: List[int]):
    r = torch.floor_divide(indices, shape[1] * shape[2])
    c = torch.floor_divide(indices % (shape[1] * shape[2]), shape[2])
    d = indices % shape[2]
    return r, c, d

@torch.jit.script
def find_and_extract_max_idx(tensor: Tensor):
    # Compute the argmax along the last three axes
    flat_dim = tensor.shape[1] * tensor.shape[2] * tensor.shape[3]
    flat_argmax = tensor.reshape(tensor.shape[0], flat_dim).argmax(dim=-1)

    # Convert flat indices to 3D indices
    idx_3d = unravel_index(flat_argmax, tensor.shape[1:])

    return torch.stack((idx_3d[0], idx_3d[1], idx_3d[2]), dim=1)

@torch.jit.script
def assignment_cross_entropy_loss(prediction: Tensor, target_data: Tensor, target_mask: Tensor, double_mask: Tensor, gamma: float) -> Tensor:
    batch_size = prediction.shape[0]
    prediction_shape = prediction.shape[1:]

    # Remove missing jets
    target_data = target_data.clamp(0, None)

    # Find the unravelling shape required to flatten the target indices
    ravel_sizes = torch.tensor(prediction_shape).flip(0)
    ravel_sizes = torch.cumprod(ravel_sizes, 0)
    ravel_sizes = torch.div(ravel_sizes, ravel_sizes[0], rounding_mode='floor')
    # ravel_sizes = ravel_sizes // ravel_sizes[0]
    ravel_sizes = ravel_sizes.flip(0).unsqueeze(0)
    ravel_sizes = ravel_sizes.to(target_data.device)

    # Flatten the target and predicted data to be one dimensional
    ravel_target = (target_data * ravel_sizes).sum(1)
    ravel_prediction = prediction.reshape(batch_size, -1).contiguous()

    log_probability = ravel_prediction.gather(-1, ravel_target.view(-1, 1)).squeeze()
    log_probability = log_probability.masked_fill(~target_mask, 0.0)

    pos_focal = -log_probability * (1 - torch.exp(log_probability)) ** gamma
    max_idx = find_and_extract_max_idx(prediction)
    ravel_target_2 = (max_idx * ravel_sizes).sum(1)
    
    neg_log_prob = ravel_prediction.gather(-1, ravel_target.view(-1, 1)).squeeze()
    neg_prob = torch.exp(neg_log_prob)
    
    double_mask = torch.logical_and(target_mask, ~double_mask)
    triple_mask = torch.logical_and(double_mask, ~torch.all(max_idx == target_data, dim=1))
    
    neg_prob = neg_prob.masked_fill(~triple_mask, 0.0)
    neg_focal = -torch.pow(neg_prob, gamma) * torch.log(1 - neg_prob)

    return pos_focal + neg_prob


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
