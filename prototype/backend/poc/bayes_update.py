from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F

from .cos_components import EditBundle, weighted_cos_logprob


@dataclass
class GaussianPosterior:
    mean: torch.Tensor  # (K,)
    cov: torch.Tensor  # (K, K)


def _neg_log_posterior(
    lambdas: torch.Tensor,
    bundle_pos: EditBundle,
    bundle_neg: EditBundle,
    prior: GaussianPosterior,
    beta: float,
) -> torch.Tensor:
    delta = weighted_cos_logprob(bundle_pos, lambdas) - weighted_cos_logprob(bundle_neg, lambdas)
    log_likelihood = F.logsigmoid(beta * delta)
    diff = lambdas - prior.mean.to(lambdas.device)
    precision = torch.linalg.inv(prior.cov.to(lambdas.device))
    quad = 0.5 * diff @ precision @ diff
    return -log_likelihood + quad


def map_update(
    bundle_pos: EditBundle,
    bundle_neg: EditBundle,
    prior: GaussianPosterior,
    beta: float = 1.0,
    steps: int = 25,
    lr: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = bundle_pos.base_lp.device
    lambdas = prior.mean.clone().detach().to(device).requires_grad_(True)
    optimizer = torch.optim.Adam([lambdas], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = _neg_log_posterior(lambdas, bundle_pos, bundle_neg, prior, beta)
        loss.backward()
        optimizer.step()
    return lambdas.detach(), loss.detach()


def adf_update(
    bundle_pos: EditBundle,
    bundle_neg: EditBundle,
    prior: GaussianPosterior,
    beta: float = 1.0,
    steps: int = 25,
    lr: float = 0.1,
) -> GaussianPosterior:
    lambda_map, _ = map_update(
        bundle_pos=bundle_pos,
        bundle_neg=bundle_neg,
        prior=prior,
        beta=beta,
        steps=steps,
        lr=lr,
    )

    lambda_map = lambda_map.detach().requires_grad_(True)
    delta = weighted_cos_logprob(bundle_pos, lambda_map) - weighted_cos_logprob(bundle_neg, lambda_map)
    prob = torch.sigmoid(beta * delta)
    grad = torch.autograd.grad(delta, lambda_map, create_graph=False)[0]
    w = (beta ** 2) * prob * (1 - prob)
    precision = torch.linalg.inv(prior.cov.to(lambda_map.device))
    precision_new = precision + w * torch.outer(grad, grad)
    cov_new = torch.linalg.inv(precision_new)

    return GaussianPosterior(mean=lambda_map.detach(), cov=cov_new.detach())
