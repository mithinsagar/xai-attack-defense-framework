"""
=====================================================================
XAI Attack and Defense Framework - Few-Shot Episode Sampler
Author : Mithin Sagar S  (https://github.com/mithinsagar)
=====================================================================

Samples N-way K-shot episodes for Prototypical Network training.
"""

from __future__ import annotations

from typing import Tuple

import torch

import config


def create_episode(
    X: torch.Tensor,
    y: torch.Tensor,
    n_way: int = config.FEWSHOT_CONFIG["n_way"],
    k_shot: int = config.FEWSHOT_CONFIG["k_shot"],
    q_query: int = config.FEWSHOT_CONFIG["q_query"],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build an N-way K-shot classification episode.

    Returns
    -------
    support_x, support_y, query_x, query_y
    """
    classes = torch.unique(y)
    perm = classes[torch.randperm(len(classes))[:n_way]]

    sx, sy, qx, qy = [], [], [], []
    for i, cls in enumerate(perm):
        idx = (y == cls).nonzero(as_tuple=True)[0]
        idx = idx[torch.randperm(len(idx))]

        s_idx = idx[:k_shot]
        q_idx = idx[k_shot : k_shot + q_query]

        sx.append(X[s_idx])
        sy.append(torch.full((k_shot,), i, dtype=torch.long))

        qx.append(X[q_idx])
        qy.append(torch.full((q_query,), i, dtype=torch.long))

    return (
        torch.cat(sx),
        torch.cat(sy),
        torch.cat(qx),
        torch.cat(qy),
    )
