from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class DeepEZDualExpertLateralityHead(nn.Module):
    """
    Laterality head on top of a DeepEZ-style trunk that outputs node logits (N,2).

    The head pools simple summary statistics separately on LH/RH node blocks and predicts
    the lesion hemisphere (L/R).
    """

    def __init__(self, trunk: nn.Module, *, n_hemi: int, topk: int = 20) -> None:
        super().__init__()
        self.trunk = trunk
        self.n_hemi = int(n_hemi)
        self.topk = int(topk)
        self.classifier = nn.Linear(6, 1)

    def forward(self, x_fc: torch.Tensor, adj: torch.Tensor, l_loc: torch.Tensor) -> torch.Tensor:
        logits_nodes, _bias = self.trunk(x_fc, adj, l_loc)
        d = logits_nodes[:, 1] - logits_nodes[:, 0]  # (N,)

        n_hemi = int(self.n_hemi)
        if d.numel() < 2 * n_hemi:
            raise ValueError(f"n_hemi={n_hemi} incompatible with n_nodes={d.numel()}")

        a = d[:n_hemi]
        b = d[n_hemi : 2 * n_hemi]

        k = int(self.topk)
        if k <= 0:
            k = 1
        k = min(k, int(a.numel()), int(b.numel()))

        feat = torch.stack(
            [
                a.mean(),
                a.max(),
                torch.topk(a, k=k, sorted=False).values.mean(),
                b.mean(),
                b.max(),
                torch.topk(b, k=k, sorted=False).values.mean(),
            ],
            dim=0,
        ).view(1, -1)
        return self.classifier(feat).view(-1)  # (1,)


class _TrunkFCOnly(nn.Module):
    def __init__(self, trunk: nn.Module) -> None:
        super().__init__()
        self.trunk = trunk

    def forward(self, x_fc: torch.Tensor, adj: torch.Tensor, l_loc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.trunk(x_fc, adj)


class _TrunkLocOnly(nn.Module):
    def __init__(self, trunk: nn.Module) -> None:
        super().__init__()
        self.trunk = trunk

    def forward(self, x_fc: torch.Tensor, adj: torch.Tensor, l_loc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.trunk(l_loc, adj)

