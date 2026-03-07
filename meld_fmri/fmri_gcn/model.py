from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


class GraphConvolution(nn.Module):
    """
    Minimal GCN layer (DeepEZ-style): output = A @ (X @ W) + b
    where:
      - X: (N, Fin)
      - A: (N, N) dense adjacency (ideally normalized)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = nn.Parameter(torch.empty(self.in_features, self.out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(max(self.out_features, 1))
        nn.init.uniform_(self.weight, -stdv, stdv)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -stdv, stdv)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (N, Fin), adj: (N, N)
        support = x @ self.weight
        out = adj @ support
        if self.bias is not None:
            out = out + self.bias
        return out


class GraphAttention(nn.Module):
    """
    Dense GAT layer (no torch_geometric dependency).

    Implementation notes:
      - `adj` is used as an edge mask (edges are where adj > 0).
      - Supports multi-head attention; heads are concatenated if `concat=True`,
        otherwise averaged.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        heads: int = 1,
        concat: bool = True,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.heads = int(heads)
        self.concat = bool(concat)
        self.dropout = float(dropout)
        self.negative_slope = float(negative_slope)

        # Linear projection per head: (H, Fin, Fout)
        self.weight = nn.Parameter(torch.empty(self.heads, self.in_features, self.out_features))
        # Attention parameters per head (additive form): a^T [Wh_i || Wh_j]
        self.attn_src = nn.Parameter(torch.empty(self.heads, self.out_features, 1))
        self.attn_dst = nn.Parameter(torch.empty(self.heads, self.out_features, 1))

        out_dim = self.heads * self.out_features if self.concat else self.out_features
        if bias:
            self.bias = nn.Parameter(torch.empty(out_dim))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(max(self.out_features, 1))
        nn.init.uniform_(self.weight, -stdv, stdv)
        nn.init.uniform_(self.attn_src, -stdv, stdv)
        nn.init.uniform_(self.attn_dst, -stdv, stdv)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -stdv, stdv)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (N, Fin), adj: (N, N) (dense). Use adj>0 as the edge mask.
        if x.dim() != 2:
            raise ValueError(f"Expected x as (N,Fin), got {tuple(x.shape)}")
        if adj.dim() != 2:
            raise ValueError(f"Expected adj as (N,N), got {tuple(adj.shape)}")

        n = int(x.shape[0])
        if int(adj.shape[0]) != n or int(adj.shape[1]) != n:
            raise ValueError(f"Adjacency shape {tuple(adj.shape)} incompatible with x {tuple(x.shape)}")

        # Project: (H, N, Fout)
        wh = torch.einsum("nf,hfo->hno", x, self.weight)

        # Additive attention logits:
        #   e_ij = LeakyReLU( (Wh_i @ a_src) + (Wh_j @ a_dst) )
        f1 = torch.einsum("hno,hoq->hnq", wh, self.attn_src)  # (H,N,1)
        f2 = torch.einsum("hno,hoq->hnq", wh, self.attn_dst)  # (H,N,1)
        e = f1 + f2.transpose(1, 2)  # (H,N,N)
        e = F.leaky_relu(e, negative_slope=self.negative_slope)

        # Mask non-edges with large negative values so softmax -> ~0.
        mask = (adj > 0).unsqueeze(0)  # (1,N,N)
        e = e.masked_fill(~mask, -9e15)

        alpha = torch.softmax(e, dim=-1)  # (H,N,N)
        if self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)
            wh = F.dropout(wh, p=self.dropout, training=self.training)

        out = torch.einsum("hij,hjo->hio", alpha, wh)  # (H,N,Fout)
        if self.concat:
            out2 = out.permute(1, 0, 2).reshape(n, self.heads * self.out_features)
        else:
            out2 = out.mean(dim=0)

        if self.bias is not None:
            out2 = out2 + self.bias
        return out2


@dataclass(frozen=True)
class DeepEZConfig:
    hidden1: int = 120
    hidden2: int = 50
    ssdb_hidden: int = 60
    negative_slope: float = 0.1
    gat_heads1: int = 4
    gat_dropout: float = 0.2


class DeepEZGCN(nn.Module):
    """
    Generalised DeepEZ GCN:
      - Node features: FC row (N dims) => input dim = N
      - SSDB: subject-specific detection bias from per-class logits across nodes

    Returns:
      logits: (N, 2)
      ssdb_bias: (2,)  (per-class scalar bias)
    """

    def __init__(self, n_nodes: int, in_features: int | None = None, cfg: DeepEZConfig | None = None) -> None:
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.in_features = int(in_features) if in_features is not None else int(n_nodes)
        self.cfg = cfg or DeepEZConfig()

        self.gc1 = GraphConvolution(self.in_features, self.cfg.hidden1)
        self.gc2 = GraphConvolution(self.cfg.hidden1, self.cfg.hidden2)
        self.fc_class = nn.Linear(self.cfg.hidden2, 2, bias=False)

        # SSDB head: take per-class logits across nodes (shape 2 x N), output 2 scalar biases.
        self.ssdb_fc1 = nn.Linear(self.n_nodes, self.cfg.ssdb_hidden)
        self.ssdb_fc2 = nn.Linear(self.cfg.ssdb_hidden, 1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # DeepEZ reference uses LeakyReLU throughout (including after fc_class and SSDB).
        h = F.leaky_relu(self.gc1(x, adj), negative_slope=self.cfg.negative_slope)
        h = F.leaky_relu(self.gc2(h, adj), negative_slope=self.cfg.negative_slope)
        logits = F.leaky_relu(self.fc_class(h), negative_slope=self.cfg.negative_slope)  # (N,2)

        # SSDB: per-class bias computed from logits distribution across nodes.
        # logits.T: (2,N) -> fc1 expects (.., N)
        bias = F.leaky_relu(self.ssdb_fc1(logits.T), negative_slope=self.cfg.negative_slope)  # (2,ssdb_hidden)
        bias = F.leaky_relu(self.ssdb_fc2(bias), negative_slope=self.cfg.negative_slope)  # (2,1)
        bias2 = bias.T  # (1,2)
        logits = logits + bias2  # broadcast

        return logits, bias.squeeze(-1).squeeze(-1)  # (N,2), (2,)


class DeepEZGAT(nn.Module):
    """
    DeepEZ-style model but with GAT layers (dense attention) instead of GCN layers.

    Design choices:
      - Layer1: multi-head attention with concatenation to reach cfg.hidden1 dims.
      - Layer2: single-head attention to cfg.hidden2 dims.
      - Keep SSDB head identical to DeepEZGCN for subject-specific bias correction.
    """

    def __init__(self, n_nodes: int, in_features: int | None = None, cfg: DeepEZConfig | None = None) -> None:
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.in_features = int(in_features) if in_features is not None else int(n_nodes)
        self.cfg = cfg or DeepEZConfig()

        h1 = int(self.cfg.hidden1)
        heads1 = int(self.cfg.gat_heads1)
        if heads1 <= 0:
            raise ValueError(f"gat_heads1 must be >0, got {heads1}")
        if h1 % heads1 != 0:
            raise ValueError(f"hidden1={h1} must be divisible by gat_heads1={heads1}")
        h1_per = h1 // heads1

        self.gat1 = GraphAttention(
            self.in_features,
            h1_per,
            heads=heads1,
            concat=True,
            dropout=float(self.cfg.gat_dropout),
            negative_slope=float(self.cfg.negative_slope),
        )
        self.gat2 = GraphAttention(
            h1,
            int(self.cfg.hidden2),
            heads=1,
            concat=False,
            dropout=float(self.cfg.gat_dropout),
            negative_slope=float(self.cfg.negative_slope),
        )
        self.fc_class = nn.Linear(int(self.cfg.hidden2), 2, bias=False)

        self.ssdb_fc1 = nn.Linear(self.n_nodes, int(self.cfg.ssdb_hidden))
        self.ssdb_fc2 = nn.Linear(int(self.cfg.ssdb_hidden), 1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.leaky_relu(self.gat1(x, adj), negative_slope=self.cfg.negative_slope)
        h = F.leaky_relu(self.gat2(h, adj), negative_slope=self.cfg.negative_slope)
        logits = F.leaky_relu(self.fc_class(h), negative_slope=self.cfg.negative_slope)  # (N,2)

        bias = F.leaky_relu(self.ssdb_fc1(logits.T), negative_slope=self.cfg.negative_slope)  # (2,ssdb_hidden)
        bias = F.leaky_relu(self.ssdb_fc2(bias), negative_slope=self.cfg.negative_slope)  # (2,1)
        bias2 = bias.T  # (1,2)
        logits = logits + bias2
        return logits, bias.squeeze(-1).squeeze(-1)


class DeepEZGCNFiLM(nn.Module):
    """
    DeepEZ GCN with a local-feature FiLM residual branch.

    Intended use:
      - Trunk operates on FC node features (X_fc) as in S4 baseline.
      - Local branch consumes per-node local features (L_loc) and modulates trunk hidden state (H2)
        via FiLM (gamma,beta) and a reliability gate (sigmoid) attenuated by parcel coverage.

    forward:
      logits, ssdb_bias = model(x_fc, adj, l_loc)
    """

    def __init__(
        self,
        *,
        n_nodes: int,
        in_features: int,
        local_dim: int,
        cfg: DeepEZConfig | None = None,
        film_z: int = 32,
        gate_bias: float = -2.0,
        coverage_power: float = 1.0,
        zero_init_film: bool = True,
    ) -> None:
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.in_features = int(in_features)
        self.local_dim = int(local_dim)
        self.cfg = cfg or DeepEZConfig()
        self.film_z = int(film_z)
        self.coverage_power = float(coverage_power)

        # Trunk (same as DeepEZGCN)
        self.gc1 = GraphConvolution(self.in_features, int(self.cfg.hidden1))
        self.gc2 = GraphConvolution(int(self.cfg.hidden1), int(self.cfg.hidden2))
        self.fc_class = nn.Linear(int(self.cfg.hidden2), 2, bias=False)

        # Local branch
        self.loc_fc1 = nn.Linear(self.local_dim, self.film_z)
        self.loc_fc2 = nn.Linear(self.film_z, self.film_z)
        self.film_gamma = nn.Linear(self.film_z, int(self.cfg.hidden2))
        self.film_beta = nn.Linear(self.film_z, int(self.cfg.hidden2))
        self.film_gate = nn.Linear(self.film_z, 1)

        # SSDB head (same as DeepEZGCN)
        self.ssdb_fc1 = nn.Linear(self.n_nodes, int(self.cfg.ssdb_hidden))
        self.ssdb_fc2 = nn.Linear(int(self.cfg.ssdb_hidden), 1)

        if bool(zero_init_film):
            nn.init.zeros_(self.film_gamma.weight)
            nn.init.zeros_(self.film_gamma.bias)
            nn.init.zeros_(self.film_beta.weight)
            nn.init.zeros_(self.film_beta.bias)
            nn.init.zeros_(self.film_gate.weight)
            nn.init.constant_(self.film_gate.bias, float(gate_bias))

    def forward(self, x_fc: torch.Tensor, adj: torch.Tensor, l_loc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if l_loc is None:
            raise ValueError("DeepEZGCNFiLM requires l_loc (local features)")
        if x_fc.dim() != 2 or l_loc.dim() != 2:
            raise ValueError(f"Expected 2D tensors: x_fc={tuple(x_fc.shape)}, l_loc={tuple(l_loc.shape)}")
        if int(x_fc.shape[0]) != int(l_loc.shape[0]):
            raise ValueError(f"Node mismatch: x_fc N={x_fc.shape[0]} vs l_loc N={l_loc.shape[0]}")

        # Trunk
        h = F.leaky_relu(self.gc1(x_fc, adj), negative_slope=self.cfg.negative_slope)
        h = F.leaky_relu(self.gc2(h, adj), negative_slope=self.cfg.negative_slope)  # (N,hidden2)

        # Local encoding
        z = F.leaky_relu(self.loc_fc1(l_loc), negative_slope=self.cfg.negative_slope)
        z = F.leaky_relu(self.loc_fc2(z), negative_slope=self.cfg.negative_slope)  # (N,film_z)

        gamma = self.film_gamma(z)  # (N,hidden2)
        beta = self.film_beta(z)  # (N,hidden2)

        # Reliability gate attenuated by coverage (assume last channel is parcel coverage in [0,1]).
        g = torch.sigmoid(self.film_gate(z))  # (N,1)
        cov = l_loc[:, -1].clamp(min=0.0, max=1.0)
        if self.coverage_power != 0.0:
            cov = cov.pow(self.coverage_power)
        g = g * cov.unsqueeze(1)

        delta = gamma * h + beta
        h_fused = h + g * delta

        logits = F.leaky_relu(self.fc_class(h_fused), negative_slope=self.cfg.negative_slope)  # (N,2)

        bias = F.leaky_relu(self.ssdb_fc1(logits.T), negative_slope=self.cfg.negative_slope)  # (2,ssdb_hidden)
        bias = F.leaky_relu(self.ssdb_fc2(bias), negative_slope=self.cfg.negative_slope)  # (2,1)
        logits = logits + bias.T  # broadcast to (N,2)
        return logits, bias.squeeze(-1).squeeze(-1)


class DeepEZGATFiLM(nn.Module):
    """
    DeepEZ GAT trunk with local-feature FiLM residual branch (see DeepEZGCNFiLM).
    """

    def __init__(
        self,
        *,
        n_nodes: int,
        in_features: int,
        local_dim: int,
        cfg: DeepEZConfig | None = None,
        film_z: int = 32,
        gate_bias: float = -2.0,
        coverage_power: float = 1.0,
        zero_init_film: bool = True,
    ) -> None:
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.in_features = int(in_features)
        self.local_dim = int(local_dim)
        self.cfg = cfg or DeepEZConfig()
        self.film_z = int(film_z)
        self.coverage_power = float(coverage_power)

        h1 = int(self.cfg.hidden1)
        heads1 = int(self.cfg.gat_heads1)
        if heads1 <= 0:
            raise ValueError(f"gat_heads1 must be >0, got {heads1}")
        if h1 % heads1 != 0:
            raise ValueError(f"hidden1={h1} must be divisible by gat_heads1={heads1}")
        h1_per = h1 // heads1

        self.gat1 = GraphAttention(
            self.in_features,
            h1_per,
            heads=heads1,
            concat=True,
            dropout=float(self.cfg.gat_dropout),
            negative_slope=float(self.cfg.negative_slope),
        )
        self.gat2 = GraphAttention(
            h1,
            int(self.cfg.hidden2),
            heads=1,
            concat=False,
            dropout=float(self.cfg.gat_dropout),
            negative_slope=float(self.cfg.negative_slope),
        )
        self.fc_class = nn.Linear(int(self.cfg.hidden2), 2, bias=False)

        self.loc_fc1 = nn.Linear(self.local_dim, self.film_z)
        self.loc_fc2 = nn.Linear(self.film_z, self.film_z)
        self.film_gamma = nn.Linear(self.film_z, int(self.cfg.hidden2))
        self.film_beta = nn.Linear(self.film_z, int(self.cfg.hidden2))
        self.film_gate = nn.Linear(self.film_z, 1)

        self.ssdb_fc1 = nn.Linear(self.n_nodes, int(self.cfg.ssdb_hidden))
        self.ssdb_fc2 = nn.Linear(int(self.cfg.ssdb_hidden), 1)

        if bool(zero_init_film):
            nn.init.zeros_(self.film_gamma.weight)
            nn.init.zeros_(self.film_gamma.bias)
            nn.init.zeros_(self.film_beta.weight)
            nn.init.zeros_(self.film_beta.bias)
            nn.init.zeros_(self.film_gate.weight)
            nn.init.constant_(self.film_gate.bias, float(gate_bias))

    def forward(self, x_fc: torch.Tensor, adj: torch.Tensor, l_loc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if l_loc is None:
            raise ValueError("DeepEZGATFiLM requires l_loc (local features)")
        if x_fc.dim() != 2 or l_loc.dim() != 2:
            raise ValueError(f"Expected 2D tensors: x_fc={tuple(x_fc.shape)}, l_loc={tuple(l_loc.shape)}")
        if int(x_fc.shape[0]) != int(l_loc.shape[0]):
            raise ValueError(f"Node mismatch: x_fc N={x_fc.shape[0]} vs l_loc N={l_loc.shape[0]}")

        h = F.leaky_relu(self.gat1(x_fc, adj), negative_slope=self.cfg.negative_slope)
        h = F.leaky_relu(self.gat2(h, adj), negative_slope=self.cfg.negative_slope)  # (N,hidden2)

        z = F.leaky_relu(self.loc_fc1(l_loc), negative_slope=self.cfg.negative_slope)
        z = F.leaky_relu(self.loc_fc2(z), negative_slope=self.cfg.negative_slope)

        gamma = self.film_gamma(z)
        beta = self.film_beta(z)
        g = torch.sigmoid(self.film_gate(z))  # (N,1)
        cov = l_loc[:, -1].clamp(min=0.0, max=1.0)
        if self.coverage_power != 0.0:
            cov = cov.pow(self.coverage_power)
        g = g * cov.unsqueeze(1)

        delta = gamma * h + beta
        h_fused = h + g * delta
        logits = F.leaky_relu(self.fc_class(h_fused), negative_slope=self.cfg.negative_slope)  # (N,2)

        bias = F.leaky_relu(self.ssdb_fc1(logits.T), negative_slope=self.cfg.negative_slope)
        bias = F.leaky_relu(self.ssdb_fc2(bias), negative_slope=self.cfg.negative_slope)
        logits = logits + bias.T
        return logits, bias.squeeze(-1).squeeze(-1)


class DeepEZGCNFiLMSubGate(nn.Module):
    """
    FiLM fusion with an additional subject-level gate (w_sub) predicted from trunk logits distribution.

    Motivation:
      - local branch helps some subjects but harms others; a subject gate allows the model
        to down-weight local modulation when it is unreliable.
    """

    def __init__(
        self,
        *,
        n_nodes: int,
        in_features: int,
        local_dim: int,
        cfg: DeepEZConfig | None = None,
        film_z: int = 32,
        gate_bias: float = -2.0,
        coverage_power: float = 1.0,
        subgate_bias: float = -2.0,
        subgate_floor: float = 0.0,
        zero_init_film: bool = True,
        zero_init_subgate: bool = True,
    ) -> None:
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.in_features = int(in_features)
        self.local_dim = int(local_dim)
        self.cfg = cfg or DeepEZConfig()
        self.film_z = int(film_z)
        self.coverage_power = float(coverage_power)
        self.subgate_floor = float(subgate_floor)

        self.gc1 = GraphConvolution(self.in_features, int(self.cfg.hidden1))
        self.gc2 = GraphConvolution(int(self.cfg.hidden1), int(self.cfg.hidden2))
        self.fc_class = nn.Linear(int(self.cfg.hidden2), 2, bias=False)

        self.loc_fc1 = nn.Linear(self.local_dim, self.film_z)
        self.loc_fc2 = nn.Linear(self.film_z, self.film_z)
        self.film_gamma = nn.Linear(self.film_z, int(self.cfg.hidden2))
        self.film_beta = nn.Linear(self.film_z, int(self.cfg.hidden2))
        self.film_gate = nn.Linear(self.film_z, 1)

        self.ssdb_fc1 = nn.Linear(self.n_nodes, int(self.cfg.ssdb_hidden))
        self.ssdb_fc2 = nn.Linear(int(self.cfg.ssdb_hidden), 1)

        # Subject gate from SSDB hidden features computed on trunk-only logits.
        self.subgate_fc = nn.Linear(2 * int(self.cfg.ssdb_hidden), 1)

        if bool(zero_init_film):
            nn.init.zeros_(self.film_gamma.weight)
            nn.init.zeros_(self.film_gamma.bias)
            nn.init.zeros_(self.film_beta.weight)
            nn.init.zeros_(self.film_beta.bias)
            nn.init.zeros_(self.film_gate.weight)
            nn.init.constant_(self.film_gate.bias, float(gate_bias))

        if bool(zero_init_subgate):
            nn.init.zeros_(self.subgate_fc.weight)
            nn.init.constant_(self.subgate_fc.bias, float(subgate_bias))

    def forward(self, x_fc: torch.Tensor, adj: torch.Tensor, l_loc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if l_loc is None:
            raise ValueError("DeepEZGCNFiLMSubGate requires l_loc (local features)")
        if x_fc.dim() != 2 or l_loc.dim() != 2:
            raise ValueError(f"Expected 2D tensors: x_fc={tuple(x_fc.shape)}, l_loc={tuple(l_loc.shape)}")
        if int(x_fc.shape[0]) != int(l_loc.shape[0]):
            raise ValueError(f"Node mismatch: x_fc N={x_fc.shape[0]} vs l_loc N={l_loc.shape[0]}")

        h = F.leaky_relu(self.gc1(x_fc, adj), negative_slope=self.cfg.negative_slope)
        h = F.leaky_relu(self.gc2(h, adj), negative_slope=self.cfg.negative_slope)  # (N,hidden2)

        # Subject gate from trunk-only logits distribution.
        logits0 = F.leaky_relu(self.fc_class(h), negative_slope=self.cfg.negative_slope)  # (N,2)
        ssdb_h0 = F.leaky_relu(self.ssdb_fc1(logits0.T), negative_slope=self.cfg.negative_slope)  # (2,ssdb_hidden)
        w_sub = torch.sigmoid(self.subgate_fc(ssdb_h0.reshape(1, -1)))  # (1,1)
        if self.subgate_floor > 0.0:
            w_sub = self.subgate_floor + (1.0 - self.subgate_floor) * w_sub

        z = F.leaky_relu(self.loc_fc1(l_loc), negative_slope=self.cfg.negative_slope)
        z = F.leaky_relu(self.loc_fc2(z), negative_slope=self.cfg.negative_slope)
        gamma = self.film_gamma(z)
        beta = self.film_beta(z)

        g = torch.sigmoid(self.film_gate(z))
        cov = l_loc[:, -1].clamp(min=0.0, max=1.0)
        if self.coverage_power != 0.0:
            cov = cov.pow(self.coverage_power)
        g = g * cov.unsqueeze(1)
        g = g * w_sub  # subject-level attenuation

        delta = gamma * h + beta
        h_fused = h + g * delta
        logits = F.leaky_relu(self.fc_class(h_fused), negative_slope=self.cfg.negative_slope)

        bias = F.leaky_relu(self.ssdb_fc1(logits.T), negative_slope=self.cfg.negative_slope)
        bias = F.leaky_relu(self.ssdb_fc2(bias), negative_slope=self.cfg.negative_slope)
        logits = logits + bias.T
        return logits, bias.squeeze(-1).squeeze(-1)


class DeepEZGATFiLMSubGate(nn.Module):
    """
    GAT variant of FiLM+subject gate (see DeepEZGCNFiLMSubGate).
    """

    def __init__(
        self,
        *,
        n_nodes: int,
        in_features: int,
        local_dim: int,
        cfg: DeepEZConfig | None = None,
        film_z: int = 32,
        gate_bias: float = -2.0,
        coverage_power: float = 1.0,
        subgate_bias: float = -2.0,
        subgate_floor: float = 0.0,
        zero_init_film: bool = True,
        zero_init_subgate: bool = True,
    ) -> None:
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.in_features = int(in_features)
        self.local_dim = int(local_dim)
        self.cfg = cfg or DeepEZConfig()
        self.film_z = int(film_z)
        self.coverage_power = float(coverage_power)
        self.subgate_floor = float(subgate_floor)

        h1 = int(self.cfg.hidden1)
        heads1 = int(self.cfg.gat_heads1)
        if heads1 <= 0:
            raise ValueError(f"gat_heads1 must be >0, got {heads1}")
        if h1 % heads1 != 0:
            raise ValueError(f"hidden1={h1} must be divisible by gat_heads1={heads1}")
        h1_per = h1 // heads1

        self.gat1 = GraphAttention(
            self.in_features,
            h1_per,
            heads=heads1,
            concat=True,
            dropout=float(self.cfg.gat_dropout),
            negative_slope=float(self.cfg.negative_slope),
        )
        self.gat2 = GraphAttention(
            h1,
            int(self.cfg.hidden2),
            heads=1,
            concat=False,
            dropout=float(self.cfg.gat_dropout),
            negative_slope=float(self.cfg.negative_slope),
        )
        self.fc_class = nn.Linear(int(self.cfg.hidden2), 2, bias=False)

        self.loc_fc1 = nn.Linear(self.local_dim, self.film_z)
        self.loc_fc2 = nn.Linear(self.film_z, self.film_z)
        self.film_gamma = nn.Linear(self.film_z, int(self.cfg.hidden2))
        self.film_beta = nn.Linear(self.film_z, int(self.cfg.hidden2))
        self.film_gate = nn.Linear(self.film_z, 1)

        self.ssdb_fc1 = nn.Linear(self.n_nodes, int(self.cfg.ssdb_hidden))
        self.ssdb_fc2 = nn.Linear(int(self.cfg.ssdb_hidden), 1)
        self.subgate_fc = nn.Linear(2 * int(self.cfg.ssdb_hidden), 1)

        if bool(zero_init_film):
            nn.init.zeros_(self.film_gamma.weight)
            nn.init.zeros_(self.film_gamma.bias)
            nn.init.zeros_(self.film_beta.weight)
            nn.init.zeros_(self.film_beta.bias)
            nn.init.zeros_(self.film_gate.weight)
            nn.init.constant_(self.film_gate.bias, float(gate_bias))
        if bool(zero_init_subgate):
            nn.init.zeros_(self.subgate_fc.weight)
            nn.init.constant_(self.subgate_fc.bias, float(subgate_bias))

    def forward(self, x_fc: torch.Tensor, adj: torch.Tensor, l_loc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if l_loc is None:
            raise ValueError("DeepEZGATFiLMSubGate requires l_loc (local features)")
        if x_fc.dim() != 2 or l_loc.dim() != 2:
            raise ValueError(f"Expected 2D tensors: x_fc={tuple(x_fc.shape)}, l_loc={tuple(l_loc.shape)}")
        if int(x_fc.shape[0]) != int(l_loc.shape[0]):
            raise ValueError(f"Node mismatch: x_fc N={x_fc.shape[0]} vs l_loc N={l_loc.shape[0]}")

        h = F.leaky_relu(self.gat1(x_fc, adj), negative_slope=self.cfg.negative_slope)
        h = F.leaky_relu(self.gat2(h, adj), negative_slope=self.cfg.negative_slope)

        logits0 = F.leaky_relu(self.fc_class(h), negative_slope=self.cfg.negative_slope)
        ssdb_h0 = F.leaky_relu(self.ssdb_fc1(logits0.T), negative_slope=self.cfg.negative_slope)
        w_sub = torch.sigmoid(self.subgate_fc(ssdb_h0.reshape(1, -1)))
        if self.subgate_floor > 0.0:
            w_sub = self.subgate_floor + (1.0 - self.subgate_floor) * w_sub

        z = F.leaky_relu(self.loc_fc1(l_loc), negative_slope=self.cfg.negative_slope)
        z = F.leaky_relu(self.loc_fc2(z), negative_slope=self.cfg.negative_slope)
        gamma = self.film_gamma(z)
        beta = self.film_beta(z)
        g = torch.sigmoid(self.film_gate(z))
        cov = l_loc[:, -1].clamp(min=0.0, max=1.0)
        if self.coverage_power != 0.0:
            cov = cov.pow(self.coverage_power)
        g = g * cov.unsqueeze(1)
        g = g * w_sub

        delta = gamma * h + beta
        h_fused = h + g * delta
        logits = F.leaky_relu(self.fc_class(h_fused), negative_slope=self.cfg.negative_slope)

        bias = F.leaky_relu(self.ssdb_fc1(logits.T), negative_slope=self.cfg.negative_slope)
        bias = F.leaky_relu(self.ssdb_fc2(bias), negative_slope=self.cfg.negative_slope)
        logits = logits + bias.T
        return logits, bias.squeeze(-1).squeeze(-1)


class DeepEZGCNFiLMHemiSSDB(nn.Module):
    """
    FiLM fusion with hemisphere-aware SSDB (separate bias for ipsi/contra node blocks).

    Notes:
      - Node ordering is assumed to be: [ipsi_hemi_nodes, contra_hemi_nodes, midline_nodes].
      - Midline bias uses the average of ipsi/contra biases.
    """

    def __init__(
        self,
        *,
        n_nodes: int,
        in_features: int,
        local_dim: int,
        n_mid: int = 0,
        cfg: DeepEZConfig | None = None,
        film_z: int = 32,
        gate_bias: float = -2.0,
        coverage_power: float = 1.0,
        zero_init_film: bool = True,
    ) -> None:
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.in_features = int(in_features)
        self.local_dim = int(local_dim)
        self.n_mid = int(n_mid)
        self.cfg = cfg or DeepEZConfig()
        self.film_z = int(film_z)
        self.coverage_power = float(coverage_power)

        if self.n_mid < 0 or self.n_mid > self.n_nodes:
            raise ValueError(f"Invalid n_mid={self.n_mid} for n_nodes={self.n_nodes}")
        if (self.n_nodes - self.n_mid) % 2 != 0:
            raise ValueError(f"n_nodes - n_mid must be even (n_nodes={self.n_nodes}, n_mid={self.n_mid})")
        self.n_hemi = int((self.n_nodes - self.n_mid) // 2)
        if 2 * self.n_hemi + self.n_mid != self.n_nodes:
            raise ValueError("Invalid hemi split")

        self.gc1 = GraphConvolution(self.in_features, int(self.cfg.hidden1))
        self.gc2 = GraphConvolution(int(self.cfg.hidden1), int(self.cfg.hidden2))
        self.fc_class = nn.Linear(int(self.cfg.hidden2), 2, bias=False)

        self.loc_fc1 = nn.Linear(self.local_dim, self.film_z)
        self.loc_fc2 = nn.Linear(self.film_z, self.film_z)
        self.film_gamma = nn.Linear(self.film_z, int(self.cfg.hidden2))
        self.film_beta = nn.Linear(self.film_z, int(self.cfg.hidden2))
        self.film_gate = nn.Linear(self.film_z, 1)

        # Hemisphere SSDB head: shared weights for ipsi and contra blocks.
        self.ssdb_hemi_fc1 = nn.Linear(self.n_hemi, int(self.cfg.ssdb_hidden))
        self.ssdb_hemi_fc2 = nn.Linear(int(self.cfg.ssdb_hidden), 1)

        if bool(zero_init_film):
            nn.init.zeros_(self.film_gamma.weight)
            nn.init.zeros_(self.film_gamma.bias)
            nn.init.zeros_(self.film_beta.weight)
            nn.init.zeros_(self.film_beta.bias)
            nn.init.zeros_(self.film_gate.weight)
            nn.init.constant_(self.film_gate.bias, float(gate_bias))

    def forward(self, x_fc: torch.Tensor, adj: torch.Tensor, l_loc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if l_loc is None:
            raise ValueError("DeepEZGCNFiLMHemiSSDB requires l_loc (local features)")
        if x_fc.dim() != 2 or l_loc.dim() != 2:
            raise ValueError(f"Expected 2D tensors: x_fc={tuple(x_fc.shape)}, l_loc={tuple(l_loc.shape)}")
        if int(x_fc.shape[0]) != int(l_loc.shape[0]):
            raise ValueError(f"Node mismatch: x_fc N={x_fc.shape[0]} vs l_loc N={l_loc.shape[0]}")

        h = F.leaky_relu(self.gc1(x_fc, adj), negative_slope=self.cfg.negative_slope)
        h = F.leaky_relu(self.gc2(h, adj), negative_slope=self.cfg.negative_slope)

        z = F.leaky_relu(self.loc_fc1(l_loc), negative_slope=self.cfg.negative_slope)
        z = F.leaky_relu(self.loc_fc2(z), negative_slope=self.cfg.negative_slope)
        gamma = self.film_gamma(z)
        beta = self.film_beta(z)
        g = torch.sigmoid(self.film_gate(z))
        cov = l_loc[:, -1].clamp(min=0.0, max=1.0)
        if self.coverage_power != 0.0:
            cov = cov.pow(self.coverage_power)
        g = g * cov.unsqueeze(1)

        delta = gamma * h + beta
        h_fused = h + g * delta
        logits = F.leaky_relu(self.fc_class(h_fused), negative_slope=self.cfg.negative_slope)  # (N,2)

        t = logits.T  # (2,N)
        ipsi = t[:, : self.n_hemi]
        contra = t[:, self.n_hemi : 2 * self.n_hemi]
        h_ipsi = F.leaky_relu(self.ssdb_hemi_fc1(ipsi), negative_slope=self.cfg.negative_slope)
        h_contra = F.leaky_relu(self.ssdb_hemi_fc1(contra), negative_slope=self.cfg.negative_slope)
        b_ipsi = F.leaky_relu(self.ssdb_hemi_fc2(h_ipsi), negative_slope=self.cfg.negative_slope)  # (2,1)
        b_contra = F.leaky_relu(self.ssdb_hemi_fc2(h_contra), negative_slope=self.cfg.negative_slope)  # (2,1)
        b_mid = 0.5 * (b_ipsi + b_contra)
        bias_nodes = torch.cat(
            [
                b_ipsi.T.expand(self.n_hemi, 2),
                b_contra.T.expand(self.n_hemi, 2),
                b_mid.T.expand(self.n_mid, 2) if self.n_mid > 0 else logits.new_zeros((0, 2)),
            ],
            dim=0,
        )
        logits = logits + bias_nodes

        bias_avg = b_mid.squeeze(-1).squeeze(-1)
        return logits, bias_avg


class DeepEZGATFiLMHemiSSDB(nn.Module):
    """
    GAT variant of FiLM + hemisphere-aware SSDB (see DeepEZGCNFiLMHemiSSDB).
    """

    def __init__(
        self,
        *,
        n_nodes: int,
        in_features: int,
        local_dim: int,
        n_mid: int = 0,
        cfg: DeepEZConfig | None = None,
        film_z: int = 32,
        gate_bias: float = -2.0,
        coverage_power: float = 1.0,
        zero_init_film: bool = True,
    ) -> None:
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.in_features = int(in_features)
        self.local_dim = int(local_dim)
        self.n_mid = int(n_mid)
        self.cfg = cfg or DeepEZConfig()
        self.film_z = int(film_z)
        self.coverage_power = float(coverage_power)

        if self.n_mid < 0 or self.n_mid > self.n_nodes:
            raise ValueError(f"Invalid n_mid={self.n_mid} for n_nodes={self.n_nodes}")
        if (self.n_nodes - self.n_mid) % 2 != 0:
            raise ValueError(f"n_nodes - n_mid must be even (n_nodes={self.n_nodes}, n_mid={self.n_mid})")
        self.n_hemi = int((self.n_nodes - self.n_mid) // 2)
        if 2 * self.n_hemi + self.n_mid != self.n_nodes:
            raise ValueError("Invalid hemi split")

        h1 = int(self.cfg.hidden1)
        heads1 = int(self.cfg.gat_heads1)
        if heads1 <= 0:
            raise ValueError(f"gat_heads1 must be >0, got {heads1}")
        if h1 % heads1 != 0:
            raise ValueError(f"hidden1={h1} must be divisible by gat_heads1={heads1}")
        h1_per = h1 // heads1

        self.gat1 = GraphAttention(
            self.in_features,
            h1_per,
            heads=heads1,
            concat=True,
            dropout=float(self.cfg.gat_dropout),
            negative_slope=float(self.cfg.negative_slope),
        )
        self.gat2 = GraphAttention(
            h1,
            int(self.cfg.hidden2),
            heads=1,
            concat=False,
            dropout=float(self.cfg.gat_dropout),
            negative_slope=float(self.cfg.negative_slope),
        )
        self.fc_class = nn.Linear(int(self.cfg.hidden2), 2, bias=False)

        self.loc_fc1 = nn.Linear(self.local_dim, self.film_z)
        self.loc_fc2 = nn.Linear(self.film_z, self.film_z)
        self.film_gamma = nn.Linear(self.film_z, int(self.cfg.hidden2))
        self.film_beta = nn.Linear(self.film_z, int(self.cfg.hidden2))
        self.film_gate = nn.Linear(self.film_z, 1)

        self.ssdb_hemi_fc1 = nn.Linear(self.n_hemi, int(self.cfg.ssdb_hidden))
        self.ssdb_hemi_fc2 = nn.Linear(int(self.cfg.ssdb_hidden), 1)

        if bool(zero_init_film):
            nn.init.zeros_(self.film_gamma.weight)
            nn.init.zeros_(self.film_gamma.bias)
            nn.init.zeros_(self.film_beta.weight)
            nn.init.zeros_(self.film_beta.bias)
            nn.init.zeros_(self.film_gate.weight)
            nn.init.constant_(self.film_gate.bias, float(gate_bias))

    def forward(self, x_fc: torch.Tensor, adj: torch.Tensor, l_loc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if l_loc is None:
            raise ValueError("DeepEZGATFiLMHemiSSDB requires l_loc (local features)")
        if x_fc.dim() != 2 or l_loc.dim() != 2:
            raise ValueError(f"Expected 2D tensors: x_fc={tuple(x_fc.shape)}, l_loc={tuple(l_loc.shape)}")
        if int(x_fc.shape[0]) != int(l_loc.shape[0]):
            raise ValueError(f"Node mismatch: x_fc N={x_fc.shape[0]} vs l_loc N={l_loc.shape[0]}")

        h = F.leaky_relu(self.gat1(x_fc, adj), negative_slope=self.cfg.negative_slope)
        h = F.leaky_relu(self.gat2(h, adj), negative_slope=self.cfg.negative_slope)

        z = F.leaky_relu(self.loc_fc1(l_loc), negative_slope=self.cfg.negative_slope)
        z = F.leaky_relu(self.loc_fc2(z), negative_slope=self.cfg.negative_slope)
        gamma = self.film_gamma(z)
        beta = self.film_beta(z)
        g = torch.sigmoid(self.film_gate(z))
        cov = l_loc[:, -1].clamp(min=0.0, max=1.0)
        if self.coverage_power != 0.0:
            cov = cov.pow(self.coverage_power)
        g = g * cov.unsqueeze(1)

        delta = gamma * h + beta
        h_fused = h + g * delta
        logits = F.leaky_relu(self.fc_class(h_fused), negative_slope=self.cfg.negative_slope)

        t = logits.T
        ipsi = t[:, : self.n_hemi]
        contra = t[:, self.n_hemi : 2 * self.n_hemi]
        h_ipsi = F.leaky_relu(self.ssdb_hemi_fc1(ipsi), negative_slope=self.cfg.negative_slope)
        h_contra = F.leaky_relu(self.ssdb_hemi_fc1(contra), negative_slope=self.cfg.negative_slope)
        b_ipsi = F.leaky_relu(self.ssdb_hemi_fc2(h_ipsi), negative_slope=self.cfg.negative_slope)
        b_contra = F.leaky_relu(self.ssdb_hemi_fc2(h_contra), negative_slope=self.cfg.negative_slope)
        b_mid = 0.5 * (b_ipsi + b_contra)
        bias_nodes = torch.cat(
            [
                b_ipsi.T.expand(self.n_hemi, 2),
                b_contra.T.expand(self.n_hemi, 2),
                b_mid.T.expand(self.n_mid, 2) if self.n_mid > 0 else logits.new_zeros((0, 2)),
            ],
            dim=0,
        )
        logits = logits + bias_nodes

        bias_avg = b_mid.squeeze(-1).squeeze(-1)
        return logits, bias_avg


class DeepEZDualExpertGCN(nn.Module):
    """
    Dual-expert model:
      - FC expert: DeepEZGCN on FC fingerprint features (with its own SSDB)
      - Local expert: DeepEZGCN on local pooled features (with its own SSDB)

    Fused logits:
      logits = (fc_raw + bias_fc) + w_sub * g_node * (loc_raw) + w_sub * bias_loc
    where:
      - w_sub is a subject-level gate predicted from [bias_fc, bias_loc, mean_coverage]
      - g_node is coverage^coverage_power (node-wise attenuation)
    """

    def __init__(
        self,
        *,
        n_nodes: int,
        in_features_fc: int,
        in_features_loc: int,
        cfg: DeepEZConfig | None = None,
        gate_hidden: int = 16,
        gate_bias: float = -2.0,
        gate_floor: float = 0.0,
        gate_use_ssdb_hidden_stats: bool = False,
        coverage_power: float = 1.0,
        bias_mode: str = "shared",
        bias_gate_hidden: int = 16,
        bias_gate_bias: float = -4.0,
        bias_gate_floor: float = 0.0,
        node_gate: str = "coverage",
        node_gate_bias: float = 2.0,
        node_gate_floor: float = 0.0,
        zero_init_gate: bool = True,
    ) -> None:
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.in_features_fc = int(in_features_fc)
        self.in_features_loc = int(in_features_loc)
        self.cfg = cfg or DeepEZConfig()
        self.coverage_power = float(coverage_power)
        self.gate_floor = float(gate_floor)
        self.gate_use_ssdb_hidden_stats = bool(gate_use_ssdb_hidden_stats)

        bias_mode = str(bias_mode).strip().lower()
        if bias_mode not in {"shared", "none", "mean_gnode", "square", "mlp"}:
            raise ValueError(f"Invalid bias_mode={bias_mode} (expected shared|none|mean_gnode|square|mlp)")
        self.bias_mode = bias_mode
        self.bias_gate_floor = float(bias_gate_floor)

        node_gate = str(node_gate).strip().lower()
        if node_gate not in {"coverage", "learned"}:
            raise ValueError(f"Invalid node_gate={node_gate} (expected coverage|learned)")
        self.node_gate = node_gate
        self.node_gate_floor = float(node_gate_floor)

        self.fc_expert = DeepEZGCN(n_nodes=self.n_nodes, in_features=self.in_features_fc, cfg=self.cfg)
        self.loc_expert = DeepEZGCN(n_nodes=self.n_nodes, in_features=self.in_features_loc, cfg=self.cfg)

        gate_hidden = int(gate_hidden)
        gate_hidden = max(1, gate_hidden)
        gate_in_dim = 5 + (4 if self.gate_use_ssdb_hidden_stats else 0)
        self.gate_fc1 = nn.Linear(gate_in_dim, gate_hidden)
        self.gate_fc2 = nn.Linear(gate_hidden, 1)

        if bool(zero_init_gate):
            nn.init.zeros_(self.gate_fc2.weight)
            nn.init.constant_(self.gate_fc2.bias, float(gate_bias))

        if self.bias_mode == "mlp":
            bias_gate_hidden = max(1, int(bias_gate_hidden))
            self.bias_gate_fc1 = nn.Linear(gate_in_dim, bias_gate_hidden)
            self.bias_gate_fc2 = nn.Linear(bias_gate_hidden, 1)
            if bool(zero_init_gate):
                nn.init.zeros_(self.bias_gate_fc2.weight)
                nn.init.constant_(self.bias_gate_fc2.bias, float(bias_gate_bias))
        else:
            self.bias_gate_fc1 = None
            self.bias_gate_fc2 = None

        if self.node_gate == "learned":
            if self.in_features_loc < 2:
                raise ValueError(f"in_features_loc must be >=2 for node_gate=learned (got {self.in_features_loc})")
            self.node_gate_fc = nn.Linear(self.in_features_loc - 1, 1)
            nn.init.zeros_(self.node_gate_fc.weight)
            nn.init.constant_(self.node_gate_fc.bias, float(node_gate_bias))
        else:
            self.node_gate_fc = None

    def forward(self, x_fc: torch.Tensor, adj: torch.Tensor, l_loc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if l_loc is None:
            raise ValueError("DeepEZDualExpertGCN requires l_loc (local features)")
        if x_fc.dim() != 2 or l_loc.dim() != 2:
            raise ValueError(f"Expected 2D tensors: x_fc={tuple(x_fc.shape)}, l_loc={tuple(l_loc.shape)}")
        if int(x_fc.shape[0]) != int(l_loc.shape[0]):
            raise ValueError(f"Node mismatch: x_fc N={x_fc.shape[0]} vs l_loc N={l_loc.shape[0]}")

        logits_fc, bias_fc = self.fc_expert(x_fc, adj)  # logits already includes bias_fc
        logits_loc, bias_loc = self.loc_expert(l_loc, adj)  # logits already includes bias_loc

        bias_fc2 = bias_fc.view(1, -1)
        bias_loc2 = bias_loc.view(1, -1)
        logits_fc_raw = logits_fc - bias_fc2
        logits_loc_raw = logits_loc - bias_loc2

        cov_raw = l_loc[:, -1].clamp(min=0.0, max=1.0)
        cov_pow = cov_raw
        if self.coverage_power != 0.0:
            cov_pow = cov_pow.pow(self.coverage_power)
        if self.node_gate == "coverage":
            g_node = cov_pow.unsqueeze(1)  # (N,1)
        else:
            assert self.node_gate_fc is not None
            loc_feat = l_loc[:, :-1]
            g_loc = torch.sigmoid(self.node_gate_fc(loc_feat))
            if self.node_gate_floor > 0.0:
                g_loc = torch.clamp(g_loc, min=self.node_gate_floor)
            g_node = g_loc * cov_pow.unsqueeze(1)

        mean_cov = cov_raw.mean().view(1)
        gate_vec = [bias_fc.view(-1), bias_loc.view(-1), mean_cov]
        if self.gate_use_ssdb_hidden_stats:
            h_fc = F.leaky_relu(self.fc_expert.ssdb_fc1(logits_fc_raw.T), negative_slope=self.cfg.negative_slope)
            h_loc = F.leaky_relu(self.loc_expert.ssdb_fc1(logits_loc_raw.T), negative_slope=self.cfg.negative_slope)
            e_fc = (h_fc * h_fc).mean(dim=1).view(-1)
            e_loc = (h_loc * h_loc).mean(dim=1).view(-1)
            gate_vec.extend([e_fc, e_loc])
        gate_in = torch.cat(gate_vec, dim=0).view(1, -1)
        w = torch.sigmoid(self.gate_fc2(F.leaky_relu(self.gate_fc1(gate_in), negative_slope=self.cfg.negative_slope)))  # (1,1)
        if self.gate_floor > 0.0:
            w = torch.clamp(w, min=self.gate_floor)

        if self.bias_mode == "mlp":
            assert self.bias_gate_fc1 is not None and self.bias_gate_fc2 is not None
            w_bias = torch.sigmoid(
                self.bias_gate_fc2(F.leaky_relu(self.bias_gate_fc1(gate_in), negative_slope=self.cfg.negative_slope))
            )
            if self.bias_gate_floor > 0.0:
                w_bias = torch.clamp(w_bias, min=self.bias_gate_floor)
        elif self.bias_mode == "shared":
            w_bias = w
        elif self.bias_mode == "none":
            w_bias = torch.zeros_like(w)
        elif self.bias_mode == "square":
            w_bias = w * w
        else:
            w_bias = w * g_node.mean().view(1, 1)

        logits = logits_fc_raw + bias_fc2 + (w * g_node) * logits_loc_raw + (w_bias * bias_loc2)
        fused_bias = bias_fc + (w_bias.view(-1) * bias_loc)
        return logits, fused_bias


class DeepEZDualExpertGAT(nn.Module):
    """
    GAT version of DeepEZDualExpertGCN (keeps the same SSDB per-expert design).
    """

    def __init__(
        self,
        *,
        n_nodes: int,
        in_features_fc: int,
        in_features_loc: int,
        cfg: DeepEZConfig | None = None,
        gate_hidden: int = 16,
        gate_bias: float = -2.0,
        gate_floor: float = 0.0,
        gate_use_ssdb_hidden_stats: bool = False,
        coverage_power: float = 1.0,
        bias_mode: str = "shared",
        bias_gate_hidden: int = 16,
        bias_gate_bias: float = -4.0,
        bias_gate_floor: float = 0.0,
        node_gate: str = "coverage",
        node_gate_bias: float = 2.0,
        node_gate_floor: float = 0.0,
        zero_init_gate: bool = True,
    ) -> None:
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.in_features_fc = int(in_features_fc)
        self.in_features_loc = int(in_features_loc)
        self.cfg = cfg or DeepEZConfig()
        self.coverage_power = float(coverage_power)
        self.gate_floor = float(gate_floor)
        self.gate_use_ssdb_hidden_stats = bool(gate_use_ssdb_hidden_stats)

        bias_mode = str(bias_mode).strip().lower()
        if bias_mode not in {"shared", "none", "mean_gnode", "square", "mlp"}:
            raise ValueError(f"Invalid bias_mode={bias_mode} (expected shared|none|mean_gnode|square|mlp)")
        self.bias_mode = bias_mode
        self.bias_gate_floor = float(bias_gate_floor)

        node_gate = str(node_gate).strip().lower()
        if node_gate not in {"coverage", "learned"}:
            raise ValueError(f"Invalid node_gate={node_gate} (expected coverage|learned)")
        self.node_gate = node_gate
        self.node_gate_floor = float(node_gate_floor)

        self.fc_expert = DeepEZGAT(n_nodes=self.n_nodes, in_features=self.in_features_fc, cfg=self.cfg)
        self.loc_expert = DeepEZGAT(n_nodes=self.n_nodes, in_features=self.in_features_loc, cfg=self.cfg)

        gate_hidden = int(gate_hidden)
        gate_hidden = max(1, gate_hidden)
        gate_in_dim = 5 + (4 if self.gate_use_ssdb_hidden_stats else 0)
        self.gate_fc1 = nn.Linear(gate_in_dim, gate_hidden)
        self.gate_fc2 = nn.Linear(gate_hidden, 1)
        if bool(zero_init_gate):
            nn.init.zeros_(self.gate_fc2.weight)
            nn.init.constant_(self.gate_fc2.bias, float(gate_bias))

        if self.bias_mode == "mlp":
            bias_gate_hidden = max(1, int(bias_gate_hidden))
            self.bias_gate_fc1 = nn.Linear(gate_in_dim, bias_gate_hidden)
            self.bias_gate_fc2 = nn.Linear(bias_gate_hidden, 1)
            if bool(zero_init_gate):
                nn.init.zeros_(self.bias_gate_fc2.weight)
                nn.init.constant_(self.bias_gate_fc2.bias, float(bias_gate_bias))
        else:
            self.bias_gate_fc1 = None
            self.bias_gate_fc2 = None

        if self.node_gate == "learned":
            if self.in_features_loc < 2:
                raise ValueError(f"in_features_loc must be >=2 for node_gate=learned (got {self.in_features_loc})")
            self.node_gate_fc = nn.Linear(self.in_features_loc - 1, 1)
            nn.init.zeros_(self.node_gate_fc.weight)
            nn.init.constant_(self.node_gate_fc.bias, float(node_gate_bias))
        else:
            self.node_gate_fc = None

    def forward(self, x_fc: torch.Tensor, adj: torch.Tensor, l_loc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if l_loc is None:
            raise ValueError("DeepEZDualExpertGAT requires l_loc (local features)")
        if x_fc.dim() != 2 or l_loc.dim() != 2:
            raise ValueError(f"Expected 2D tensors: x_fc={tuple(x_fc.shape)}, l_loc={tuple(l_loc.shape)}")
        if int(x_fc.shape[0]) != int(l_loc.shape[0]):
            raise ValueError(f"Node mismatch: x_fc N={x_fc.shape[0]} vs l_loc N={l_loc.shape[0]}")

        logits_fc, bias_fc = self.fc_expert(x_fc, adj)
        logits_loc, bias_loc = self.loc_expert(l_loc, adj)

        bias_fc2 = bias_fc.view(1, -1)
        bias_loc2 = bias_loc.view(1, -1)
        logits_fc_raw = logits_fc - bias_fc2
        logits_loc_raw = logits_loc - bias_loc2

        cov_raw = l_loc[:, -1].clamp(min=0.0, max=1.0)
        cov_pow = cov_raw
        if self.coverage_power != 0.0:
            cov_pow = cov_pow.pow(self.coverage_power)
        if self.node_gate == "coverage":
            g_node = cov_pow.unsqueeze(1)
        else:
            assert self.node_gate_fc is not None
            loc_feat = l_loc[:, :-1]
            g_loc = torch.sigmoid(self.node_gate_fc(loc_feat))
            if self.node_gate_floor > 0.0:
                g_loc = torch.clamp(g_loc, min=self.node_gate_floor)
            g_node = g_loc * cov_pow.unsqueeze(1)

        mean_cov = cov_raw.mean().view(1)
        gate_vec = [bias_fc.view(-1), bias_loc.view(-1), mean_cov]
        if self.gate_use_ssdb_hidden_stats:
            h_fc = F.leaky_relu(self.fc_expert.ssdb_fc1(logits_fc_raw.T), negative_slope=self.cfg.negative_slope)
            h_loc = F.leaky_relu(self.loc_expert.ssdb_fc1(logits_loc_raw.T), negative_slope=self.cfg.negative_slope)
            e_fc = (h_fc * h_fc).mean(dim=1).view(-1)
            e_loc = (h_loc * h_loc).mean(dim=1).view(-1)
            gate_vec.extend([e_fc, e_loc])
        gate_in = torch.cat(gate_vec, dim=0).view(1, -1)
        w = torch.sigmoid(self.gate_fc2(F.leaky_relu(self.gate_fc1(gate_in), negative_slope=self.cfg.negative_slope)))
        if self.gate_floor > 0.0:
            w = torch.clamp(w, min=self.gate_floor)

        if self.bias_mode == "mlp":
            assert self.bias_gate_fc1 is not None and self.bias_gate_fc2 is not None
            w_bias = torch.sigmoid(
                self.bias_gate_fc2(F.leaky_relu(self.bias_gate_fc1(gate_in), negative_slope=self.cfg.negative_slope))
            )
            if self.bias_gate_floor > 0.0:
                w_bias = torch.clamp(w_bias, min=self.bias_gate_floor)
        elif self.bias_mode == "shared":
            w_bias = w
        elif self.bias_mode == "none":
            w_bias = torch.zeros_like(w)
        elif self.bias_mode == "square":
            w_bias = w * w
        else:
            w_bias = w * g_node.mean().view(1, 1)

        logits = logits_fc_raw + bias_fc2 + (w * g_node) * logits_loc_raw + (w_bias * bias_loc2)
        fused_bias = bias_fc + (w_bias.view(-1) * bias_loc)
        return logits, fused_bias
