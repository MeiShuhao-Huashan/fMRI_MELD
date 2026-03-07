import torch

from meld_fmri.fmri_gcn.model import DeepEZConfig, DeepEZDualExpertGCN, DeepEZGCN


def test_deepez_gcn_forward_shapes() -> None:
    n_nodes = 10
    cfg = DeepEZConfig(hidden1=12, hidden2=5, ssdb_hidden=7, gat_heads1=2, gat_dropout=0.0)
    model = DeepEZGCN(n_nodes=n_nodes, in_features=n_nodes, cfg=cfg)

    x = torch.randn(n_nodes, n_nodes)
    adj = torch.eye(n_nodes)
    logits, bias = model(x, adj)

    assert logits.shape == (n_nodes, 2)
    assert bias.shape == (2,)


def test_deepez_dual_expert_forward_shapes() -> None:
    n_nodes = 10
    cfg = DeepEZConfig(hidden1=12, hidden2=5, ssdb_hidden=7, gat_heads1=2, gat_dropout=0.0)
    model = DeepEZDualExpertGCN(n_nodes=n_nodes, in_features_fc=n_nodes, in_features_loc=3, cfg=cfg)

    x_fc = torch.randn(n_nodes, n_nodes)
    adj = torch.eye(n_nodes)

    l_loc = torch.randn(n_nodes, 3)
    l_loc[:, -1] = 1.0  # coverage in [0,1]

    logits, bias = model(x_fc, adj, l_loc)
    assert logits.shape == (n_nodes, 2)
    assert bias.shape == (2,)

