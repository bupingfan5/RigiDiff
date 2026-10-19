import torch
import torch.nn as nn
from src import const
from src.egnn import EGNN

class ValueNetwork(nn.Module):
    def __init__(self, in_node_nf=None, hidden_nf=128, n_layers=4):
        super().__init__()
        if in_node_nf is None:
            in_node_nf = const.GEOM_NUMBER_OF_FRAGMENT_TYPES
        # Append the diffusion timestep to node features.
        self.egnn = EGNN(
            in_node_nf=in_node_nf + 1,
            in_edge_nf=0,
            hidden_nf=hidden_nf,
            n_layers=n_layers,
            attention=True,
            normalization="batch_norm"
        )

        self.score_head = nn.Sequential(
            nn.Linear(in_node_nf + 1, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf // 2),
            nn.SiLU(),
            nn.Linear(hidden_nf // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x, rot, h, t, node_mask, edge_index):
        """Predict a scalar score for each rigid-fragment graph."""
        bs, n_nodes = x.shape[:2]

        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], device=x.device, dtype=torch.float32)

        t_embed = (t.view(-1, 1, 1).expand(-1, h.size(1), 1)) / 1000.0
        h_in = torch.cat([h, t_embed], dim=-1)

        # Flatten batch dimensions for the global edge index.
        x_flat = x.view(bs * n_nodes, 3)
        rot_flat = rot.view(bs * n_nodes, 3, 3)
        h_in_flat = h_in.view(bs * n_nodes, -1)
        mask_flat = node_mask.view(bs * n_nodes, 1)

        h_out_flat, _, _, _, _ = self.egnn(
            h=h_in_flat,
            x=x_flat,
            rot=rot_flat,
            edge_index=edge_index,
            node_mask=mask_flat
        )

        h_out = h_out_flat.view(bs, n_nodes, -1)

        # Mean-pool valid nodes for an SE(3)-invariant representation.
        mask_expanded = node_mask.expand_as(h_out)
        h_sum = (h_out * mask_expanded).sum(dim=1)
        h_mean = h_sum / (node_mask.sum(dim=1) + 1e-8)

        expected_score = self.score_head(h_mean)
        return expected_score
