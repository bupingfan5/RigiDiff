import torch
import torch.nn as nn
import torch.nn.functional as F


class PocketPoseValueNetwork(nn.Module):
    """Differentiable pocket-conditioned surrogate for Vina Score."""

    def __init__(self, ligand_feature_dim, hidden_dim=128, num_rbf=24):
        super().__init__()
        self.ligand_projection = nn.Linear(ligand_feature_dim, hidden_dim)
        self.pocket_embedding = nn.Embedding(119, hidden_dim)
        self.register_buffer("rbf_centers", torch.linspace(1.0, 12.0, num_rbf))
        self.rbf_gamma = 2.0
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim * 2 + num_rbf + 2, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
        )
        self.score_head = nn.Linear(hidden_dim, 1)
        self.negative_head = nn.Linear(hidden_dim, 1)

    def forward(self, ligand_pos, ligand_features, ligand_mask, pocket_pos, pocket_elements, pocket_mask):
        ligand_valid = ligand_mask.squeeze(-1).to(dtype=ligand_pos.dtype)
        pocket_valid = pocket_mask.squeeze(-1).to(dtype=ligand_pos.dtype)
        ligand_hidden = self.ligand_projection(ligand_features)
        pocket_hidden = self.pocket_embedding(pocket_elements.long().clamp(0, 118))
        ligand_pool = (ligand_hidden * ligand_valid.unsqueeze(-1)).sum(dim=1) / ligand_valid.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        pocket_pool = (pocket_hidden * pocket_valid.unsqueeze(-1)).sum(dim=1) / pocket_valid.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)

        distances = torch.cdist(ligand_pos, pocket_pos)
        pair_mask = ligand_valid.unsqueeze(-1) * pocket_valid.unsqueeze(1)
        rbf = torch.exp(
            -self.rbf_gamma * (distances.unsqueeze(-1) - self.rbf_centers.view(1, 1, 1, -1)).square()
        )
        rbf_pool = (rbf * pair_mask.unsqueeze(-1)).sum(dim=(1, 2)) / pair_mask.sum(
            dim=(1, 2), keepdim=False
        ).clamp_min(1.0).unsqueeze(-1)
        masked_distances = distances.masked_fill(pair_mask < 0.5, float("inf"))
        min_per_ligand = masked_distances.amin(dim=2)
        finite = torch.isfinite(min_per_ligand) & (ligand_valid > 0.5)
        safe_min = torch.where(finite, min_per_ligand, torch.zeros_like(min_per_ligand))
        mean_contact = safe_min.sum(dim=1) / finite.sum(dim=1).clamp_min(1)
        closest_contact = masked_distances.amin(dim=(1, 2))
        closest_contact = torch.where(
            torch.isfinite(closest_contact), closest_contact, torch.zeros_like(closest_contact)
        )

        representation = self.trunk(torch.cat([
            ligand_pool,
            pocket_pool,
            rbf_pool,
            mean_contact.unsqueeze(-1),
            closest_contact.unsqueeze(-1),
        ], dim=-1))
        return self.score_head(representation).squeeze(-1), self.negative_head(representation).squeeze(-1)


class PocketPoseValueGuidance:
    requires_pocket_context = True

    def __init__(self, model, scale=0.01, start_step=0, end_step=100, max_norm=0.2):
        self.model = model.eval()
        self.scale = float(scale)
        self.start_step = int(start_step)
        self.end_step = int(end_step)
        self.max_norm = float(max_norm)

    def __call__(self, x, h, t, pred_eps, node_mask, pocket_pos, pocket_elements, pocket_mask):
        step = int(t[0].item())
        if step < self.start_step or step > self.end_step:
            return pred_eps
        with torch.enable_grad():
            guided_x = x.detach().clone().requires_grad_(True)
            score, _ = self.model(
                guided_x, h.detach(), node_mask, pocket_pos, pocket_elements, pocket_mask
            )
            gradient = torch.autograd.grad(score.sum(), guided_x)[0]
        gradient = torch.nan_to_num(gradient) * node_mask
        count = node_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        gradient = (gradient - gradient.sum(dim=1, keepdim=True) / count) * node_mask
        norm = gradient.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        if self.max_norm > 0:
            gradient = gradient * torch.clamp(self.max_norm / norm, max=1.0)
        return pred_eps + self.scale * gradient.detach()


def pocket_pose_value_loss(pred_score, negative_logit, vina_score):
    target = vina_score.clamp(-15.0, 20.0)
    regression = F.smooth_l1_loss(pred_score, target)
    classification = F.binary_cross_entropy_with_logits(negative_logit, (vina_score < 0).float())
    return regression + classification, regression, classification
