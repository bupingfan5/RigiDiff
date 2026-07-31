import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src import utils
from src.pocket_geometry import (
    build_fragment_template_tensors,
    contact_distance_loss,
    pocket_clash_terms,
    late_clash_guidance_weight,
    reconstruct_fragment_atoms,
    retain_global_clash_gradient,
    update_guided_center_offset,
)


class ValueNetworkGuidance:
    'Gradient guidance wrapper for diffusion sampling.'
    def __init__(self, value_net, guidance_scale=0.4, start_step=50, end_step=700, max_norm=1.0):
        'Initialize guidance parameters.'
        self.value_net = value_net
        self.guidance_scale = guidance_scale
        self.start_step = start_step
        self.end_step = end_step
        self.max_norm = max_norm
        self.last_delta_norm = 0.0

    def __call__(self, x, rot, h, t, pred_eps_x, node_mask, edge_index):
        'Apply value-function gradient guidance to predicted noise.'
        
        t_val = t[0].item() if isinstance(t, torch.Tensor) else t
        if t_val < self.start_step or t_val > self.end_step:
            self.last_delta_norm = 0.0
            return pred_eps_x

        
        x_guided = x.detach().clone().requires_grad_(True)

        
        value = self.value_net(x_guided, rot, h, t, node_mask, edge_index)

        
        grad_value = torch.autograd.grad(
            outputs=value.sum(),
            inputs=x_guided,
            create_graph=False,
            retain_graph=False,
            allow_unused=True
        )[0]

        
        if grad_value is None:
            grad_value = torch.zeros_like(x_guided)
        grad_value = torch.nan_to_num(grad_value, nan=0.0, posinf=0.0, neginf=0.0)

        if node_mask is None:
            mask = torch.ones_like(grad_value[..., :1])
        else:
            mask = node_mask
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            mask = mask.to(device=grad_value.device, dtype=grad_value.dtype)
        grad_value = grad_value * mask

        grad_norm = torch.linalg.norm(grad_value, dim=-1, keepdim=True)
        clip_scale = torch.clamp(self.max_norm / (grad_norm + 1e-8), max=1.0)
        grad_value = grad_value * clip_scale

        guided_eps_x = pred_eps_x - self.guidance_scale * grad_value
        guided_eps_x = torch.nan_to_num(guided_eps_x, nan=0.0, posinf=0.0, neginf=0.0)
        guided_eps_x = pred_eps_x * (1.0 - mask) + guided_eps_x * mask
        self.last_delta_norm = float(torch.linalg.norm((guided_eps_x - pred_eps_x) * mask).detach().cpu())

        return guided_eps_x


# =====================================================================
# 1. Core math: tangent-space diffusion on SO(3)
# =====================================================================

def isotropic_gaussian_so3(rot, t, sigmas):
    """
    SO(3) forward diffusion (noise injection) in tangent space:
      sample eps ~ N(0, I) in so(3)
      omega = sigma_t * eps
      R_t = Exp(omega) @ R_0   (LEFT-multiply)

    Args:
        rot:   [B, N, 3, 3]
        t:     [B] (long)
        sigmas:[T]  (buffer)
    Returns:
        z_rot:  [B, N, 3, 3]
        eps_so3:[B, N, 3]  (standard normal, training target)
    """
    B, N, _, _ = rot.shape
    device = rot.device

    if t.device != device:
        t = t.to(device)
    if sigmas.device != device:
        sigmas = sigmas.to(device)

    sigma_t = sigmas[t].view(B, 1, 1)  # [B,1,1]
    eps_so3 = torch.randn(B, N, 3, device=device)  # [B,N,3]
    omega = sigma_t * eps_so3  # [B,N,3]

    rot_noise = utils.axis_angle_to_matrix(omega)  # [B,N,3,3]
    z_rot = torch.matmul(rot_noise, rot)  # LEFT-multiply

    return z_rot, eps_so3


def _haar_random_rotations(B, N, device):
    """
    Sample approximate Haar rotations via QR and det fix.
    Returns [B, N, 3, 3] with det=+1.
    """
    A = torch.randn(B, N, 3, 3, device=device)
    Q, R = torch.linalg.qr(A)  # Q orthonormal but det may be -1

    detQ = torch.det(Q)  # [B,N]
    sign = detQ.sign().unsqueeze(-1)  # [B,N,1]
    # Flip one column when det is negative to enforce det=+1
    Q[..., :, 0] = Q[..., :, 0] * sign
    return Q


def _enforce_bond_symmetry(z_bond):
    """
    Force bond matrix to be symmetric with zero diagonal.
    Upper triangle wins: z_bond[i,j] = z_bond[j,i] = triu value.
    """
    upper = torch.triu(z_bond, diagonal=1)
    return upper + upper.transpose(1, 2)


def _sample_noisy_bond_state(bond_state, t, num_classes, bond_alphas_cumprod, device):
    """
    Forward diffusion using cumulative Q_bar_t (strict D3PM):

        q(E_t | E_0) = ᾱ_t · one_hot(E_0) + (1 − ᾱ_t) / C

    This samples from E_0 directly to E_t in one shot, which is what
    the training objective requires.  The old single-step Q_t formula
    only moved one step at a time and gave wrong marginals.

    Args:
        bond_state:          [B, N, N] clean bond types (long)
        t:                   [B] timesteps
        num_classes:         int  (5: none/single/double/triple/aromatic)
        bond_alphas_cumprod: [T]  precomputed ᾱ_t buffer
        device:              torch device

    Returns:
        noisy_bond_state: [B, N, N] symmetric noisy bond types (long)
    """
    B, N, _ = bond_state.shape
    C = num_classes

    alpha_bar_t = bond_alphas_cumprod[t].to(device)   # [B]
    abt = alpha_bar_t.view(B, 1, 1, 1)               # broadcast over [B,N,N,C]

    bond_one_hot = F.one_hot(
        bond_state.long().clamp(0, C - 1), num_classes=C
    ).float()  # [B, N, N, C]

    # q(E_t | E_0) = ᾱ_t * one_hot(E_0) + (1 − ᾱ_t) / C
    probs = abt * bond_one_hot + (1.0 - abt) / C  # [B, N, N, C]
    probs = probs.clamp(min=1e-8)

    noisy = torch.multinomial(
        probs.view(B * N * N, C), num_samples=1
    ).view(B, N, N)

    # Enforce symmetry: bond graph is undirected
    return _enforce_bond_symmetry(noisy)


def _sample_noisy_type_state(type_state, t, num_classes, type_alphas_cumprod, device):
    """
    Forward D3PM noising for fragment type ids.
    """
    B, N = type_state.shape
    C = num_classes

    alpha_bar_t = type_alphas_cumprod[t].to(device)
    abt = alpha_bar_t.view(B, 1, 1)

    clean_types = type_state.long().clamp(0, C - 1)
    type_one_hot = F.one_hot(clean_types, num_classes=C).float()
    probs = abt * type_one_hot + (1.0 - abt) / C
    probs = probs.clamp(min=1e-8)

    return torch.multinomial(
        probs.view(B * N, C),
        num_samples=1,
    ).view(B, N)


def _bond_reverse_posterior(z_bond_t, pred_bonds_logits, t, bond_alphas_cumprod, num_classes, device):
    """
    Strict D3PM reverse posterior:
        p_theta(z_{t-1} | z_t) ∝ q(z_t | z_{t-1}) · Σ_{z0} p_theta(z0|z_t) · q(z_{t-1}|z0)

    For the uniform absorbing schedule:
        Q_bar_t[z0, k] = ᾱ_t · δ(z0,k) + (1−ᾱ_t)/C

    So:
        Σ_{z0} p(z0) · Q_bar_{t-1}[z0, k] = ᾱ_{t-1}·p_k + (1−ᾱ_{t-1})/C   ... (theta_t1)
        q(z_t=m | z_{t-1}=k) = Q_t[k, m] = (1−β_t)·δ(k,m) + β_t/C          ... (q_zt_zt1)

    Final: posterior ∝ q_zt_zt1 · theta_t1, then normalize and sample.

    Args:
        z_bond_t:          [B, N, N]    current noisy bond state (long)
        pred_bonds_logits: [B, N, N, C] predicted logits from network
        t:                 [B]          current timesteps
        bond_alphas_cumprod: [T]        precomputed ᾱ_t for bond diffusion
        num_classes:       int          number of bond classes
        device:            torch device

    Returns:
        z_bond_t_minus_1: [B, N, N] denoised bond state (long)
    """
    B, N, _ = z_bond_t.shape
    C = num_classes

    # ᾱ_t and ᾱ_{t-1}
    alpha_bar_t  = bond_alphas_cumprod[t].to(device)            # [B]
    t_m1         = (t - 1).clamp(min=0)
    alpha_bar_t1 = bond_alphas_cumprod[t_m1].to(device)         # [B]

    abt  = alpha_bar_t.view(B, 1, 1, 1)   # broadcast over [B,N,N,C]
    abt1 = alpha_bar_t1.view(B, 1, 1, 1)

    # single-step beta:  β_t = 1 − ᾱ_t / ᾱ_{t-1}  (clamp for t=0)
    alpha_t = torch.where(t > 0,
                          alpha_bar_t / alpha_bar_t1.clamp(min=1e-8),
                          torch.ones_like(alpha_bar_t))
    beta_t = (1.0 - alpha_t).view(B, 1, 1, 1)

    # ---------- term 1: q(z_t=m | z_{t-1}=k) evaluated at z_t=known --------
    # z_t_onehot[..., k] = 1 iff z_t == k  →  plays role of selecting column m=z_t
    z_t_onehot = F.one_hot(z_bond_t.long(), num_classes=C).float()  # [B,N,N,C]
    # q(z_t | z_{t-1}=k) = (1−β)·δ(k,z_t) + β/C
    q_zt_zt1 = (1.0 - beta_t) * z_t_onehot + beta_t / C            # [B,N,N,C]

    # ---------- term 2: Σ_{z0} p(z0) · Q_bar_{t-1}[z0, k] -----------------
    pred_probs = torch.softmax(pred_bonds_logits, dim=-1)            # [B,N,N,C]
    theta_t1   = abt1 * pred_probs + (1.0 - abt1) / C              # [B,N,N,C]

    # ---------- unnormalized posterior, normalize, sample -------------------
    posterior = q_zt_zt1 * theta_t1                                  # [B,N,N,C]
    posterior = posterior.clamp(min=1e-8)
    posterior = posterior / posterior.sum(dim=-1, keepdim=True)

    z_bond_t_minus_1 = torch.multinomial(
        posterior.view(B * N * N, C),
        num_samples=1
    ).view(B, N, N)

    return z_bond_t_minus_1


def _type_reverse_posterior(z_type_t, pred_type_logits, t, type_alphas_cumprod, num_classes, device):
    """
    Strict D3PM reverse posterior for fragment type ids.
    """
    B, N = z_type_t.shape
    C = num_classes

    alpha_bar_t = type_alphas_cumprod[t].to(device)
    t_m1 = (t - 1).clamp(min=0)
    alpha_bar_t1 = type_alphas_cumprod[t_m1].to(device)

    abt1 = alpha_bar_t1.view(B, 1, 1)
    alpha_t = torch.where(
        t > 0,
        alpha_bar_t / alpha_bar_t1.clamp(min=1e-8),
        torch.ones_like(alpha_bar_t),
    )
    beta_t = (1.0 - alpha_t).view(B, 1, 1)

    z_t_onehot = F.one_hot(z_type_t.long().clamp(0, C - 1), num_classes=C).float()
    q_zt_zt1 = (1.0 - beta_t) * z_t_onehot + beta_t / C

    pred_probs = torch.softmax(pred_type_logits, dim=-1)
    theta_t1 = abt1 * pred_probs + (1.0 - abt1) / C

    posterior = q_zt_zt1 * theta_t1
    posterior = posterior.clamp(min=1e-8)
    posterior = posterior / posterior.sum(dim=-1, keepdim=True)

    return torch.multinomial(
        posterior.view(B * N, C),
        num_samples=1,
    ).view(B, N)


# =====================================================================
# 2. Diffusion model main class (FrameDiff-style)
# =====================================================================

class EDM(nn.Module):
    def __init__(
        self,
        dynamics,            # your Rigid-EGNN Dynamics
        in_node_nf: int,
        n_dims: int,
        timesteps: int = 1000,
        noise_schedule: str = "cosine",
        noise_precision: float = 1e-4,   # kept for compatibility
        loss_type: str = "l2",
        norm_values=(1.0, 1.0, 1.0),
        norm_biases=(None, 0.0, 0.0),
        lambda_pocket_clash: float = 1.0,
        lambda_contact: float = 0.1,
        lambda_attach: float = 1.0,
        lambda_connectivity: float = 0.2,
        lambda_gf: float = 0.5,
        bond_class_weights=(0.2, 1.0, 1.0, 1.0, 1.0),
    ):
        super().__init__()
        self.dynamics = dynamics
        self.in_node_nf = in_node_nf
        self.n_dims = n_dims
        self.T = timesteps
        self.norm_values = norm_values
        self.norm_biases = norm_biases
        self.lambda_pocket_clash = float(lambda_pocket_clash)
        self.lambda_contact = float(lambda_contact)
        self.lambda_attach = float(lambda_attach)
        self.lambda_connectivity = float(lambda_connectivity)
        self.lambda_gf = float(lambda_gf)
        self.register_buffer(
            "bond_class_weights",
            torch.as_tensor(bond_class_weights, dtype=torch.float32),
        )

        template_coords, template_mask, template_radii = build_fragment_template_tensors()
        self.register_buffer("fragment_template_coords", template_coords, persistent=False)
        self.register_buffer("fragment_template_mask", template_mask, persistent=False)
        self.register_buffer("fragment_template_radii", template_radii, persistent=False)

        if noise_schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, timesteps)
        elif noise_schedule == "cosine":
            s = 0.008
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0, 0.999)
        else:
            raise NotImplementedError(f"Schedule {noise_schedule} not implemented")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        # --- Discrete bond diffusion: independent uniform-noise schedule ---
        # Use a milder linear schedule so bonds denoise stably
        bond_betas = torch.linspace(1e-4, 0.02, timesteps)
        bond_alphas_cp = torch.cumprod(1.0 - bond_betas, dim=0)  # ᾱ_t for bonds
        self.register_buffer("bond_betas", bond_betas)
        self.register_buffer("bond_alphas_cumprod", bond_alphas_cp)

        type_betas = torch.linspace(1e-4, 0.02, timesteps)
        type_alphas_cp = torch.cumprod(1.0 - type_betas, dim=0)
        self.register_buffer("type_betas", type_betas, persistent=False)
        self.register_buffer("type_alphas_cumprod", type_alphas_cp, persistent=False)

        # --- SO(3) rotation diffusion: independent, much smaller sigma ---
        # Prevents huge rotation angles at large t that corrupt eps_rot targets
        rot_betas = torch.linspace(1e-5, 5e-3, timesteps)
        rot_sigmas = torch.sqrt(torch.cumsum(rot_betas, dim=0))
        self.register_buffer("rot_sigmas", rot_sigmas)

        # Ideal inter-fragment distances per bond type (Angstroms):
        # idx: 0=no-bond, 1=single, 2=double, 3=triple, 4=aromatic
        # (fragment-level bonds are longer than atom-level; scale up by ~1.0x)
        self.register_buffer(
            "bond_d_ideal",
            torch.tensor([0.0, 1.54, 1.34, 1.20, 1.40])
        )

    # -------------------------
    # Training forward
    # -------------------------
    def forward(self, x, rot, h, node_mask, fragment_mask, linker_mask, edge_mask,
                context=None, pocket_pos=None, pocket_rot=None, pocket_mask=None,
                pocket_element=None, pocket_aa_type=None,
                pocket_is_backbone=None,
                pocket_frames=None, true_bonds=None,
                true_attach_src=None, true_attach_dst=None,
                true_fragment_types=None, **kwargs):

        x, h = self.normalize(x, h)
        if pocket_pos is not None:
            pocket_pos = pocket_pos / self.norm_values[0]
        bs, n_nodes = x.shape[:2]
        device = x.device
        num_type_classes = h.shape[-1]

        t = torch.randint(0, self.T, (bs,), device=device).long()
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(bs, 1, 1)
        sqrt_sigma = self.sqrt_one_minus_alphas_cumprod[t].view(bs, 1, 1)

        
        eps_x = torch.randn_like(x)
        z_x = sqrt_alpha * x + sqrt_sigma * eps_x

        
        z_rot, eps_rot = isotropic_gaussian_so3(rot, t, self.rot_sigmas)

        
        z_bond = None
        if true_bonds is not None:
            num_bond_classes = 5  # 0:none, 1:single, 2:double, 3:triple, 4:aromatic
            z_bond = _sample_noisy_bond_state(
                true_bonds, t, num_bond_classes, self.bond_alphas_cumprod, device
            )

        h_t = h
        if true_fragment_types is not None:
            z_type = _sample_noisy_type_state(
                true_fragment_types,
                t,
                num_type_classes,
                self.type_alphas_cumprod,
                device,
            )
            h_t = F.one_hot(z_type, num_classes=num_type_classes).to(dtype=h.dtype)
            h_t = h_t * node_mask

        
        (
            pred_eps_x,
            pred_eps_rot,
            pred_bonds_logits,
            pred_type_logits,
            pred_attach_src_logits,
            pred_attach_dst_logits,
            pred_node_logits,
        ) = self.dynamics(
            t=t, x=z_x, rot=z_rot, h=h_t,
            node_mask=node_mask, edge_mask=edge_mask,
            context=context, pocket_pos=pocket_pos,
            pocket_rot=pocket_rot, pocket_mask=pocket_mask,
            pocket_element=pocket_element,
            pocket_aa_type=pocket_aa_type,
            pocket_is_backbone=pocket_is_backbone,
            linker_mask=linker_mask,
            bond_state=z_bond  
        )

        weight = (linker_mask * node_mask).squeeze(-1)
        loss_x = (pred_eps_x.float() - eps_x.float()).pow(2).sum(dim=-1)
        loss_x = (loss_x * weight).sum() / (weight.sum() + 1e-8)

        
        loss_rot = (pred_eps_rot.float() - eps_rot.float()).pow(2).sum(dim=-1)
        loss_rot = (loss_rot * weight).sum() / (weight.sum() + 1e-8)

        
        
        # Auxiliary x0 geometry is evaluated in FP32. Near the end of the
        # diffusion schedule, division by sqrt(alpha_bar) can overflow FP16.
        pred_x0 = (
            z_x.float() - sqrt_sigma.float() * pred_eps_x.float()
        ) / sqrt_alpha.float().clamp_min(1e-5)
        pred_x0_aux = torch.nan_to_num(
            pred_x0,
            nan=0.0,
            posinf=100.0,
            neginf=-100.0,
        ).clamp(min=-100.0, max=100.0)
        rot_scale = self.rot_sigmas[t].view(bs, 1, 1).float()
        pred_rot0 = torch.matmul(
            utils.axis_angle_to_matrix(-rot_scale * pred_eps_rot.float()),
            z_rot.float(),
        )
        pred_rot0 = torch.nan_to_num(
            pred_rot0,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        ).clamp(min=-1.0, max=1.0)

        
        dist_sq = torch.sum(
            (pred_x0_aux.unsqueeze(1) - pred_x0_aux.unsqueeze(2)) ** 2,
            dim=-1,
        )

        
        dist = torch.sqrt(dist_sq + 1e-8)
        mask_2d = weight.unsqueeze(1) * weight.unsqueeze(2)
        mask_2d.diagonal(dim1=1, dim2=2).fill_(0)  
        
        # =====================================================================
        loss_bond = torch.tensor(0.0, device=device)
        loss_gf = torch.tensor(0.0, device=device)
        loss_connectivity = torch.tensor(0.0, device=device)
        loss_type = torch.tensor(0.0, device=device)
        bond_precision = torch.tensor(0.0, device=device)
        bond_recall = torch.tensor(0.0, device=device)
        predicted_no_bond_rate = torch.tensor(0.0, device=device)
        isolated_fragment_rate = torch.tensor(0.0, device=device)

        if true_fragment_types is not None:
            valid_type_mask = (weight > 0.5) & (true_fragment_types >= 0)
            if valid_type_mask.any():
                valid_type_logits = pred_type_logits[valid_type_mask].float()
                valid_type_true = true_fragment_types[valid_type_mask].long().clamp(0, num_type_classes - 1)
                loss_type = F.cross_entropy(valid_type_logits, valid_type_true)

        if true_bonds is not None:
            
            upper_triangle = torch.triu(
                torch.ones(n_nodes, n_nodes, dtype=torch.bool, device=device),
                diagonal=1,
            ).unsqueeze(0)
            active_mask_2d = (mask_2d > 0.5) & upper_triangle
            valid_logits = pred_bonds_logits[active_mask_2d].float()
            valid_true = true_bonds[active_mask_2d]  # [M]

            if valid_logits.numel() > 0:
                loss_bond = F.cross_entropy(
                    valid_logits,
                    valid_true,
                    weight=self.bond_class_weights.to(valid_logits.dtype),
                )

            # Fragment centers follow their native connected-center distances.
            bond_probs_all = torch.softmax(pred_bonds_logits.float(), dim=-1)
            true_dist = torch.cdist(x.float(), x.float())
            true_edge_mask = (true_bonds > 0) & active_mask_2d
            if true_edge_mask.any():
                loss_gf = F.smooth_l1_loss(
                    dist[true_edge_mask],
                    true_dist[true_edge_mask],
                )

            p_edge = (1.0 - bond_probs_all[..., 0]) * mask_2d
            expected_degree = p_edge.sum(dim=-1)
            multi_fragment = weight.sum(dim=-1, keepdim=True) > 1.5
            connectivity_mask = (weight > 0.5) & multi_fragment
            if connectivity_mask.any():
                loss_connectivity = F.relu(
                    1.0 - expected_degree[connectivity_mask]
                ).square().mean()
                isolated_fragment_rate = (
                    expected_degree[connectivity_mask] < 0.5
                ).float().mean()

            if active_mask_2d.any():
                predicted_edge = p_edge[active_mask_2d] >= 0.5
                true_edge = true_bonds[active_mask_2d] > 0
                true_positive = (predicted_edge & true_edge).sum().float()
                bond_precision = true_positive / predicted_edge.sum().clamp_min(1).float()
                bond_recall = true_positive / true_edge.sum().clamp_min(1).float()
                predicted_no_bond_rate = (~predicted_edge).float().mean()

            
        safe_dist_sq = 3.6 ** 2
        no_bond_probs = torch.softmax(pred_bonds_logits.float(), dim=-1)[..., 0] if true_bonds is not None else 1.0
        repulsive_energy = torch.relu(safe_dist_sq - dist_sq) * mask_2d * no_bond_probs
        loss_rep = repulsive_energy.sum() / (mask_2d.sum() + 1e-8)
        loss_attach = torch.tensor(0.0, device=device)

        if (
            true_bonds is not None
            and true_attach_src is not None
            and true_attach_dst is not None
        ):
            attachment_mask = (
                (true_bonds > 0)
                & (true_attach_src >= 0)
                & (true_attach_dst >= 0)
                & (mask_2d > 0.5)
            )
            if attachment_mask.any():
                src_targets = (true_attach_src[attachment_mask].long() + 1).clamp(
                    min=0, max=pred_attach_src_logits.shape[-1] - 1
                )
                dst_targets = (true_attach_dst[attachment_mask].long() + 1).clamp(
                    min=0, max=pred_attach_dst_logits.shape[-1] - 1
                )
                loss_attach = 0.5 * (
                    F.cross_entropy(pred_attach_src_logits[attachment_mask].float(), src_targets)
                    + F.cross_entropy(pred_attach_dst_logits[attachment_mask].float(), dst_targets)
                )

        loss_pocket_clash = torch.tensor(0.0, device=device)
        loss_contact = torch.tensor(0.0, device=device)
        clash_count = torch.zeros(bs, device=device)
        min_pocket_distance = torch.full((bs,), float("inf"), device=device)
        fragment_types = true_fragment_types
        if fragment_types is None:
            fragment_types = torch.argmax(h, dim=-1)
        if pocket_pos is not None and pocket_mask is not None and self.fragment_template_coords.numel() > 0:
            pred_atoms, atom_valid, atom_radii = reconstruct_fragment_atoms(
                pred_x0_aux,
                pred_rot0,
                fragment_types,
                node_mask,
                self.fragment_template_coords,
                self.fragment_template_mask,
                self.fragment_template_radii,
            )
            true_atoms, _, _ = reconstruct_fragment_atoms(
                x.float(),
                rot.float(),
                fragment_types,
                node_mask,
                self.fragment_template_coords,
                self.fragment_template_mask,
                self.fragment_template_radii,
            )
            loss_pocket_clash, clash_count, min_pocket_distance = pocket_clash_terms(
                pred_atoms,
                atom_valid,
                atom_radii,
                pocket_pos,
                pocket_mask,
                pocket_elements=pocket_element,
                tolerance=0.4,
                sample_weight=sqrt_alpha.reshape(bs),
            )
            loss_contact = contact_distance_loss(
                pred_atoms,
                true_atoms,
                atom_valid,
                pocket_pos,
                pocket_mask,
                sample_weight=sqrt_alpha.reshape(bs),
            )

        
        lambda_bond = 1.0
        lambda_type = 1.0
        clash_weight = float(kwargs.get("clash_weight", 1.0))
        loss = (
            loss_x + loss_rot + 0.5 * loss_rep
            + lambda_bond * loss_bond
            + self.lambda_gf * loss_gf
            + self.lambda_connectivity * loss_connectivity
            + self.lambda_attach * loss_attach
            + lambda_type * loss_type
            + clash_weight * self.lambda_pocket_clash * loss_pocket_clash
            + self.lambda_contact * loss_contact
        )

        return loss, {"loss": loss, "loss_x": loss_x, "loss_rot": loss_rot,
                      "loss_rep": loss_rep, "loss_bond": loss_bond,
                      "loss_gf": loss_gf, "loss_type": loss_type,
                      "loss_connectivity": loss_connectivity,
                      "bond_precision": bond_precision,
                      "bond_recall": bond_recall,
                      "predicted_no_bond_rate": predicted_no_bond_rate,
                      "isolated_fragment_rate": isolated_fragment_rate,
                      "loss_attach": loss_attach,
                      "loss_pocket_clash": loss_pocket_clash,
                      "loss_contact": loss_contact,
                      "pocket_clash_rate": (clash_count > 0).float().mean(),
                      "pocket_min_distance": min_pocket_distance[torch.isfinite(min_pocket_distance)].mean()
                      if torch.isfinite(min_pocket_distance).any() else torch.tensor(0.0, device=device)}

    # -------------------------
    # Sampling
    # -------------------------
    def sample_chain(self, x, rot, h, node_mask, fragment_mask, linker_mask, edge_mask,
                     context, pocket_pos=None, pocket_rot=None, pocket_mask=None,
                     pocket_element=None, pocket_aa_type=None,
                     pocket_is_backbone=None, keep_frames=None,
                     guidance_fn=None, temperature=1.0, top_k=None, slot_mask=None,
                     clash_guidance_scale=0.0, clash_guidance_start=0,
                     clash_guidance_end=100, clash_guidance_max_norm=0.2,
                     clash_guidance_global_fraction=0.25,
                     clash_guidance_max_com_shift=1.0,
                     return_bond_probs=False):
        """
        Denoising sampling chain.

        Returns 5 values to match lightning.py unpacking:
            (chain, node_mask, final_bonds, final_attach_src, final_attach_dst)

        Args:
            temperature: sampling temperature (>1 = more diverse)
            top_k:       top-k filtering on bond logits before D3PM posterior
            slot_mask:   optional slot mask (passed through from lightning, unused here)
        """
        bs, n_nodes = x.shape[:2]
        device = x.device
        if linker_mask is None:
            linker_mask = torch.ones(bs, n_nodes, 1, device=device)
        if fragment_mask is None:
            fragment_mask = torch.zeros(bs, n_nodes, 1, device=device)
        x, h = self.normalize(x, h)
        if pocket_pos is not None:
            pocket_pos = pocket_pos / self.norm_values[0]
        num_type_classes = h.shape[-1]

        # Initial states
        z_x = torch.randn_like(x) * linker_mask + x * fragment_mask
        rand_rot = _haar_random_rotations(bs, n_nodes, device)
        mask_frag_4 = (fragment_mask > 0.5).unsqueeze(-1).expand(-1, -1, 3, 3)
        z_rot = torch.where(mask_frag_4, rot, rand_rot)

        # Bond initial state: symmetric uniform noise
        num_bond_classes = 5
        z_bond = _enforce_bond_symmetry(
            torch.randint(0, num_bond_classes, (bs, n_nodes, n_nodes), device=device)
        )

        # Fragment type initial state: uniform categorical noise for generated nodes.
        # Preserved fragments keep their provided type ids.
        z_type = torch.randint(0, num_type_classes, (bs, n_nodes), device=device)
        fixed_type = torch.argmax(h, dim=-1)
        frag_mask_2d = fragment_mask.squeeze(-1) > 0.5
        z_type = torch.where(frag_mask_2d, fixed_type, z_type)
        h_t = F.one_hot(z_type, num_classes=num_type_classes).to(dtype=h.dtype)

        if keep_frames is None:
            keep_frames = self.T
        keep_every = max(1, self.T // keep_frames)
        chain = []

        # Track final attach predictions across all steps
        final_attach_src = None
        final_attach_dst = None
        guided_com_offset = torch.zeros(bs, 1, 3, device=device, dtype=z_x.dtype)

        for i in reversed(range(0, self.T)):
            t = torch.full((bs,), i, device=device, dtype=torch.long)
            clash_position_grad = None
            clash_rotation_grad = None
            effective_clash_scale = 0.0

            with torch.no_grad():
                (
                    pred_eps_x, pred_eps_rot, pred_bonds_logits,
                    pred_type_logits, pred_attach_src_logits,
                    pred_attach_dst_logits, pred_node_logits,
                ) = self.dynamics(
                    t=t, x=z_x, rot=z_rot, h=h_t,
                    node_mask=node_mask, edge_mask=edge_mask, context=context,
                    pocket_pos=pocket_pos, pocket_rot=pocket_rot,
                    pocket_mask=pocket_mask,
                    pocket_element=pocket_element,
                    pocket_aa_type=pocket_aa_type,
                    pocket_is_backbone=pocket_is_backbone,
                    linker_mask=linker_mask,
                    bond_state=z_bond,
                )

            if guidance_fn is not None and getattr(guidance_fn, "requires_pocket_context", False):
                guidance_pocket_elements = pocket_element
                if guidance_pocket_elements is None:
                    guidance_pocket_elements = torch.zeros(
                        pocket_pos.shape[:2], dtype=torch.long, device=device
                    )
                pred_eps_x = guidance_fn(
                    z_x,
                    h_t,
                    t,
                    pred_eps_x,
                    node_mask,
                    pocket_pos,
                    guidance_pocket_elements,
                    pocket_mask,
                )
            elif guidance_fn is not None and isinstance(guidance_fn, ValueNetworkGuidance):
                edge_index = self._build_edge_index(bs, n_nodes, device)
                pred_eps_x = guidance_fn(z_x, z_rot, h_t, t, pred_eps_x, node_mask, edge_index)
            elif guidance_fn is not None:
                pred_eps_x = guidance_fn(z_x, t, pred_eps_x)

            if (
                float(clash_guidance_scale) > 0.0
                and int(clash_guidance_start) <= i <= int(clash_guidance_end)
                and pocket_pos is not None
                and pocket_mask is not None
                and self.fragment_template_coords.numel() > 0
            ):
                with torch.enable_grad():
                    guided_x = z_x.detach().clone().requires_grad_(True)
                    rot_delta = torch.zeros_like(pred_eps_rot).requires_grad_(True)
                    guided_rot = torch.matmul(
                        utils.axis_angle_to_matrix(rot_delta),
                        z_rot.detach(),
                    )
                    ligand_atoms, ligand_atom_mask, ligand_radii = reconstruct_fragment_atoms(
                        guided_x,
                        guided_rot,
                        z_type.detach(),
                        node_mask,
                        self.fragment_template_coords,
                        self.fragment_template_mask,
                        self.fragment_template_radii,
                    )
                    clash_energy, _, _ = pocket_clash_terms(
                        ligand_atoms,
                        ligand_atom_mask,
                        ligand_radii,
                        pocket_pos,
                        pocket_mask,
                        pocket_elements=pocket_element,
                        tolerance=0.4,
                    )
                    # pocket_clash_terms averages over the batch for training.
                    # Undo that reduction here so inference guidance is invariant
                    # to the generation batch size.
                    guidance_clash_energy = clash_energy * guided_x.shape[0]
                    clash_grad, clash_rot_grad = torch.autograd.grad(
                        guidance_clash_energy,
                        (guided_x, rot_delta),
                        allow_unused=True,
                    )
                if clash_grad is not None:
                    clash_grad = torch.nan_to_num(clash_grad) * linker_mask * node_mask
                    clash_grad = retain_global_clash_gradient(
                        clash_grad,
                        linker_mask * node_mask,
                        global_fraction=clash_guidance_global_fraction,
                    )
                    grad_norm = clash_grad.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                    max_norm = max(float(clash_guidance_max_norm), 0.0)
                    if max_norm > 0.0:
                        clash_grad = clash_grad * torch.clamp(max_norm / grad_norm, max=1.0)
                    clash_position_grad = clash_grad.detach()
                if clash_rot_grad is not None:
                    clash_rot_grad = torch.nan_to_num(clash_rot_grad) * linker_mask * node_mask
                    rot_grad_norm = clash_rot_grad.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                    max_norm = max(float(clash_guidance_max_norm), 0.0)
                    if max_norm > 0.0:
                        clash_rot_grad = clash_rot_grad * torch.clamp(
                            max_norm / rot_grad_norm,
                            max=1.0,
                        )
                    clash_rotation_grad = clash_rot_grad.detach()

                cosine_weight = late_clash_guidance_weight(
                    i,
                    start=clash_guidance_start,
                    end=clash_guidance_end,
                )
                effective_clash_scale = float(clash_guidance_scale) * cosine_weight

            # Keep latest attach predictions (used after loop)
            final_attach_src = pred_attach_src_logits
            final_attach_dst = pred_attach_dst_logits

            # Position update (DDPM reverse step)
            beta    = self.betas[i]
            c_mean  = 1.0 / torch.sqrt(self.alphas[i])
            c_noise = beta / torch.sqrt(1.0 - self.alphas_cumprod[i] + 1e-12)
            noise_z = torch.randn_like(z_x) * temperature if i > 0 else 0
            z_x = c_mean * (z_x - c_noise * pred_eps_x) + torch.sqrt(beta) * noise_z
            z_x = x * fragment_mask + z_x * linker_mask  # Inpaint fragments

            gen_mask = (linker_mask * node_mask).to(dtype=z_x.dtype)
            has_fixed_fragment = fragment_mask.sum(dim=(1, 2), keepdim=True) > 0.5
            if clash_position_grad is not None and effective_clash_scale > 0.0:
                gen_count = gen_mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
                global_grad = (clash_position_grad * gen_mask).sum(dim=1, keepdim=True) / gen_count
                relative_grad = (clash_position_grad - global_grad) * gen_mask
                fixed_fragment_update = z_x - effective_clash_scale * clash_position_grad
                de_novo_update = z_x - effective_clash_scale * relative_grad
                z_x = torch.where(has_fixed_fragment, fixed_fragment_update, de_novo_update)

                coordinate_scale = max(float(self.norm_values[0]), 1e-8)
                guided_com_offset = update_guided_center_offset(
                    guided_com_offset,
                    global_grad,
                    step_size=effective_clash_scale,
                    max_shift=float(clash_guidance_max_com_shift) / coordinate_scale,
                )

            # Random/model global drift is removed; only the explicitly tracked
            # clash-descent translation is restored for de novo generation.
            gen_count = gen_mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
            gen_mean = (z_x * gen_mask).sum(dim=1, keepdim=True) / gen_count
            z_x_centered = (z_x - gen_mean) * gen_mask
            z_x_guided = z_x_centered + guided_com_offset * gen_mask
            z_x = torch.where(has_fixed_fragment, z_x, z_x_guided)

            # Rotation update (geodesic step on SO3)
            update_mat = utils.axis_angle_to_matrix(-c_noise * pred_eps_rot)
            z_rot = torch.matmul(update_mat, z_rot)
            z_rot = torch.where(mask_frag_4, rot, z_rot)  # Inpaint
            if clash_rotation_grad is not None and effective_clash_scale > 0.0:
                clash_rotation_update = utils.axis_angle_to_matrix(
                    -effective_clash_scale * clash_rotation_grad
                )
                z_rot = torch.matmul(clash_rotation_update, z_rot)
                z_rot = torch.where(mask_frag_4, rot, z_rot)

            # Bond update: strict D3PM reverse posterior with temperature / top-k
            if i > 0:
                scaled_logits = pred_bonds_logits / max(temperature, 1e-3)
                if top_k is not None and top_k < num_bond_classes:
                    topk_vals, topk_idx = torch.topk(scaled_logits, k=top_k, dim=-1)
                    mask_inf = torch.full_like(scaled_logits, float('-inf'))
                    mask_inf.scatter_(-1, topk_idx, topk_vals)
                    scaled_logits = mask_inf
                z_bond = _bond_reverse_posterior(
                    z_bond, scaled_logits, t, self.bond_alphas_cumprod,
                    num_bond_classes, device,
                )
            else:
                z_bond = torch.argmax(pred_bonds_logits, dim=-1)
            # Enforce undirected bond graph at every step
            z_bond = _enforce_bond_symmetry(z_bond)

            # Fragment type update: strict D3PM reverse posterior.
            scaled_type_logits = pred_type_logits / max(temperature, 1e-3)
            if i > 0:
                z_type = _type_reverse_posterior(
                    z_type,
                    scaled_type_logits,
                    t,
                    self.type_alphas_cumprod,
                    num_type_classes,
                    device,
                )
            else:
                z_type = torch.argmax(pred_type_logits, dim=-1)
            z_type = torch.where(frag_mask_2d, fixed_type, z_type)
            h_t = F.one_hot(z_type, num_classes=num_type_classes).to(dtype=h.dtype)

            if (i % keep_every == 0) or (i == 0):
                x_vis, _ = self.unnormalize(z_x, h)
                rot_flat  = z_rot.reshape(bs, n_nodes, 9)
                type_logits = pred_type_logits
                if frag_mask_2d.any():
                    type_logits = torch.where(frag_mask_2d.unsqueeze(-1), h, pred_type_logits)
                chain.append(torch.cat([x_vis, rot_flat, type_logits], dim=-1))

        # Convert final attach logits to argmax indices
        if final_attach_src is not None:
            final_attach_src = torch.argmax(final_attach_src, dim=-1)  # [B, N, N]
            final_attach_dst = torch.argmax(final_attach_dst, dim=-1)  # [B, N, N]

        result = (
            torch.stack(chain),
            node_mask,
            z_bond,
            final_attach_src,
            final_attach_dst,
        )
        if return_bond_probs:
            final_bond_probs = torch.softmax(pred_bonds_logits, dim=-1)
            return result + (final_bond_probs,)
        return result

    def _build_edge_index(self, bs, n_nodes, device):
        'Build fully connected edge indices for the value network.'
        rows, cols = [], []
        for b in range(bs):
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i != j:
                        rows.append(i + b * n_nodes)
                        cols.append(j + b * n_nodes)
        return torch.tensor([rows, cols], dtype=torch.long, device=device)

    # -------------------------
    # Normalization
    # -------------------------
    def normalize(self, x, h):
        new_x = x / self.norm_values[0]
        if self.norm_biases[1] is not None:
            new_h = (h.float() - self.norm_biases[1]) / self.norm_values[1]
        else:
            new_h = h
        return new_x, new_h

    def unnormalize(self, x, h):
        new_x = x * self.norm_values[0]
        if self.norm_biases[1] is not None:
            new_h = h * self.norm_values[1] + self.norm_biases[1]
        else:
            new_h = h
        return new_x, new_h


class InpaintingEDM(EDM):
    pass
