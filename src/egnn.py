import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src import const
from src import utils
from pdb import set_trace


# -------------------------
# SO(3) helpers
# -------------------------
def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Axis-angle (w) -> rotation matrix Exp(w) in SO(3).

    axis_angle: [..., 3]
    returns:    [..., 3, 3]
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)  # [..., 1]
    half_angles = angles * 0.5
    eps = 1e-6
    small = angles.abs() < eps

    sin_half_over_angle = torch.empty_like(angles)
    sin_half_over_angle[~small] = torch.sin(half_angles[~small]) / angles[~small]
    # Taylor for small angles: sin(x/2)/x ≈ 1/2 - x^2/48
    sin_half_over_angle[small] = 0.5 - (angles[small] * angles[small]) / 48.0

    quat = torch.cat([torch.cos(half_angles), axis_angle * sin_half_over_angle], dim=-1)  # [..., 4]
    r, i, j, k = torch.unbind(quat, -1)
    two_s = 2.0 / (quat * quat).sum(-1).clamp_min(1e-12)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quat.shape[:-1] + (3, 3))


def get_local_edge_features(x: torch.Tensor, rot: torch.Tensor, edge_index):
    """
    Rigid relative features:
      - local coord of displacement: R_i^T (x_j - x_i)
      - squared distance in local frame

    x:   [N, 3]
    rot: [N, 3, 3]
    edge_index: [2, E] as (row, col)
    return: [E, 4] => (local_dx, local_dy, local_dz, dist_sq)
    """
    row, col = edge_index
    diff_global = x[col] - x[row]  # [E, 3]
    local_coord = torch.matmul(rot[row].transpose(-1, -2), diff_global.unsqueeze(-1)).squeeze(-1)  # [E, 3]
    dist_sq = torch.sum(local_coord ** 2, dim=-1, keepdim=True)  # [E, 1]
    return torch.cat([local_coord, dist_sq], dim=-1)

def coord2diff(x, edge_index, eps: float = 1e-8):
    """
    Legacy helper for older SizeClassifier code.
    x: [N,3]
    edge_index: [2,E] (row, col)
    return:
      diff: [E,3]  (x[row]-x[col])
      dist: [E,1]  (||diff||)
    """
    row, col = edge_index
    diff = x[row] - x[col]
    dist = torch.sqrt((diff * diff).sum(dim=-1, keepdim=True) + eps)
    return diff, dist

# -------------------------
# Graph blocks
# -------------------------
class GCL(nn.Module):
    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_nf,
        normalization_factor,
        aggregation_method,
        activation,
        edges_in_d=0,
        nodes_att_dim=0,
        attention=False,
        normalization=None,
    ):
        super().__init__()
        input_edge = input_nf * 2
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.attention = attention

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edges_in_d, hidden_nf),
            activation,
            nn.Linear(hidden_nf, hidden_nf),
            activation,
        )

        if normalization is None:
            self.node_mlp = nn.Sequential(
                nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
                activation,
                nn.Linear(hidden_nf, output_nf),
            )
        elif normalization == "batch_norm":
            self.node_mlp = nn.Sequential(
                nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
                nn.BatchNorm1d(hidden_nf),
                activation,
                nn.Linear(hidden_nf, output_nf),
                nn.BatchNorm1d(output_nf),
            )
        else:
            raise NotImplementedError

        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    def edge_model(self, source, target, edge_attr, edge_mask):
        if edge_attr is None:
            out = torch.cat([source, target], dim=1)
        else:
            out = torch.cat([source, target, edge_attr], dim=1)

        mij = self.edge_mlp(out)
        if self.attention:
            out = mij * self.att_mlp(mij)
        else:
            out = mij

        if edge_mask is not None:
            out = out * edge_mask
        return out, mij

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(
            edge_attr,
            row,
            num_segments=x.size(0),
            normalization_factor=self.normalization_factor,
            aggregation_method=self.aggregation_method,
        )
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = x + self.node_mlp(agg)
        return out, agg

    def forward(self, h, edge_index, edge_attr=None, node_attr=None, node_mask=None, edge_mask=None):
        row, col = edge_index
        edge_feat, _ = self.edge_model(h[row], h[col], edge_attr, edge_mask)
        h, _ = self.node_model(h, edge_index, edge_feat, node_attr)
        if node_mask is not None:
            h = h * node_mask
        return h


class RigidUpdate(nn.Module):
    """
    Rigid update head that also exposes predicted diffusion noise/score.

    We predict:
      - local translation update t_local (3)
      - local rotation axis-angle w_local (3)

    Then map to global:
      - eps_x = R @ t_local
      - eps_rot = R @ w_local

    Apply:
      - x_new   = x + eps_x
      - R_new   = Exp(eps_rot) @ R
    """

    def __init__(self, hidden_nf, activation=nn.SiLU()):
        super().__init__()
        self.rigid_mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            activation,
            nn.Linear(hidden_nf, 6),
        )
        nn.init.xavier_uniform_(self.rigid_mlp[-1].weight, gain=1e-3)
        nn.init.zeros_(self.rigid_mlp[-1].bias)

    def forward(self, h, x, rot, node_mask=None, linker_mask=None):
        updates = self.rigid_mlp(h)  # [N, 6]

        if node_mask is not None:
            updates = updates * node_mask

        t_local = updates[:, :3]
        w_local = updates[:, 3:]

        if linker_mask is not None:
            t_local = t_local * linker_mask
            w_local = w_local * linker_mask

        eps_x = torch.matmul(rot, t_local.unsqueeze(-1)).squeeze(-1)      # [N, 3]
        eps_rot = torch.matmul(rot, w_local.unsqueeze(-1)).squeeze(-1)    # [N, 3]

        x_new = x + eps_x
        rot_update = axis_angle_to_matrix(eps_rot)
        rot_new = torch.matmul(rot_update, rot)

        return x_new, rot_new, eps_x, eps_rot


class SinusoidsEmbeddingNew(nn.Module):
    def __init__(self, max_res=15.0, min_res=15.0 / 2000.0, div_factor=4):
        super().__init__()
        self.n_frequencies = int(math.log(max_res / min_res, div_factor)) + 1
        self.frequencies = 2 * math.pi * div_factor ** torch.arange(self.n_frequencies) / max_res
        self.dim = len(self.frequencies) * 2

    def forward(self, x):
        x = torch.sqrt(x + 1e-8)
        emb = x * self.frequencies[None, :].to(x.device)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.detach()


class EquivariantBlock(nn.Module):
    def __init__(
        self,
        hidden_nf,
        edge_feat_nf=0,
        device="cpu",
        activation=nn.SiLU(),
        n_layers=2,
        attention=True,
        sin_embedding=None,
        normalization_factor=100,
        aggregation_method="sum",
    ):
        super().__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.sin_embedding = sin_embedding

        rigid_feat_nf = 4  # local(3) + dist(1)

        for i in range(n_layers):
            self.add_module(
                f"gcl_{i}",
                GCL(
                    self.hidden_nf,
                    self.hidden_nf,
                    self.hidden_nf,
                    normalization_factor=normalization_factor,
                    aggregation_method=aggregation_method,
                    activation=activation,
                    attention=attention,
                    edges_in_d=edge_feat_nf + rigid_feat_nf,
                ),
            )

        self.add_module("gcl_equiv", RigidUpdate(hidden_nf, activation=activation))
        self.to(self.device if torch.cuda.is_available() else "cpu")

    def forward(self, h, x, rot, edge_index, node_mask=None, linker_mask=None, edge_mask=None, edge_attr=None):
        rigid_edge_feat = get_local_edge_features(x, rot, edge_index)  # [E,4]

        if self.sin_embedding is not None:
            dist = torch.sqrt(rigid_edge_feat[:, -1:] + 1e-8)
            dist_emb = self.sin_embedding(dist)
            final_edge_feat = torch.cat([rigid_edge_feat, dist_emb], dim=1)
            if edge_attr is not None:
                final_edge_feat = torch.cat([final_edge_feat, edge_attr], dim=1)
        else:
            final_edge_feat = torch.cat([rigid_edge_feat, edge_attr], dim=1) if edge_attr is not None else rigid_edge_feat

        for i in range(self.n_layers):
            h = self._modules[f"gcl_{i}"](h, edge_index, edge_attr=final_edge_feat, node_mask=node_mask, edge_mask=edge_mask)

        x_new, rot_new, eps_x, eps_rot = self._modules["gcl_equiv"](h, x, rot, node_mask=node_mask, linker_mask=linker_mask)

        if node_mask is not None:
            h = h * node_mask
        return h, x_new, rot_new, eps_x, eps_rot


class EGNN(nn.Module):
    """
    Rigid EGNN backbone that outputs:
      - updated h
      - updated x
      - updated rot
      - predicted eps_x, eps_rot
    """

    def __init__(
        self,
        in_node_nf,
        in_edge_nf,
        hidden_nf,
        device="cpu",
        activation=nn.SiLU(),
        n_layers=3,
        attention=False,
        inv_sublayers=2,
        sin_embedding=False,
        normalization_factor=100,
        aggregation_method="sum",
        norm_constant=1,
        tanh=False,
        normalization=None,
    ):
        super().__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.norm_constant = norm_constant
        self.tanh = tanh
        self.normalization = normalization

        if sin_embedding:
            self.sin_embedding = SinusoidsEmbeddingNew()
            edge_feat_nf = self.sin_embedding.dim * 2 + in_edge_nf
        else:
            self.sin_embedding = None
            edge_feat_nf = in_edge_nf

        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, in_node_nf)

        for i in range(n_layers):
            self.add_module(
                f"e_block_{i}",
                EquivariantBlock(
                    hidden_nf,
                    edge_feat_nf=edge_feat_nf,
                    device=device,
                    activation=activation,
                    n_layers=inv_sublayers,
                    attention=attention,
                    sin_embedding=self.sin_embedding,
                    normalization_factor=normalization_factor,
                    aggregation_method=aggregation_method,
                ),
            )

        self.to(self.device if torch.cuda.is_available() else "cpu")

    def forward(self, h, x, rot, edge_index, node_mask=None, linker_mask=None, edge_mask=None, edge_attr=None):
        h = self.embedding(h)

        last_eps_x = None
        last_eps_rot = None

        for i in range(self.n_layers):
            h, x, rot, eps_x, eps_rot = self._modules[f"e_block_{i}"](
                h,
                x,
                rot,
                edge_index,
                node_mask=node_mask,
                linker_mask=linker_mask,
                edge_mask=edge_mask,
                edge_attr=edge_attr,
            )
            last_eps_x, last_eps_rot = eps_x, eps_rot

        h = self.embedding_out(h)
        if node_mask is not None:
            h = h * node_mask

        return h, x, rot, last_eps_x, last_eps_rot


# -------------------------
# Seg-sum helper
# -------------------------
def unsorted_segment_sum(data, segment_ids, num_segments, normalization_factor, aggregation_method: str):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == "sum":
        result = result / normalization_factor
    if aggregation_method == "mean":
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result


# -------------------------
# Dynamics (diffusion interface)
# -------------------------
class PocketFeatureEncoder(nn.Module):
    """Encode pocket chemistry without changing the checkpoint EGNN input width."""

    def __init__(self, output_dim):
        super().__init__()
        self.element_embedding = nn.Embedding(119, 8, padding_idx=0)
        self.amino_acid_embedding = nn.Embedding(21, 8)
        self.backbone_embedding = nn.Embedding(2, 2)
        self.projection = nn.Sequential(
            nn.Linear(18, max(32, output_dim)),
            nn.SiLU(),
            nn.Linear(max(32, output_dim), output_dim),
        )
        nn.init.zeros_(self.projection[-1].weight)
        nn.init.zeros_(self.projection[-1].bias)

    def forward(self, elements, amino_acids, is_backbone):
        features = torch.cat(
            [
                self.element_embedding(elements.long().clamp(0, 118)),
                self.amino_acid_embedding(amino_acids.long().clamp(0, 20)),
                self.backbone_embedding(is_backbone.long().clamp(0, 1)),
            ],
            dim=-1,
        )
        return self.projection(features)


class Dynamics(nn.Module):
    'EGNN diffusion dynamics with sparse k-nearest-neighbor graphs.'

    def __init__(
        self,
        n_dims,
        in_node_nf,
        context_node_nf,
        hidden_nf=64,
        device="cpu",
        activation=nn.SiLU(),
        n_layers=4,
        attention=False,
        condition_time=True,
        inv_sublayers=2,
        sin_embedding=False,
        normalization_factor=100,
        aggregation_method="sum",
        model="egnn_dynamics",
        centering=False,
        graph_type="FC",
        norm_constant=1,
        tanh=False,
        normalization=None,
        
        knn_lig=32,
        knn_lp=64,
        lp_undirected=True,
    ):
        super().__init__()
        self.device = device
        self.n_dims = n_dims
        self.context_node_nf = context_node_nf
        self.condition_time = condition_time
        self.model = model
        self.centering = centering
        self.graph_type = graph_type
        self.norm_constant = norm_constant
        self.tanh = tanh
        self.normalization = normalization

        
        self.knn_lig = int(knn_lig)
        self.knn_lp = int(knn_lp)
        self.lp_undirected = bool(lp_undirected)

        self.fragment_feature_dim = in_node_nf
        self.presence_feature_dim = 2
        self.pre_context_node_nf = in_node_nf + self.presence_feature_dim
        self.pocket_feature_encoder = PocketFeatureEncoder(self.pre_context_node_nf)
        self.node_role_embedding = nn.Embedding(2, self.pre_context_node_nf)
        nn.init.zeros_(self.node_role_embedding.weight)
        attachment_counts = [
            len(points) for points in getattr(const, "FRAGMENT_ATTACHMENT_POINTS", [])
        ]
        self.max_attachment_sites = max(attachment_counts) if attachment_counts else 1
        self.num_attachment_classes = self.max_attachment_sites + 1
        self.num_bond_classes = 5

        in_node_nf = in_node_nf + self.presence_feature_dim + context_node_nf + int(condition_time)
        self.total_node_nf = in_node_nf

        if self.model == "egnn_dynamics":
            self.dynamics = EGNN(
                in_node_nf=in_node_nf,
                in_edge_nf=self.num_bond_classes + 2 * self.num_attachment_classes,
                hidden_nf=hidden_nf,
                device=device,
                activation=activation,
                n_layers=n_layers,
                attention=attention,
                inv_sublayers=inv_sublayers,
                sin_embedding=sin_embedding,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
                norm_constant=norm_constant,
                tanh=tanh,
                normalization=normalization,
            )
        else:
            raise NotImplementedError("Only Rigid EGNN dynamics is supported")

        
        self.edge_cache = {}
        self.opt_edge_cache = {}
        
        # ==============================================================
        
        self.edge_predictor = nn.Sequential(
            nn.Linear(in_node_nf * 2, hidden_nf),
            activation if not isinstance(activation, str) else nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf // 2),
            activation if not isinstance(activation, str) else nn.SiLU(),
            nn.Linear(hidden_nf // 2, 2)  
        )
        self.edge_predictor[-1] = nn.Linear(hidden_nf // 2, 5)
        self.node_predictor = nn.Sequential(
            nn.Linear(in_node_nf, hidden_nf),
            activation if not isinstance(activation, str) else nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf // 2),
            activation if not isinstance(activation, str) else nn.SiLU(),
            nn.Linear(hidden_nf // 2, self.fragment_feature_dim),
        )
        self.attach_src_predictor = nn.Sequential(
            nn.Linear(in_node_nf * 2, hidden_nf),
            activation if not isinstance(activation, str) else nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf // 2),
            activation if not isinstance(activation, str) else nn.SiLU(),
            nn.Linear(hidden_nf // 2, self.num_attachment_classes),
        )
        self.attach_dst_predictor = nn.Sequential(
            nn.Linear(in_node_nf * 2, hidden_nf),
            activation if not isinstance(activation, str) else nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf // 2),
            activation if not isinstance(activation, str) else nn.SiLU(),
            nn.Linear(hidden_nf // 2, self.num_attachment_classes),
        )
        self.presence_predictor = nn.Sequential(
            nn.Linear(in_node_nf, hidden_nf),
            activation if not isinstance(activation, str) else nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf // 2),
            activation if not isinstance(activation, str) else nn.SiLU(),
            nn.Linear(hidden_nf // 2, self.presence_feature_dim),
        )

    # -------------------------
    
    # -------------------------
    @torch.no_grad()
    def _knn_edges_ligand(self, x_lig: torch.Tensor, lig_mask: torch.Tensor, k: int):
        """
        x_lig:   [B, Nl, 3]
        lig_mask:[B, Nl, 1] (1=valid,0=pad)
        returns: rows, cols (flattened graph indices, dtype long)
        """
        B, Nl, _ = x_lig.shape
        device = x_lig.device

        if Nl <= 1:
            return (torch.zeros(0, device=device, dtype=torch.long),
                    torch.zeros(0, device=device, dtype=torch.long))

        x_lig = x_lig.detach()
        m = lig_mask.squeeze(-1).detach()  # [B,Nl]

        big = 1e6
        # pairwise distances: [B,Nl,Nl]
        dist = torch.cdist(x_lig, x_lig)  # float

        # mask padded nodes
        dist = dist + (1.0 - m[:, None, :]) * big
        dist = dist + (1.0 - m[:, :, None]) * big

        # remove self loops
        eye = torch.eye(Nl, device=device).unsqueeze(0)
        dist = dist + eye * big

        k_eff = min(max(1, int(k)), max(1, Nl - 1))
        nn_idx = torch.topk(dist, k_eff, dim=-1, largest=False).indices  # [B,Nl,k]

        rows_all, cols_all = [], []
        for b in range(B):
            valid = m[b] > 0.5
            if valid.sum() == 0:
                continue
            src_ids = torch.arange(Nl, device=device)[valid]                 # [Nv]
            dst_ids = nn_idx[b][valid]                                       # [Nv,k]
            rows = src_ids[:, None].expand_as(dst_ids).reshape(-1)           # [Nv*k]
            cols = dst_ids.reshape(-1)                                       # [Nv*k]
            rows_all.append(rows + b * Nl)
            cols_all.append(cols + b * Nl)

        if len(rows_all) == 0:
            return (torch.zeros(0, device=device, dtype=torch.long),
                    torch.zeros(0, device=device, dtype=torch.long))

        return torch.cat(rows_all).long(), torch.cat(cols_all).long()

    # -------------------------
    
    # -------------------------
    @torch.no_grad()
    def _knn_edges_lig_pocket(self, x_lig, x_poc, lig_mask, poc_mask, k: int, make_undirected: bool = True):
        """
        ligand -> pocket kNN edges, optional pocket -> ligand as reverse edges.

        x_lig: [B,Nl,3], x_poc:[B,Np,3]
        masks: [B,N,1]
        Returns flattened rows/cols in a merged graph of size (Nl+Np) per batch:
          ligand indices: [0..Nl-1]
          pocket indices: [Nl..Nl+Np-1]
        """
        B, Nl, _ = x_lig.shape
        Np = x_poc.shape[1]
        device = x_lig.device

        if Np <= 0 or Nl <= 0:
            return (torch.zeros(0, device=device, dtype=torch.long),
                    torch.zeros(0, device=device, dtype=torch.long))

        x_lig = x_lig.detach()
        x_poc = x_poc.detach()
        lm = lig_mask.squeeze(-1).detach()
        pm = poc_mask.squeeze(-1).detach()

        big = 1e6
        dist = torch.cdist(x_lig, x_poc)  # [B,Nl,Np]
        dist = dist + (1.0 - lm[:, :, None]) * big
        dist = dist + (1.0 - pm[:, None, :]) * big

        k_eff = min(max(1, int(k)), max(1, Np))
        nn_idx = torch.topk(dist, k_eff, dim=-1, largest=False).indices  # [B,Nl,k]

        rows_all, cols_all = [], []
        Ntot = Nl + Np

        for b in range(B):
            valid_src = lm[b] > 0.5
            if valid_src.sum() == 0:
                continue
            src_ids = torch.arange(Nl, device=device)[valid_src]        # [Nv]
            dst_ids = nn_idx[b][valid_src]                              # [Nv,k]

            rows = src_ids[:, None].expand_as(dst_ids).reshape(-1)      # ligand rows
            cols = (dst_ids.reshape(-1) + Nl)                           # pocket cols offset

            rows_all.append(rows + b * Ntot)
            cols_all.append(cols + b * Ntot)

            if make_undirected:
                rows_all.append(cols + b * Ntot)
                cols_all.append(rows + b * Ntot)

        if len(rows_all) == 0:
            return (torch.zeros(0, device=device, dtype=torch.long),
                    torch.zeros(0, device=device, dtype=torch.long))

        return torch.cat(rows_all).long(), torch.cat(cols_all).long()

    # -------------------------
    
    # -------------------------
    def get_optimised_edges(self, n_ligand, n_pocket, batch_size):
        cache_key = (n_ligand, n_pocket, batch_size)
        if cache_key in self.opt_edge_cache:
            return self.opt_edge_cache[cache_key]
        rows, cols = [], []
        n_total = n_ligand + n_pocket

        lig_idx = torch.arange(n_ligand, device=self.device)
        r1, c1 = torch.meshgrid(lig_idx, lig_idx, indexing="ij")
        rows.append(r1.flatten())
        cols.append(c1.flatten())

        pock_idx = torch.arange(n_ligand, n_total, device=self.device)
        r2, c2 = torch.meshgrid(lig_idx, pock_idx, indexing="ij")
        rows.append(r2.flatten())
        cols.append(c2.flatten())
        r3, c3 = torch.meshgrid(pock_idx, lig_idx, indexing="ij")
        rows.append(r3.flatten())
        cols.append(c3.flatten())

        base_rows = torch.cat(rows)
        base_cols = torch.cat(cols)

        final_rows, final_cols = [], []
        for b in range(batch_size):
            offset = b * n_total
            final_rows.append(base_rows + offset)
            final_cols.append(base_cols + offset)

        edge_index = [torch.cat(final_rows), torch.cat(final_cols)]
        self.opt_edge_cache[cache_key] = edge_index
        return edge_index

    def get_edges(self, n_nodes, batch_size):
        
        if n_nodes in self.edge_cache:
            edges_dic_b = self.edge_cache[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
        else:
            self.edge_cache[n_nodes] = {}
            edges_dic_b = self.edge_cache[n_nodes]

        rows, cols = [], []
        for b in range(batch_size):
            for i in range(n_nodes):
                for j in range(n_nodes):
                    rows.append(i + b * n_nodes)
                    cols.append(j + b * n_nodes)

        edges = [torch.LongTensor(rows).to(self.device), torch.LongTensor(cols).to(self.device)]
        edges_dic_b[batch_size] = edges
        return edges

    # -------------------------
    # forward
    # -------------------------
    def forward(
        self,
        t,
        x,
        rot,
        h,
        node_mask,
        edge_mask,
        context,
        pocket_pos=None,
        pocket_rot=None,
        pocket_mask=None,
        pocket_element=None,
        pocket_aa_type=None,
        pocket_is_backbone=None,
        linker_mask=None,
        bond_state=None,
        attach_src_state=None,
        attach_dst_state=None,
        node_state=None,
    ):
        """
        Inputs:
          x:   [B, N_lig, 3]
          rot: [B, N_lig, 3, 3]
          h:   [B, N_lig, F]
          node_mask: [B, N_lig, 1] or [B, N_lig]
          pocket_pos: [B, N_poc, 3] (optional)
        Returns:
          pred_eps_x:  [B, N_lig, 3]
          pred_eps_rot:[B, N_lig, 3]
        """
        bs, n_nodes = x.shape[:2]
        device = x.device

        # Ensure masks are [B, N, 1]
        def _fix_mask(m, B, N, default_val=1.0):
            if m is None:
                return torch.full((B, N, 1), float(default_val), device=device)
            if m.dim() == 2:
                return m.unsqueeze(-1)
            if m.dim() == 4:
                return m.squeeze(-1)
            return m

        node_mask = _fix_mask(node_mask, bs, n_nodes, default_val=1.0).float()
        movable_mask = _fix_mask(linker_mask, bs, n_nodes, default_val=1.0).float()

        if node_state is not None:
            node_state = node_state.long().clamp(min=0, max=self.presence_feature_dim - 1)
            node_presence = F.one_hot(node_state, num_classes=self.presence_feature_dim).float()
        else:
            node_presence = torch.cat([1.0 - node_mask, node_mask], dim=-1)

        h = torch.cat([h, node_presence], dim=-1)
        ligand_roles = torch.zeros(bs, n_nodes, dtype=torch.long, device=device)
        h = h + self.node_role_embedding(ligand_roles)

        # Pocket defaults
        if pocket_pos is not None:
            n_pocket = pocket_pos.size(1)
            if pocket_rot is None:
                pocket_rot = torch.eye(3, device=device).view(1, 1, 3, 3).repeat(bs, n_pocket, 1, 1)
            pocket_mask = _fix_mask(pocket_mask, bs, n_pocket, default_val=1.0).float()

            if pocket_element is None or pocket_aa_type is None or pocket_is_backbone is None:
                raise ValueError(
                    "Pocket element, amino-acid type, and backbone features are required"
                )

            # Merge graphs: ligand + chemically encoded pocket nodes.
            all_x = torch.cat([x, pocket_pos], dim=1)                       # [B, Ntot, 3]
            all_rot = torch.cat([rot, pocket_rot], dim=1)                   # [B, Ntot, 3, 3]
            n_total = n_nodes + n_pocket

            pocket_h = self.pocket_feature_encoder(
                pocket_element,
                pocket_aa_type,
                pocket_is_backbone,
            ).to(dtype=h.dtype)
            protein_roles = torch.ones(bs, n_pocket, dtype=torch.long, device=device)
            pocket_h = pocket_h + self.node_role_embedding(protein_roles)
            all_h = torch.cat([h, pocket_h], dim=1)                         # [B, Ntot, F]

            pock_movable = torch.zeros(bs, n_pocket, 1, device=device, dtype=movable_mask.dtype)
            all_movable = torch.cat([movable_mask, pock_movable], dim=1)    # [B, Ntot, 1]

            all_node_mask = torch.cat([node_mask, pocket_mask], dim=1)      # [B, Ntot, 1]

            
            r_ll, c_ll = self._knn_edges_ligand(x, node_mask, k=self.knn_lig)
            r_lp, c_lp = self._knn_edges_lig_pocket(
                x_lig=x,
                x_poc=pocket_pos,
                lig_mask=node_mask,
                poc_mask=pocket_mask,
                k=self.knn_lp,
                make_undirected=self.lp_undirected,
            )

            
            
            
            if n_pocket > 0 and r_ll.numel() > 0:
                
                
                b_id = (r_ll // n_nodes)
                local_r = r_ll % n_nodes
                local_c = c_ll % n_nodes
                r_ll = (b_id * n_total + local_r).long()
                c_ll = (b_id * n_total + local_c).long()

            edges = [torch.cat([r_ll, r_lp]).long(), torch.cat([c_ll, c_lp]).long()]

        else:
            
            all_x, all_rot, all_h = x, rot, h
            n_total = n_nodes
            all_movable = movable_mask
            all_node_mask = node_mask

            r, c = self._knn_edges_ligand(x, node_mask, k=self.knn_lig)
            edges = [r.long(), c.long()]

        edge_attr = None
        if edges[0].numel() > 0:
            edge_attr = torch.zeros(
                edges[0].numel(),
                self.num_bond_classes + 2 * self.num_attachment_classes,
                device=device,
                dtype=all_h.dtype,
            )
            edge_rows, edge_cols = edges
            batch_ids = torch.div(edge_rows, n_total, rounding_mode='floor')
            local_rows = edge_rows - batch_ids * n_total
            local_cols = edge_cols - batch_ids * n_total
            lig_edge_mask = (local_rows < n_nodes) & (local_cols < n_nodes)
            if lig_edge_mask.any():
                if bond_state is not None:
                    if bond_state.dtype != torch.long:
                        bond_state = bond_state.long()
                    bond_one_hot = F.one_hot(
                        bond_state.clamp(min=0, max=self.num_bond_classes - 1),
                        num_classes=self.num_bond_classes,
                    ).float()
                else:
                    bond_one_hot = torch.zeros(
                        bs, n_nodes, n_nodes, self.num_bond_classes, device=device, dtype=all_h.dtype
                    )

                if attach_src_state is not None:
                    if attach_src_state.dtype != torch.long:
                        attach_src_state = attach_src_state.long()
                    attach_src_one_hot = F.one_hot(
                        attach_src_state.clamp(min=0, max=self.num_attachment_classes - 1),
                        num_classes=self.num_attachment_classes,
                    ).float()
                else:
                    attach_src_one_hot = torch.zeros(
                        bs, n_nodes, n_nodes, self.num_attachment_classes, device=device, dtype=all_h.dtype
                    )

                if attach_dst_state is not None:
                    if attach_dst_state.dtype != torch.long:
                        attach_dst_state = attach_dst_state.long()
                    attach_dst_one_hot = F.one_hot(
                        attach_dst_state.clamp(min=0, max=self.num_attachment_classes - 1),
                        num_classes=self.num_attachment_classes,
                    ).float()
                else:
                    attach_dst_one_hot = torch.zeros(
                        bs, n_nodes, n_nodes, self.num_attachment_classes, device=device, dtype=all_h.dtype
                    )

                edge_attr[lig_edge_mask] = torch.cat(
                    [
                        bond_one_hot[
                            batch_ids[lig_edge_mask],
                            local_rows[lig_edge_mask],
                            local_cols[lig_edge_mask],
                        ],
                        attach_src_one_hot[
                            batch_ids[lig_edge_mask],
                            local_rows[lig_edge_mask],
                            local_cols[lig_edge_mask],
                        ],
                        attach_dst_one_hot[
                            batch_ids[lig_edge_mask],
                            local_rows[lig_edge_mask],
                            local_cols[lig_edge_mask],
                        ],
                    ],
                    dim=-1,
                ).to(edge_attr.dtype)

        # Flatten to [B*N, ...]
        all_x_f = all_x.view(bs * n_total, 3)
        all_rot_f = all_rot.view(bs * n_total, 3, 3)
        all_h_f = all_h.view(bs * n_total, -1)
        all_node_mask_f = all_node_mask.view(bs * n_total, 1)
        all_movable_f = all_movable.view(bs * n_total, 1)

        # Time conditioning
        if self.condition_time:
            if np.prod(t.size()) == 1:
                h_time = torch.empty_like(all_h_f[:, 0:1]).fill_(t.item())
            else:
                h_time = t.view(bs, 1).repeat(1, n_total).view(bs * n_total, 1)
            all_h_f = torch.cat([all_h_f, h_time], dim=1)

        # Context conditioning (auto-pad/truncate to context_node_nf)
        if self.context_node_nf > 0:
            c_dim = self.context_node_nf
            if context is None:
                if pocket_pos is not None:
                    all_ctx = torch.zeros(bs * n_total, c_dim, device=device, dtype=all_h_f.dtype)
                else:
                    all_ctx = torch.zeros(bs * n_total, c_dim, device=device, dtype=all_h_f.dtype)
                all_h_f = torch.cat([all_h_f, all_ctx], dim=1)
            else:
                cur_dim = context.numel() // (bs * n_nodes)
                context = context.view(bs, n_nodes, cur_dim)
                if cur_dim < c_dim:
                    context = F.pad(context, (0, c_dim - cur_dim), "constant", 0.0)
                elif cur_dim > c_dim:
                    context = context[:, :, :c_dim]

                if pocket_pos is not None:
                    pock_ctx = torch.zeros(bs, n_pocket, c_dim, device=device, dtype=context.dtype)
                    all_ctx = torch.cat([context, pock_ctx], dim=1).view(bs * n_total, c_dim)
                else:
                    all_ctx = context.view(bs * n_total, c_dim)

                all_h_f = torch.cat([all_h_f, all_ctx], dim=1)

        
        h_final_f, _, _, eps_x_f, eps_rot_f = self.dynamics(
            all_h_f,
            all_x_f,
            all_rot_f,
            edges,
            node_mask=all_node_mask_f,
            linker_mask=all_movable_f,
            edge_mask=None,
            edge_attr=edge_attr,
        )

        # Unflatten and slice ligand part
        eps_x = eps_x_f.view(bs, n_total, 3)[:, :n_nodes, :]
        eps_rot = eps_rot_f.view(bs, n_total, 3)[:, :n_nodes, :]

        
        h_ligand = h_final_f.view(bs, n_total, -1)[:, :n_nodes, :]

        eps_x = eps_x * node_mask
        eps_rot = eps_rot * node_mask

        # ==============================================================
        
        # ==============================================================
        
        # h_ligand: [B, N, F] -> h_row: [B, N, N, F], h_col: [B, N, N, F]
        h_row = h_ligand.unsqueeze(2).expand(bs, n_nodes, n_nodes, -1)
        h_col = h_ligand.unsqueeze(1).expand(bs, n_nodes, n_nodes, -1)
        h_pairs = torch.cat([h_row, h_col], dim=-1)  

        
        pred_bonds_logits = self.edge_predictor(h_pairs)  

        
        pred_bonds_logits = (pred_bonds_logits + pred_bonds_logits.transpose(1, 2)) / 2.0

        
        mask_2d = (node_mask @ node_mask.transpose(1, 2)).unsqueeze(-1)  # [B, N, N, 1]
        pred_bonds_logits = pred_bonds_logits * mask_2d
        pred_type_logits = self.node_predictor(h_ligand) * node_mask

        
        pred_attach_src_logits = self.attach_src_predictor(h_pairs) * mask_2d
        pred_attach_dst_logits = self.attach_dst_predictor(h_pairs) * mask_2d
        pred_node_logits = self.presence_predictor(h_ligand) * node_mask
        return (
            eps_x,
            eps_rot,
            pred_bonds_logits,
            pred_type_logits,
            pred_attach_src_logits,
            pred_attach_dst_logits,
            pred_node_logits,
        )


class DynamicsWithPockets(Dynamics):
    pass
