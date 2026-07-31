import sys
import random
from datetime import datetime

import torch
import numpy as np


# =====================================================================

# =====================================================================

def axis_angle_to_matrix(axis_angle):
    'Convert axis-angle vectors to rotation matrices.'
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps

    sin_half_angles_over_angles = torch.empty_like(angles)

    
    sin_half_angles_over_angles[~small_angles] = (
            torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    
    sin_half_angles_over_angles[small_angles] = (
            0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )

    
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )

    
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

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
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def compute_geodesic_distance(R1, R2):
    'Compute the geodesic distance between rotations.'
    # R_diff = R1^T @ R2
    R_diff = torch.matmul(R1.transpose(-1, -2), R2)
    # trace(R) = 1 + 2cos(theta)
    tr = R_diff.diagonal(dim1=-2, dim2=-1).sum(-1)
    dist = torch.acos(((tr - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7))
    return dist
def compute_rotation_loss(true_rot, pred_rot, mask=None):
    'Compute the chordal loss between rotation matrices.'
    # Frobenious Norm Squared: || R1 - R2 ||^2
    loss = (true_rot - pred_rot) ** 2
    loss = loss.sum(dim=(-1, -2))  # [B, N]

    if mask is not None:
        mask = mask.squeeze(-1) if mask.dim() == 3 else mask
        loss = loss * mask
        return loss.sum() / (mask.sum() + 1e-8)
    return loss.mean()


def apply_rigid_transformation(x, rot, t):
    'Apply a rigid transformation to coordinates.'
    # x: [..., 3], rot: [..., 3, 3], t: [..., 3]
    x_rotated = torch.matmul(rot, x.unsqueeze(-1)).squeeze(-1)
    return x_rotated + t


def rigid_transform_Kabsch_3D(A, B):
    'Estimate the optimal rigid transform with Kabsch alignment.'
    assert A.shape == B.shape

    
    centroid_A = torch.mean(A, dim=0, keepdim=True)
    centroid_B = torch.mean(B, dim=0, keepdim=True)
    AA = A - centroid_A
    BB = B - centroid_B

    
    H = torch.matmul(AA.transpose(0, 1), BB)

    # 3. SVD
    U, S, Vt = torch.linalg.svd(H)

    
    R = torch.matmul(Vt.transpose(0, 1), U.transpose(0, 1))

    
    if torch.det(R) < 0:
        Vt[2, :] *= -1
        R = torch.matmul(Vt.transpose(0, 1), U.transpose(0, 1))

    
    t = centroid_B.transpose(0, 1) - torch.matmul(R, centroid_A.transpose(0, 1))

    return R, t.transpose(0, 1)


# =====================================================================

# =====================================================================

class Logger(object):
    def __init__(self, logpath, syspart=sys.stdout):
        self.terminal = syspart
        self.log = open(logpath, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


def log(*args):
    print(f'[{datetime.now()}]', *args)


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def sum_except_batch(x):
    return x.reshape(x.size(0), -1).sum(dim=-1)


def remove_mean(x):
    mean = torch.mean(x, dim=1, keepdim=True)
    x = x - mean
    return x


def remove_mean_with_mask(x, node_mask):
    masked_max_abs_value = (x * (1 - node_mask)).abs().sum().item()
    # assert masked_max_abs_value < 1e-5, f'Error {masked_max_abs_value} too high'
    N = node_mask.sum(1, keepdims=True)

    mean = torch.sum(x, dim=1, keepdim=True) / N
    x = x - mean * node_mask
    return x


def remove_partial_mean_with_mask(x, node_mask, center_of_mass_mask):
    'Center valid coordinates under a node mask.'
    
    if center_of_mass_mask.dim() == 2 and x.dim() == 3:
        center_of_mass_mask = center_of_mass_mask.unsqueeze(-1)

    x_masked = x * center_of_mass_mask
    N = center_of_mass_mask.sum(1, keepdims=True)
    
    mean = torch.sum(x_masked, dim=1, keepdim=True) / (N + 1e-8)
    x = x - mean * node_mask
    return x


def assert_mean_zero(x):
    mean = torch.mean(x, dim=1, keepdim=True)
    assert mean.abs().max().item() < 1e-4


def assert_mean_zero_with_mask(x, node_mask, eps=1e-10):
    assert_correctly_masked(x, node_mask)
    largest_value = x.abs().max().item()
    error = torch.sum(x, dim=1, keepdim=True).abs().max().item()
    rel_error = error / (largest_value + eps)
    if rel_error > 1e-2:
        print(f'Warning: Mean is not zero, relative_error {rel_error}')


def assert_partial_mean_zero_with_mask(x, node_mask, center_of_mass_mask, eps=1e-10):
    assert_correctly_masked(x, node_mask)
    
    if center_of_mass_mask.dim() == 2:
        center_of_mass_mask = center_of_mass_mask.unsqueeze(-1)

    x_masked = x * center_of_mass_mask
    largest_value = x_masked.abs().max().item()
    error = torch.sum(x_masked, dim=1, keepdim=True).abs().max().item()
    rel_error = error / (largest_value + eps)
    # assert rel_error < 1e-2, f'Partial mean is not zero, relative_error {rel_error}'


def assert_correctly_masked(variable, node_mask):
    if node_mask.dim() == 2: node_mask = node_mask.unsqueeze(-1)
    assert (variable * (1 - node_mask)).abs().max().item() < 1e-4, \
        'Variables not masked properly.'


def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


# =====================================================================

# =====================================================================

def random_rotation(x, rot=None):
    'Apply a random global rotation to positions and frames.'
    bs, n_nodes, n_dims = x.size()
    device = x.device
    angle_range = np.pi * 2

    
    if n_dims == 3:
        # Build Rx
        Rx = torch.eye(3).unsqueeze(0).repeat(bs, 1, 1).to(device)
        theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        Rx[:, 1:2, 1:2] = cos
        Rx[:, 1:2, 2:3] = sin
        Rx[:, 2:3, 1:2] = - sin
        Rx[:, 2:3, 2:3] = cos

        # Build Ry
        Ry = torch.eye(3).unsqueeze(0).repeat(bs, 1, 1).to(device)
        theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        Ry[:, 0:1, 0:1] = cos
        Ry[:, 0:1, 2:3] = -sin
        Ry[:, 2:3, 0:1] = sin
        Ry[:, 2:3, 2:3] = cos

        # Build Rz
        Rz = torch.eye(3).unsqueeze(0).repeat(bs, 1, 1).to(device)
        theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        Rz[:, 0:1, 0:1] = cos
        Rz[:, 0:1, 1:2] = sin
        Rz[:, 1:2, 0:1] = -sin
        Rz[:, 1:2, 1:2] = cos

        
        R_aug = torch.matmul(Rz, torch.matmul(Ry, Rx))  # [B, 3, 3]

        
        
        x = x.transpose(1, 2)  # [B, 3, N]
        x = torch.matmul(R_aug, x)
        x = x.transpose(1, 2)  # [B, N, 3]

        
        if rot is not None:
            
            
            R_aug_expanded = R_aug.unsqueeze(1).expand(-1, n_nodes, -1, -1)
            rot = torch.matmul(R_aug_expanded, rot)
            return x.contiguous(), rot.contiguous()

    else:
        raise Exception("2D rotation not supported for rigid bodies yet")

    return x.contiguous()


# =====================================================================

# =====================================================================

def center_gravity_zero_gaussian_log_likelihood(x):
    assert len(x.size()) == 3
    B, N, D = x.size()
    assert_mean_zero(x)
    r2 = sum_except_batch(x.pow(2))
    degrees_of_freedom = (N - 1) * D
    log_normalizing_constant = -0.5 * degrees_of_freedom * np.log(2 * np.pi)
    log_px = -0.5 * r2 + log_normalizing_constant
    return log_px


def sample_center_gravity_zero_gaussian(size, device):
    assert len(size) == 3
    x = torch.randn(size, device=device)
    x_projected = remove_mean(x)
    return x_projected


def center_gravity_zero_gaussian_log_likelihood_with_mask(x, node_mask):
    assert len(x.size()) == 3
    B, N_embedded, D = x.size()
    assert_mean_zero_with_mask(x, node_mask)
    r2 = sum_except_batch(x.pow(2))
    N = node_mask.squeeze(2).sum(1)  # N has shape [B]
    degrees_of_freedom = (N - 1) * D
    log_normalizing_constant = -0.5 * degrees_of_freedom * np.log(2 * np.pi)
    log_px = -0.5 * r2 + log_normalizing_constant
    return log_px


def sample_center_gravity_zero_gaussian_with_mask(size, device, node_mask):
    assert len(size) == 3
    x = torch.randn(size, device=device)
    x_masked = x * node_mask
    x_projected = remove_mean_with_mask(x_masked, node_mask)
    return x_projected


def standard_gaussian_log_likelihood(x):
    log_px = sum_except_batch(-0.5 * x * x - 0.5 * np.log(2 * np.pi))
    return log_px


def sample_gaussian(size, device):
    x = torch.randn(size, device=device)
    return x


def standard_gaussian_log_likelihood_with_mask(x, node_mask):
    log_px_elementwise = -0.5 * x * x - 0.5 * np.log(2 * np.pi)
    log_px = sum_except_batch(log_px_elementwise * node_mask)
    return log_px


def sample_gaussian_with_mask(size, device, node_mask):
    x = torch.randn(size, device=device)
    x_masked = x * node_mask
    return x_masked


def concatenate_features(x, h):
    xh = torch.cat([x, h['categorical']], dim=2)
    if 'integer' in h:
        xh = torch.cat([xh, h['integer']], dim=2)
    return xh


def split_features(z, n_dims, num_classes, include_charges):
    assert z.size(2) == n_dims + num_classes + include_charges
    x = z[:, :, 0:n_dims]
    h = {'categorical': z[:, :, n_dims:n_dims + num_classes]}
    if include_charges:
        h['integer'] = z[:, :, n_dims + num_classes:n_dims + num_classes + 1]
    return x, h


class Queue:
    def __init__(self, max_len=50):
        self.items = []
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def add(self, item):
        self.items.insert(0, item)
        if len(self) > self.max_len:
            self.items.pop()

    def mean(self):
        return np.mean(self.items)

    def std(self):
        return np.std(self.items)


def gradient_clipping(flow, gradnorm_queue):
    max_grad_norm = 1.5 * gradnorm_queue.mean() + 2 * gradnorm_queue.std()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        flow.parameters(), max_norm=max_grad_norm, norm_type=2.0)

    if float(grad_norm) > max_grad_norm:
        gradnorm_queue.add(float(max_grad_norm))
    else:
        gradnorm_queue.add(float(grad_norm))

    if float(grad_norm) > max_grad_norm:
        print(f'Clipped gradient with value {grad_norm:.1f} while allowed {max_grad_norm:.1f}')
    return grad_norm


def disable_rdkit_logging():
    import rdkit.rdBase as rkrb
    import rdkit.RDLogger as rkl
    logger = rkl.logger()
    logger.setLevel(rkl.ERROR)
    rkrb.DisableLog('rdApp.error')


def set_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FoundNaNException(Exception):
    def __init__(self, x, h):
        x_nan_idx = self.find_nan_idx(x)
        h_nan_idx = self.find_nan_idx(h)

        self.x_h_nan_idx = x_nan_idx & h_nan_idx
        self.only_x_nan_idx = x_nan_idx.difference(h_nan_idx)
        self.only_h_nan_idx = h_nan_idx.difference(x_nan_idx)

    @staticmethod
    def find_nan_idx(z):
        idx = set()
        for i in range(z.shape[0]):
            if torch.any(torch.isnan(z[i])):
                idx.add(i)
        return idx
