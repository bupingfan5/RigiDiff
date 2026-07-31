import math

import numpy as np
import torch
import torch.nn.functional as F

from src import const


_ATOMIC_NUMBERS = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Br": 35,
    "I": 53,
}

# Bondi-style radii in Angstrom. Unknown elements use carbon's radius.
_VDW_RADII = {
    1: 1.20,
    6: 1.70,
    7: 1.55,
    8: 1.52,
    9: 1.47,
    15: 1.80,
    16: 1.80,
    17: 1.75,
    35: 1.85,
    53: 1.98,
}


def _numpy_vdw_radii(elements):
    return np.asarray(
        [_VDW_RADII.get(int(element), _VDW_RADII[6]) for element in elements],
        dtype=np.float64,
    )


def numpy_vdw_overlap_metrics(
    ligand_coords,
    ligand_elements,
    pocket_coords,
    pocket_elements,
    tolerance=0.4,
):
    ligand = np.asarray(ligand_coords, dtype=np.float64)
    pocket = np.asarray(pocket_coords, dtype=np.float64)
    if ligand.size == 0 or pocket.size == 0:
        return {
            "clash_count": 0,
            "overlap_max": 0.0,
            "overlap_mean": 0.0,
            "overlap_energy": 0.0,
            "minimum_relative_distance": float("inf"),
        }
    differences = ligand[:, None, :] - pocket[None, :, :]
    distances = np.linalg.norm(differences, axis=-1).clip(min=1e-8)
    radius_sum = (
        _numpy_vdw_radii(ligand_elements)[:, None]
        + _numpy_vdw_radii(pocket_elements)[None, :]
    )
    overlap = np.maximum(radius_sum - float(tolerance) - distances, 0.0)
    positive = overlap[overlap > 0.0]
    return {
        "clash_count": int(positive.size),
        "overlap_max": float(positive.max()) if positive.size else 0.0,
        "overlap_mean": float(positive.mean()) if positive.size else 0.0,
        "overlap_energy": float(np.square(positive).sum()) if positive.size else 0.0,
        "minimum_relative_distance": float(np.min(distances / np.maximum(radius_sum, 1e-8))),
    }


def _axis_angle_rotation_matrix(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm < 1e-12 or abs(float(angle)) < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z = axis / norm
    skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def _deterministic_orientation_matrices(coords, count):
    """Return a fixed set of global/principal-axis rigid rotations."""
    count = max(1, int(count))
    centered = np.asarray(coords, dtype=np.float64) - np.asarray(coords, dtype=np.float64).mean(axis=0)
    axes = [
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([0.0, 1.0, 0.0]),
        np.asarray([0.0, 0.0, 1.0]),
    ]
    if len(centered) >= 2 and np.linalg.norm(centered) > 1e-10:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        for axis in vh:
            axis = np.asarray(axis, dtype=np.float64)
            pivot = int(np.argmax(np.abs(axis)))
            if axis[pivot] < 0.0:
                axis = -axis
            axes.append(axis)

    unique_axes = []
    seen_axes = set()
    for axis in axes:
        axis = axis / max(np.linalg.norm(axis), 1e-12)
        key = tuple(np.round(axis, 8))
        reverse_key = tuple(np.round(-axis, 8))
        if key in seen_axes or reverse_key in seen_axes:
            continue
        seen_axes.add(key)
        unique_axes.append(axis)

    matrices = [np.eye(3, dtype=np.float64)]
    angles = [
        math.pi / 6.0,
        -math.pi / 6.0,
        math.pi / 3.0,
        -math.pi / 3.0,
        math.pi / 2.0,
        -math.pi / 2.0,
        math.pi,
    ]
    for angle in angles:
        for axis in unique_axes:
            matrices.append(_axis_angle_rotation_matrix(axis, angle))
            if len(matrices) >= count:
                return matrices
    return matrices[:count]


def _fit_rigid_coords_inside_box(coords, box_center, box_size):
    if box_center is None or box_size is None:
        return coords, True
    center = np.asarray(box_center, dtype=np.float64)
    half_size = np.asarray(box_size, dtype=np.float64) / 2.0
    lower = center - half_size
    upper = center + half_size
    span = coords.max(axis=0) - coords.min(axis=0)
    if np.any(span > 2.0 * half_size):
        return coords, False
    correction = np.zeros(3, dtype=np.float64)
    current_min = coords.min(axis=0)
    current_max = coords.max(axis=0)
    correction += np.maximum(lower - current_min, 0.0)
    correction -= np.maximum(current_max + correction - upper, 0.0)
    fitted = coords + correction
    inside = bool(np.all(fitted >= lower - 1e-6) and np.all(fitted <= upper + 1e-6))
    return fitted, inside


def rigid_body_vdw_relax(
    ligand_coords,
    ligand_elements,
    pocket_coords,
    pocket_elements,
    steps=120,
    translation_step=0.15,
    rotation_step=0.05,
    max_com_shift=4.0,
    tolerance=0.4,
    box_center=None,
    box_size=None,
    multistart_orientations=12,
    multistart_refine=8,
):
    """Reduce pocket clashes with deterministic, Vina-independent rigid search."""
    original = np.asarray(ligand_coords, dtype=np.float64)
    pocket = np.asarray(pocket_coords, dtype=np.float64)
    ligand_radii = _numpy_vdw_radii(ligand_elements)
    pocket_radii = _numpy_vdw_radii(pocket_elements)
    original_center = original.mean(axis=0)
    before = numpy_vdw_overlap_metrics(
        original, ligand_elements, pocket, pocket_elements, tolerance=tolerance
    )

    def candidate_rank(metrics, coords):
        shift = float(np.linalg.norm(coords.mean(axis=0) - original_center))
        rmsd = float(np.sqrt(np.mean(np.square(coords - original).sum(axis=1))))
        return (
            int(metrics["clash_count"] > 0),
            int(metrics["clash_count"]),
            float(metrics["overlap_max"]),
            float(metrics["overlap_energy"]),
            rmsd,
            shift,
        )

    def fit_and_bound(coords):
        proposal, inside_box = _fit_rigid_coords_inside_box(coords, box_center, box_size)
        if not inside_box:
            return None
        shift = proposal.mean(axis=0) - original_center
        shift_norm = float(np.linalg.norm(shift))
        if shift_norm > float(max_com_shift) + 1e-6:
            return None
        return proposal

    translation_directions = []
    for x in (-1.0, 0.0, 1.0):
        for y in (-1.0, 0.0, 1.0):
            for z in (-1.0, 0.0, 1.0):
                direction = np.asarray([x, y, z], dtype=np.float64)
                norm = np.linalg.norm(direction)
                if norm > 0.0:
                    translation_directions.append(direction / norm)
    pocket_direction = original_center - pocket.mean(axis=0)
    if np.linalg.norm(pocket_direction) > 1e-10:
        translation_directions.insert(0, pocket_direction / np.linalg.norm(pocket_direction))

    orientations = _deterministic_orientation_matrices(original, multistart_orientations)
    radii = (
        np.linspace(0.5, float(max_com_shift), 8)
        if float(max_com_shift) > 0.0
        else np.asarray([], dtype=np.float64)
    )
    coarse_candidates = []
    seen_candidates = set()

    def add_coarse_candidate(proposal):
        proposal = fit_and_bound(proposal)
        if proposal is None:
            return
        key = tuple(np.round(proposal.reshape(-1), 5))
        if key in seen_candidates:
            return
        seen_candidates.add(key)
        metrics = numpy_vdw_overlap_metrics(
            proposal,
            ligand_elements,
            pocket,
            pocket_elements,
            tolerance=tolerance,
        )
        coarse_candidates.append((candidate_rank(metrics, proposal), proposal.copy(), metrics))

    centered_original = original - original_center
    for rotation in orientations:
        oriented = centered_original @ rotation.T + original_center
        add_coarse_candidate(oriented)
        for radius in radii:
            for direction in translation_directions:
                add_coarse_candidate(oriented + radius * direction)

    if not coarse_candidates:
        coarse_candidates.append((candidate_rank(before, original), original.copy(), before))
    coarse_candidates.sort(key=lambda item: item[0])
    starts = coarse_candidates[:max(1, int(multistart_refine))]

    def refine_start(start_coords, start_metrics):
        current = start_coords.copy()
        current_metrics = start_metrics
        current_rank = candidate_rank(current_metrics, current)
        accepted = 0

        for _ in range(max(0, int(steps))):
            if current_metrics["clash_count"] == 0:
                break
            differences = current[:, None, :] - pocket[None, :, :]
            distances = np.linalg.norm(differences, axis=-1).clip(min=1e-8)
            radius_sum = ligand_radii[:, None] + pocket_radii[None, :]
            overlap = np.maximum(radius_sum - float(tolerance) - distances, 0.0)
            pair_directions = differences / distances[..., None]
            pair_forces = 2.0 * overlap[..., None] * pair_directions
            atom_forces = pair_forces.sum(axis=1)
            global_force = atom_forces.sum(axis=0)
            force_norm = np.linalg.norm(global_force)
            if force_norm < 1e-10:
                worst_pair = np.unravel_index(np.argmax(overlap), overlap.shape)
                global_force = pair_directions[worst_pair]
                force_norm = np.linalg.norm(global_force)

            center = current.mean(axis=0)
            centered = current - center
            torque = np.cross(centered, atom_forces).sum(axis=0)
            torque_norm = np.linalg.norm(torque)
            translation_direction = global_force / max(force_norm, 1e-12)
            rotation_axis = torque / max(torque_norm, 1e-12)
            overlap_scale = min(1.0, max(current_metrics["overlap_max"], 0.1))

            proposals = []
            for scale in (1.0, 0.5, 0.25, 0.1):
                shift = float(translation_step) * overlap_scale * scale * translation_direction
                angle = float(rotation_step) * scale if torque_norm > 1e-10 else 0.0
                moves = [(shift, angle), (shift, -angle), (shift, 0.0)]
                if angle != 0.0:
                    moves.extend([(np.zeros(3), angle), (np.zeros(3), -angle)])
                for move, candidate_angle in moves:
                    rotation = _axis_angle_rotation_matrix(rotation_axis, candidate_angle)
                    proposal = centered @ rotation.T + center + move
                    total_shift = proposal.mean(axis=0) - original_center
                    total_shift_norm = np.linalg.norm(total_shift)
                    if total_shift_norm > float(max_com_shift):
                        proposal += (
                            total_shift * (float(max_com_shift) / max(total_shift_norm, 1e-12))
                            - total_shift
                        )
                    proposal = fit_and_bound(proposal)
                    if proposal is not None:
                        proposals.append(proposal)

            improved = None
            for proposal in proposals:
                metrics = numpy_vdw_overlap_metrics(
                    proposal,
                    ligand_elements,
                    pocket,
                    pocket_elements,
                    tolerance=tolerance,
                )
                rank = candidate_rank(metrics, proposal)
                if rank < current_rank and (improved is None or rank < improved[0]):
                    improved = (rank, proposal.copy(), metrics)
            if improved is None:
                break
            current_rank, current, current_metrics = improved
            accepted += 1

        return current_rank, current, current_metrics, accepted

    refined = [refine_start(coords, metrics) for _, coords, metrics in starts]
    best_rank, best, best_metrics, accepted_steps = min(refined, key=lambda item: item[0])

    return best, {
        "before": before,
        "after": best_metrics,
        "accepted_steps": accepted_steps,
        "coarse_candidates": len(coarse_candidates),
        "orientation_count": len(orientations),
        "refined_starts": len(starts),
        "com_shift": float(np.linalg.norm(best.mean(axis=0) - original_center)),
        "rmsd": float(np.sqrt(np.mean(np.square(best - original).sum(axis=1)))),
    }


def center_ligand_and_pocket(ligand_pos, ligand_mask, pocket_pos=None, pocket_mask=None):
    """Place ligand and pocket in one translation-invariant coordinate frame."""
    mask = ligand_mask.to(device=ligand_pos.device, dtype=ligand_pos.dtype)
    mean = (ligand_pos * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
    ligand_centered = (ligand_pos - mean) * mask
    if pocket_pos is None:
        return ligand_centered, None, mean
    pocket_centered = pocket_pos - mean
    if pocket_mask is not None:
        pocket_centered = pocket_centered * pocket_mask.to(pocket_centered.dtype)
    return ligand_centered, pocket_centered, mean


def vdw_radii_from_atomic_numbers(elements, dtype=torch.float32):
    lookup = torch.full((119,), _VDW_RADII[6], dtype=dtype, device=elements.device)
    for atomic_number, radius in _VDW_RADII.items():
        lookup[atomic_number] = radius
    return lookup[elements.long().clamp(min=0, max=118)]


def retain_global_clash_gradient(gradient, movable_mask, global_fraction=0.25):
    """Remove rigid translation only partially so one-fragment poses can escape clashes."""
    mask = movable_mask.to(device=gradient.device, dtype=gradient.dtype)
    masked_gradient = gradient * mask
    count = mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
    mean_gradient = masked_gradient.sum(dim=1, keepdim=True) / count
    relative_gradient = masked_gradient - mean_gradient * mask
    fraction = max(0.0, min(float(global_fraction), 1.0))
    return (relative_gradient + fraction * mean_gradient) * mask


def bound_generated_center_shift(coords, movable_mask, max_shift=1.0):
    """Keep de novo translation bounded while allowing clash guidance to move the ligand."""
    mask = movable_mask.to(device=coords.device, dtype=coords.dtype)
    count = mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
    center = (coords * mask).sum(dim=1, keepdim=True) / count
    limit = max(float(max_shift), 0.0)
    if limit == 0.0:
        bounded_center = torch.zeros_like(center)
    else:
        center_norm = center.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        bounded_center = center * torch.clamp(limit / center_norm, max=1.0)
    return (coords - center + bounded_center) * mask, bounded_center


def update_guided_center_offset(current_offset, energy_gradient, step_size, max_shift=1.0):
    """Take a bounded gradient-descent step for the ligand's global translation."""
    updated = current_offset - float(step_size) * energy_gradient
    limit = max(float(max_shift), 0.0)
    if limit == 0.0:
        return torch.zeros_like(updated)
    norm = updated.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return updated * torch.clamp(limit / norm, max=1.0)


def late_clash_guidance_weight(step, start=0, end=100, minimum=0.1):
    """Cosine weight with a nonzero floor inside the reverse-sampling window."""
    start = int(start)
    end = int(end)
    step = int(step)
    if step < start or step > end:
        return 0.0
    if end <= start:
        return 1.0
    progress = (end - step) / float(end - start)
    progress = max(0.0, min(progress, 1.0))
    cosine_weight = 0.5 * (1.0 + math.cos(math.pi * progress))
    floor = max(0.0, min(float(minimum), 1.0))
    return max(floor, cosine_weight)


def build_fragment_template_tensors(dtype=torch.float32):
    """Build padded heavy-atom templates from the active fragment vocabulary."""
    templates = const.IDEAL_COORDS
    symbols_by_type = getattr(const, "FRAGMENT_ATOM_SYMBOLS", [])
    max_atoms = max((len(template) for template in templates), default=1)
    coords = torch.zeros(len(templates), max_atoms, 3, dtype=dtype)
    mask = torch.zeros(len(templates), max_atoms, dtype=torch.bool)
    radii = torch.zeros(len(templates), max_atoms, dtype=dtype)

    for type_idx, template in enumerate(templates):
        symbols = symbols_by_type[type_idx] if type_idx < len(symbols_by_type) else []
        for atom_idx, atom_info in enumerate(template):
            symbol = symbols[atom_idx] if atom_idx < len(symbols) else "".join(
                char for char in str(atom_info[0]).strip() if char.isalpha()
            ).capitalize()
            if symbol == "H":
                continue
            atomic_number = _ATOMIC_NUMBERS.get(symbol, 6)
            coords[type_idx, atom_idx] = torch.as_tensor(atom_info[2], dtype=dtype)
            mask[type_idx, atom_idx] = True
            radii[type_idx, atom_idx] = _VDW_RADII.get(atomic_number, _VDW_RADII[6])
    return coords, mask, radii


def reconstruct_fragment_atoms(centers, rotations, fragment_types, node_mask, template_coords, template_mask, template_radii):
    """Reconstruct padded fragment heavy atoms in global coordinates."""
    safe_types = fragment_types.long().clamp(min=0, max=template_coords.shape[0] - 1)
    local = template_coords[safe_types]
    atom_mask = template_mask[safe_types] & (node_mask.squeeze(-1) > 0.5).unsqueeze(-1)
    atom_radii = template_radii[safe_types]
    global_coords = torch.einsum("bnij,bnaj->bnai", rotations, local) + centers.unsqueeze(-2)
    batch_size = centers.shape[0]
    return (
        global_coords.reshape(batch_size, -1, 3),
        atom_mask.reshape(batch_size, -1),
        atom_radii.reshape(batch_size, -1),
    )


def pocket_clash_terms(
    ligand_atoms,
    ligand_mask,
    ligand_radii,
    pocket_pos,
    pocket_mask,
    pocket_elements=None,
    tolerance=0.4,
    sample_weight=None,
):
    """Return differentiable vdW-overlap loss and per-complex diagnostics."""
    if pocket_pos is None or pocket_pos.shape[1] == 0:
        zero = ligand_atoms.sum() * 0.0
        batch = ligand_atoms.shape[0]
        return zero, torch.zeros(batch, device=ligand_atoms.device), torch.full(
            (batch,), float("inf"), device=ligand_atoms.device
        )

    pocket_mask_2d = pocket_mask.squeeze(-1) > 0.5
    if pocket_elements is None:
        pocket_radii = torch.full(
            pocket_mask_2d.shape,
            _VDW_RADII[6],
            dtype=ligand_atoms.dtype,
            device=ligand_atoms.device,
        )
    else:
        pocket_radii = vdw_radii_from_atomic_numbers(pocket_elements, dtype=ligand_atoms.dtype)

    safe_ligand_atoms = torch.nan_to_num(
        ligand_atoms.float(), nan=0.0, posinf=100.0, neginf=-100.0
    ).clamp(min=-100.0, max=100.0)
    safe_pocket_pos = torch.nan_to_num(
        pocket_pos.float(), nan=0.0, posinf=100.0, neginf=-100.0
    ).clamp(min=-100.0, max=100.0)
    distances = torch.cdist(safe_ligand_atoms, safe_pocket_pos)
    pair_mask = ligand_mask.unsqueeze(-1) & pocket_mask_2d.unsqueeze(1)
    allowed = ligand_radii.unsqueeze(-1) + pocket_radii.unsqueeze(1) - float(tolerance)
    overlap = torch.where(
        pair_mask,
        F.relu(allowed.float() - distances).clamp(max=10.0),
        torch.zeros_like(distances),
    )
    per_complex_loss = overlap.square().sum(dim=(1, 2)) / ligand_mask.sum(dim=1).clamp_min(1).to(distances.dtype)
    clash_count = (overlap > 0).sum(dim=(1, 2)).to(distances.dtype)
    masked_distances = distances.masked_fill(~pair_mask, float("inf"))
    min_distance = masked_distances.amin(dim=(1, 2))
    if sample_weight is None:
        loss = per_complex_loss.mean()
    else:
        weights = torch.nan_to_num(
            sample_weight.to(device=per_complex_loss.device, dtype=per_complex_loss.dtype),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).reshape(-1).clamp_min(0.0)
        loss = (per_complex_loss * weights).sum() / weights.sum().clamp_min(1e-8)
    return loss, clash_count, min_distance


def contact_distance_loss(
    pred_atoms,
    true_atoms,
    ligand_mask,
    pocket_pos,
    pocket_mask,
    sample_weight=None,
    max_per_atom_loss=25.0,
):
    """Keep predicted atoms at the native pocket-contact depth while avoiding clashes."""
    if pocket_pos is None or pocket_pos.shape[1] == 0:
        return pred_atoms.sum() * 0.0
    pocket_valid = pocket_mask.squeeze(-1) > 0.5
    pair_mask = ligand_mask.unsqueeze(-1) & pocket_valid.unsqueeze(1)
    safe_pred_atoms = torch.nan_to_num(
        pred_atoms.float(), nan=0.0, posinf=100.0, neginf=-100.0
    ).clamp(min=-100.0, max=100.0)
    safe_true_atoms = torch.nan_to_num(
        true_atoms.float(), nan=0.0, posinf=100.0, neginf=-100.0
    ).clamp(min=-100.0, max=100.0)
    safe_pocket_pos = torch.nan_to_num(
        pocket_pos.float(), nan=0.0, posinf=100.0, neginf=-100.0
    ).clamp(min=-100.0, max=100.0)
    pred_dist = torch.cdist(safe_pred_atoms, safe_pocket_pos).masked_fill(~pair_mask, float("inf"))
    true_dist = torch.cdist(safe_true_atoms, safe_pocket_pos).masked_fill(~pair_mask, float("inf"))
    pred_min = pred_dist.amin(dim=2)
    true_min = true_dist.amin(dim=2)
    valid = ligand_mask & torch.isfinite(pred_min) & torch.isfinite(true_min)
    if not valid.any():
        return pred_atoms.sum() * 0.0
    safe_pred = torch.where(valid, pred_min, torch.zeros_like(pred_min))
    safe_true = torch.where(valid, true_min, torch.zeros_like(true_min))
    per_atom = F.smooth_l1_loss(safe_pred, safe_true, reduction="none")
    per_atom = per_atom.clamp(max=float(max_per_atom_loss))
    per_atom = per_atom * valid.to(per_atom.dtype)
    per_complex = per_atom.sum(dim=1) / valid.sum(dim=1).clamp_min(1).to(per_atom.dtype)
    if sample_weight is None:
        return per_complex.mean()
    weights = torch.nan_to_num(
        sample_weight.to(device=per_complex.device, dtype=per_complex.dtype),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).reshape(-1).clamp_min(0.0)
    return (per_complex * weights).sum() / weights.sum().clamp_min(1e-8)


def heavy_atom_center(coords, atomic_numbers=None):
    """Mass-weighted center for numpy-like coordinate checks without RDKit."""
    if len(coords) == 0:
        raise ValueError("Cannot compute the center of an empty coordinate set")
    if atomic_numbers is None:
        return coords.mean(axis=0)
    masses = []
    for value in atomic_numbers:
        z = int(value)
        masses.append(float(z) if z > 0 else 12.0)
    weights = torch.as_tensor(masses, dtype=torch.float64)
    xyz = torch.as_tensor(coords, dtype=torch.float64)
    return ((xyz * weights[:, None]).sum(dim=0) / weights.sum().clamp_min(1e-8)).cpu().numpy()
