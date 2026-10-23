# src/datasets.py
import os
import pickle
import lmdb
import numpy as np
import torch
from rdkit import Chem
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from rdkit.Chem import BRICS
import pandas as pd
from src import const
from src.fragment_vocab import (
    ensure_fragment_vocabulary,
    fragment_graph_to_rigid_tensors,
    load_fragment_vocabulary,
)

# =========================================================
# Config
# =========================================================
MAX_POCKET_ATOMS = 256
MAX_FRAG_NODES = 64
DEBUG_EVERY_N = 0


def _build_frames_from_pos_rot(pos: torch.Tensor, rot: torch.Tensor):
    """
    Construct frames with support for batched inputs.
    pos: [B, N, 3] or [N, 3]
    rot: [B, N, 3, 3] or [N, 3, 3]
    Returns: [B, N, 4, 4] or [N, 4, 4]
    """
    device, dtype = pos.device, pos.dtype

    # Get the original shape.
    original_shape = pos.shape[:-1]  # Get (B, N) or (N,).
    n_nodes = pos.size(-2)

    # Initialize the identity-matrix base.
    # Use reshape to accommodate the batch dimension.
    if pos.dim() == 3:  # Batched input [B, N, 3].
        batch_size = pos.size(0)
        frames = torch.eye(4, device=device, dtype=dtype).view(1, 1, 4, 4).repeat(batch_size, n_nodes, 1, 1)
        frames[..., :3, :3] = rot
        frames[..., :3, 3] = pos
    else:  # Single-sample input [N, 3].
        frames = torch.eye(4, device=device, dtype=dtype).view(1, 4, 4).repeat(n_nodes, 1, 1)
        frames[:, :3, :3] = rot
        frames[:, :3, 3] = pos

    return frames


def _bondtype_from_int(b):
    """
    Common bond_type conventions in LMDB:
    1=single, 2=double, 3=triple, 4=aromatic
    Use a robust mapping to accommodate minor differences across datasets.
    """
    if isinstance(b, (np.ndarray, list)):
        b = int(b[0])
    b = int(b)
    if b == 1:
        return Chem.BondType.SINGLE
    if b == 2:
        return Chem.BondType.DOUBLE
    if b == 3:
        return Chem.BondType.TRIPLE
    if b == 4:
        return Chem.BondType.AROMATIC
    # fallback
    return Chem.BondType.SINGLE


def _bondtype_to_int(bond_type):
    if bond_type == Chem.BondType.SINGLE:
        return 1
    if bond_type == Chem.BondType.DOUBLE:
        return 2
    if bond_type == Chem.BondType.TRIPLE:
        return 3
    if bond_type == Chem.BondType.AROMATIC:
        return 4
    return 0


def brics_fragment_atom_indices(mol):
    """
    Return the atom-index list for each BRICS fragment
    and the connectivity edge list (edge_list) between fragments.
    """
    brics_bonds = list(BRICS.FindBRICSBonds(mol))
    if len(brics_bonds) == 0:
        return [tuple(range(mol.GetNumAtoms()))], []

    # Extract the atom indices of the bonds to cut.
    bond_ids = []
    broken_atom_pairs = []
    for (a, b), _labels in brics_bonds:
        bond = mol.GetBondBetweenAtoms(int(a), int(b))
        if bond is not None:
            bond_ids.append(bond.GetIdx())
            broken_atom_pairs.append((int(a), int(b), _bondtype_to_int(bond.GetBondType())))

    if len(bond_ids) == 0:
        return [tuple(range(mol.GetNumAtoms()))], []

    # Split the molecule.
    frag_mol = Chem.FragmentOnBonds(mol, bond_ids, addDummies=False)
    frags = Chem.GetMolFrags(frag_mol, asMols=False, sanitizeFrags=False)

    # Map atom IDs to fragment IDs to recover inter-fragment connectivity.
    atom2frag = {}
    for frag_idx, atom_indices in enumerate(frags):
        for a_idx in atom_indices:
            atom2frag[a_idx] = frag_idx

    # Build ground-truth fragment-level edges (Frag_i, Frag_j).
    frag_edges = []
    for a, b, bond_type in broken_atom_pairs:
        f_a = atom2frag.get(a)
        f_b = atom2frag.get(b)
        if f_a is not None and f_b is not None and f_a != f_b:
            frag_edges.append((f_a, f_b, bond_type))
            frag_edges.append((f_b, f_a, bond_type))  # Add both directions for an undirected graph.

    # Remove duplicates.
    frag_edges = list(set(frag_edges))

    return list(frags), frag_edges


def build_ligand_mol_from_sample(sample):
    """
    Use ligand_element, ligand_bond_index/bond_type, and ligand_pos from LMDB
    to build an RDKit Mol with a conformer.
    [Fix] Explicitly compute RingInfo to avoid BRICS errors.
    """
    elems = np.asarray(sample['ligand_element'])
    pos = np.asarray(sample['ligand_pos'], dtype=np.float32)
    bond_index = np.asarray(sample['ligand_bond_index'])
    bond_type = np.asarray(sample['ligand_bond_type'])

    # bond_index may have shape [2, B] or [B, 2].
    if bond_index.ndim == 2 and bond_index.shape[0] == 2:
        bonds = bond_index.T
    else:
        bonds = bond_index

    rw = Chem.RWMol()
    for z in elems:
        a = Chem.Atom(int(z))
        rw.AddAtom(a)

    for i, (a, b) in enumerate(bonds):
        bt = _bondtype_from_int(bond_type[i])
        a = int(a);
        b = int(b)
        if a == b:
            continue
        # Avoid adding duplicates.
        if rw.GetBondBetweenAtoms(a, b) is None:
            rw.AddBond(a, b, bt)

    mol = rw.GetMol()

    # conformer
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        x, y, z = pos[i]
        conf.SetAtomPosition(i, (float(x), float(y), float(z)))
    mol.AddConformer(conf, assignId=True)

    # --- Begin core fix ---
    try:
        # 1. Try standard sanitization.
        Chem.SanitizeMol(mol)
    except Exception:
        # If sanitization fails, try manually repairing basic properties rather than silently continuing.
        try:
            mol.UpdatePropertyCache(strict=False)
        except:
            return None  # Return None if repair fails; the dataloader will skip this sample.

    # 2. Explicitly compute ring information (RingInfo).
    # BRICS requires ring information; otherwise it raises a precondition violation.
    try:
        Chem.GetSymmSSSR(mol)
    except:
        return None  # Skip the sample if ring information cannot be computed.
    # --- End core fix ---

    return mol


# =========================================================
# Utils: rigid transformations and the Kabsch algorithm (following the core FrameDiff logic).
# =========================================================
def compute_rigid_transform(positions: torch.Tensor):
    """
    positions:[M,3] -> center[3], rot[3,3]
    Use covariance SVD/PCA to obtain a stable, right-handed rigid orientation.
    """
    device, dtype = positions.device, positions.dtype
    if positions.size(0) < 3:
        center = positions.mean(dim=0)
        return center, torch.eye(3, device=device, dtype=dtype)

    center = positions.mean(dim=0)
    X = positions - center
    try:
        H = X.t() @ X
        U, S, Vh = torch.linalg.svd(H)
        rot = U
        if torch.det(rot) < 0:
            rot[:, 2] *= -1
    except Exception:
        rot = torch.eye(3, device=device, dtype=dtype)
    return center, rot


def kabsch_alignment(real_pts: torch.Tensor, ideal_pts: torch.Tensor):
    """
    Implement Kabsch: compute rotation R and translation t so that R @ ideal + t approximates real.
    Follow the rigid-fragment reconstruction approach in Section 3.3 of the paper.
    """
    mu_real = real_pts.mean(dim=0)
    mu_ideal = ideal_pts.mean(dim=0)

    real_c = real_pts - mu_real
    ideal_c = ideal_pts - mu_ideal

    # Compute the covariance matrix.
    H = ideal_c.t() @ real_c
    U, S, Vh = torch.linalg.svd(H)

    # Compute the rotation matrix R.
    R = Vh.t() @ U.t()

    # Correct reflections to ensure det(R) = 1.
    if torch.linalg.det(R) < 0:
        Vh_mod = Vh.clone()
        Vh_mod[2, :] *= -1
        R = Vh_mod.t() @ U.t()

    return mu_real, R  # mu_real is the translation vector t.


def _build_frames(pos, rot):
    """Combine R and t into [N, 4, 4] transformation matrices."""
    N = pos.shape[0]
    frames = torch.eye(4).repeat(N, 1, 1).to(pos.device)
    frames[:, :3, :3] = rot
    frames[:, :3, 3] = pos
    return frames


# Padding Helpers
def _pad_2d(t: torch.Tensor, max_len: int):
    n, d = t.shape
    out = t.new_zeros((max_len, d))
    out[:n] = t
    return out


def _pad_rot(t: torch.Tensor, max_len: int):
    out = torch.eye(3).view(1, 3, 3).repeat(max_len, 1, 1).to(t.device)
    out[:t.shape[0]] = t
    return out


def _dedupe_paths(paths):
    deduped = []
    seen = set()
    for path in paths:
        normalized = os.path.normcase(os.path.abspath(path))
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(path)
    return deduped


def _first_existing_path(paths, label):
    paths = _dedupe_paths(paths)
    for path in paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"{label} not found. Tried: {paths}")


def _split_indices_for_split(splits, split):
    if split in splits:
        target_indices = splits[split]
    elif split == 'val' and 'valid' in splits:
        target_indices = splits['valid']
    else:
        target_indices = []
    return target_indices


def _filter_valid_lmdb_keys(env, target_keys, split, found_split, strict_split=False):
    print(f"Verifying {len(target_keys)} keys for '{split}'...")
    valid_keys = []
    missing_keys = []
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        if not found_split:
            if strict_split:
                raise FileNotFoundError(f"No split file found for split '{split}'.")
            print("Warning: No split keys found. Using ALL data.")
            for key, _ in cursor:
                valid_keys.append(key)
        else:
            for key in tqdm(target_keys, desc="Filtering keys"):
                if cursor.set_key(key):
                    valid_keys.append(key)
                else:
                    missing_keys.append(key.decode() if isinstance(key, bytes) else str(key))

    if strict_split and missing_keys and split == 'test':
        raise RuntimeError(
            f"LMDB is missing {len(missing_keys)} keys from split '{split}': {missing_keys[:20]}"
        )
    if missing_keys and split != 'test':
        print(f"Pre-audit excluded {len(missing_keys)} unavailable keys from split '{split}'.")

    if not valid_keys:
        raise RuntimeError(f"No valid LMDB keys found for split '{split}'.")
    return valid_keys


# =========================================================
# PDBbindDataset: rigid-fragment-level reconstruction.
# =========================================================

class PDBbindDataset(Dataset):
    def __init__(self, data_path, prefix, split='train', device="cpu"):
        super().__init__()
        self.data_path = data_path
        self.prefix = prefix
        self.split = split
        self.device = device
        self.strict_identity = False
        self.fragment_vocab = ensure_fragment_vocabulary(data_path, prefix)
        prefix_lower = prefix.lower()

        # 1. Locate the LMDB. Do not silently fall back between CrossDock and PDBbind.
        prefix_lower = prefix.lower()
        if "crossdock" in prefix_lower:
            raise ValueError("Use CrossDockDataset for CrossDock prefixes.")
        if "pdbbind" in prefix_lower:
            possible_lmdb = [
                os.path.join(data_path, f"{prefix}.lmdb"),
                os.path.join(data_path, "PDBbind_v2020_pocket10_processed_final.lmdb"),
            ]
        else:
            possible_lmdb = [os.path.join(data_path, f"{prefix}.lmdb")]
        if '_split' in prefix:
            possible_lmdb.append(os.path.join(data_path, f"{prefix.replace('_split', '')}.lmdb"))

        lmdb_path = _first_existing_path(possible_lmdb, "PDBbind LMDB")
        self.lmdb_path = lmdb_path

        self.env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False, readahead=False, meminit=False,
                             max_readers=256)

        # 2. Load the split and validate keys.
        if "pdbbind" in prefix_lower:
            possible_split = [
                os.path.join(data_path, f"{prefix}_split.pt"),
                os.path.join(data_path, "PDBbind_pocket10_split.pt"),
            ]
        else:
            possible_split = [os.path.join(data_path, f"{prefix}_split.pt")]

        target_keys = []
        found_split = False
        for sp in possible_split:
            if os.path.exists(sp):
                print(f"📖 Loading split: {os.path.basename(sp)}")
                splits = torch.load(sp)
                target_indices = _split_indices_for_split(splits, split)
                target_keys = [str(int(i)).encode() for i in target_indices]
                found_split = True
                break

        # 3. Filter out missing keys to prevent KeyError.
        print(f"🔍 Verifying {len(target_keys)} keys for '{split}'...")
        self.valid_keys = []
        with self.env.begin(write=False) as txn:
            cursor = txn.cursor()
            if not found_split or len(target_keys) == 0:
                print("⚠️ Warning: No split keys found. Using ALL data.")
                for key, _ in cursor: self.valid_keys.append(key)
            else:
                for k in tqdm(target_keys, desc="Filtering keys"):
                    if cursor.set_key(k): self.valid_keys.append(k)

        self.length = len(self.valid_keys)
        print(f"✅ Dataset ({split}) ready. {self.length} valid samples.")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 1. Basic loading logic with sample skipping.
        max_attempts = 1 if self.strict_identity else 100
        requested_key = self.valid_keys[idx]
        last_error = None
        for attempt in range(max_attempts):
            current_idx = (idx + attempt) % len(self.valid_keys)
            key = self.valid_keys[current_idx]

            try:
                with self.env.begin(write=False) as txn:
                    value = txn.get(key)
                if value is None: continue
                sample = pickle.loads(value)
                if getattr(self, 'pocket_only', False):
                    return self._build_pocket_only_item(sample, key, current_idx)

                # 2. Generate fragment information on the fly.
                # Since the data lacks 'fragments', construct and fragment an RDKit molecule from atoms and bonds.
                from src.datasets import build_ligand_mol_from_sample, brics_fragment_atom_indices

                mol = build_ligand_mol_from_sample(sample)
                if mol is None:
                    continue

                # Use BRICS to partition atom indices.
                rigid_frags, frag_edges, attachment_edges = fragment_graph_to_rigid_tensors(
                    mol,
                    self.fragment_vocab,
                )
                if not rigid_frags:
                    continue

                n_frag = len(rigid_frags)
                true_bonds = torch.zeros((n_frag, n_frag), dtype=torch.long)
                true_attach_src = torch.full((n_frag, n_frag), -1, dtype=torch.long)
                true_attach_dst = torch.full((n_frag, n_frag), -1, dtype=torch.long)
                for u, v, bond_type in frag_edges:
                    true_bonds[u, v] = max(int(true_bonds[u, v].item()), int(bond_type))
                for u, v, attach_u, attach_v, _bond_type in attachment_edges:
                    true_attach_src[u, v] = int(attach_u)
                    true_attach_dst[u, v] = int(attach_v)

                # 3. Extract basic information.
                ligand_pos = torch.tensor(sample['ligand_pos'], dtype=const.TORCH_FLOAT)
                # Without residue_names, use the paper's 22-class vocabulary by
                # mapping fragments to 'UNK' (ID: 20), unless residue names are available.

                frag_centers, frag_rots, frag_types, frag_type_ids = [], [], [], []

                for fragment_info in rigid_frags:
                    type_idx = int(fragment_info["type_idx"])
                    frag_centers.append(fragment_info["center"])
                    frag_rots.append(fragment_info["rotation"])
                    frag_type_ids.append(type_idx)

                    # Default to UNK (ID 20 in the 22-class vocabulary).
                    one_hot = np.zeros(const.GEOM_NUMBER_OF_FRAGMENT_TYPES, dtype=np.float32)
                    one_hot[type_idx] = 1.0

                    # For generic ligand fragmentation, atom names may not match the backbone template.
                    # If Kabsch alignment cannot be applied, fall back to the center of mass.
                    frag_types.append(one_hot)

                # 4. Package the return values.
                pocket_frames = self._get_pocket_frames(sample)
                n_pocket = pocket_frames.shape[0]
                protein_element = torch.as_tensor(
                    sample['protein_element'][:MAX_POCKET_ATOMS], dtype=const.TORCH_INT
                )
                required_protein_fields = (
                    'protein_atom_to_aa_type',
                    'protein_is_backbone',
                )
                missing_protein_fields = [
                    field for field in required_protein_fields if field not in sample
                ]
                if missing_protein_fields:
                    raise KeyError(
                        "CrossDock pocket chemistry fields are missing: "
                        + ", ".join(missing_protein_fields)
                    )
                protein_aa_type = torch.as_tensor(
                    sample['protein_atom_to_aa_type'][:MAX_POCKET_ATOMS],
                    dtype=const.TORCH_INT,
                )
                protein_is_backbone = torch.as_tensor(
                    sample['protein_is_backbone'][:MAX_POCKET_ATOMS],
                    dtype=torch.bool,
                )
                if not (
                    protein_element.shape[0]
                    == protein_aa_type.shape[0]
                    == protein_is_backbone.shape[0]
                    == n_pocket
                ):
                    raise ValueError(
                        "Pocket chemistry fields and pocket coordinates have different lengths"
                    )
                ligand_atom_pos = torch.as_tensor(sample['ligand_pos'], dtype=const.TORCH_FLOAT)
                reference_center = torch.as_tensor(
                    sample.get('ligand_center_of_mass', ligand_atom_pos.mean(dim=0)),
                    dtype=const.TORCH_FLOAT,
                )
                reference_size = (
                    ligand_atom_pos.max(dim=0).values - ligand_atom_pos.min(dim=0).values + 10.0
                ).clamp_min(20.0)
                return {
                    "uuid": key.decode() if isinstance(key, bytes) else str(key),
                    "dataset_index": current_idx,
                    "protein_filename": sample.get("protein_filename", ""),
                    "ligand_filename": sample.get("ligand_filename", ""),
                    "positions": torch.stack(frag_centers),
                    "frag_rotations": torch.stack(frag_rots),
                    "one_hot": torch.tensor(np.stack(frag_types), dtype=const.TORCH_FLOAT),
                    "fragment_type_ids": torch.tensor(frag_type_ids, dtype=const.TORCH_INT),
                    "charges": torch.zeros(n_frag, dtype=const.TORCH_FLOAT),
                    "anchors": torch.zeros(n_frag, dtype=const.TORCH_FLOAT),
                    "fragment_mask": torch.ones(n_frag, dtype=const.TORCH_FLOAT),
                    "linker_mask": torch.ones(n_frag, dtype=const.TORCH_FLOAT),
                    "atom_mask": torch.ones(n_frag, dtype=const.TORCH_FLOAT),
                    "pocket_frames": pocket_frames,
                    "pocket_mask": torch.ones(n_pocket, dtype=const.TORCH_FLOAT),
                    "pocket_element": protein_element,
                    "pocket_aa_type": protein_aa_type,
                    "pocket_is_backbone": protein_is_backbone,
                    "reference_box_center": reference_center,
                    "reference_box_size": reference_size,
                    "true_bonds": true_bonds,
                    "true_attach_src": true_attach_src,
                    "true_attach_dst": true_attach_dst,
                }

            except Exception as e:
                last_error = e
                # print(f"Sample {key} fragmentation failed: {e}")
                continue

        key_text = requested_key.decode() if isinstance(requested_key, bytes) else str(requested_key)
        raise RuntimeError(f"Failed to load requested LMDB key {key_text} without target substitution: {last_error}")

    def _build_pocket_only_item(self, sample, key, current_idx):
        """Build de-novo context without mapping the reference ligand to fragment types."""
        pocket_frames = self._get_pocket_frames(sample)
        n_pocket = pocket_frames.shape[0]
        required = (
            'protein_element', 'protein_atom_to_aa_type',
            'protein_is_backbone', 'ligand_pos',
        )
        missing = [field for field in required if field not in sample]
        if missing:
            raise KeyError("Pocket-only sample is missing: " + ", ".join(missing))

        protein_element = torch.as_tensor(
            sample['protein_element'][:MAX_POCKET_ATOMS], dtype=const.TORCH_INT
        )
        protein_aa_type = torch.as_tensor(
            sample['protein_atom_to_aa_type'][:MAX_POCKET_ATOMS], dtype=const.TORCH_INT
        )
        protein_is_backbone = torch.as_tensor(
            sample['protein_is_backbone'][:MAX_POCKET_ATOMS], dtype=torch.bool
        )
        if not (
            protein_element.shape[0] == protein_aa_type.shape[0]
            == protein_is_backbone.shape[0] == n_pocket
        ):
            raise ValueError("Pocket chemistry fields and pocket coordinates have different lengths")

        ligand_pos = torch.as_tensor(sample['ligand_pos'], dtype=const.TORCH_FLOAT)
        ligand_element = sample.get('ligand_element')
        if ligand_element is not None:
            ligand_element = torch.as_tensor(ligand_element).reshape(-1)
            heavy_mask = ligand_element > 1
            if heavy_mask.shape[0] == ligand_pos.shape[0] and heavy_mask.any():
                heavy_pos = ligand_pos[heavy_mask]
            else:
                heavy_pos = ligand_pos
        else:
            heavy_pos = ligand_pos
        reference_center = torch.as_tensor(
            sample.get('ligand_center_of_mass', heavy_pos.mean(dim=0)),
            dtype=const.TORCH_FLOAT,
        )
        reference_size = (
            heavy_pos.max(dim=0).values - heavy_pos.min(dim=0).values + 10.0
        ).clamp_min(20.0)
        vocab_size = len(self.fragment_vocab.get('entries', []))
        if vocab_size <= 0:
            raise ValueError("Pocket-only generation requires a non-empty fragment vocabulary")

        return {
            'uuid': key.decode() if isinstance(key, bytes) else str(key),
            'dataset_index': current_idx,
            'protein_filename': sample.get('protein_filename', ''),
            'ligand_filename': sample.get('ligand_filename', ''),
            'positions': reference_center.view(1, 3).clone(),
            'frag_rotations': torch.eye(3, dtype=const.TORCH_FLOAT).view(1, 3, 3),
            'one_hot': torch.full((1, vocab_size), 1.0 / vocab_size, dtype=const.TORCH_FLOAT),
            'fragment_type_ids': torch.full((1,), -1, dtype=const.TORCH_INT),
            'charges': torch.zeros(1, dtype=const.TORCH_FLOAT),
            'anchors': torch.zeros(1, dtype=const.TORCH_FLOAT),
            'fragment_mask': torch.zeros(1, dtype=const.TORCH_FLOAT),
            'linker_mask': torch.ones(1, dtype=const.TORCH_FLOAT),
            'atom_mask': torch.ones(1, dtype=const.TORCH_FLOAT),
            'pocket_frames': pocket_frames,
            'pocket_mask': torch.ones(n_pocket, dtype=const.TORCH_FLOAT),
            'pocket_element': protein_element,
            'pocket_aa_type': protein_aa_type,
            'pocket_is_backbone': protein_is_backbone,
            'reference_box_center': reference_center,
            'reference_box_size': reference_size,
        }

    def _get_pocket_frames(self, sample):
        # Simplified approach: convert protein atom positions directly to frames with identity rotations.
        prot_pos = torch.tensor(sample['protein_pos'][:MAX_POCKET_ATOMS], dtype=const.TORCH_FLOAT)
        frames = torch.eye(4).repeat(prot_pos.shape[0], 1, 1)
        frames[:, :3, 3] = prot_pos
        return frames


class CrossDockDataset(PDBbindDataset):
    def __init__(self, data_path, prefix, split='train', device="cpu", split_file=None,
                 fragment_vocab_path=None, pocket_only=False):
        Dataset.__init__(self)
        self.data_path = data_path
        self.prefix = prefix
        self.split = split
        self.device = device
        self.strict_identity = True
        self.pocket_only = bool(pocket_only)
        if fragment_vocab_path:
            if not os.path.exists(fragment_vocab_path):
                raise FileNotFoundError(f"Explicit fragment vocabulary not found: {fragment_vocab_path}")
            # An explicit path is an intentional request to reuse the model's
            # training vocabulary with a rebuilt LMDB containing the same data.
            self.fragment_vocab = load_fragment_vocabulary(fragment_vocab_path)
        else:
            self.fragment_vocab = ensure_fragment_vocabulary(data_path, prefix)

        prefix_lower = prefix.lower()
        possible_lmdb = [os.path.join(data_path, f"{prefix}.lmdb")]
        if '_split' in prefix:
            possible_lmdb.append(os.path.join(data_path, f"{prefix.replace('_split', '')}.lmdb"))
        if prefix_lower != "crossdocked_pocket10":
            possible_lmdb.append(os.path.join(data_path, "crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb"))

        lmdb_path = _first_existing_path(possible_lmdb, "CrossDock LMDB")
        self.lmdb_path = lmdb_path

        self.env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False, readahead=False, meminit=False,
                             max_readers=256)

        possible_split = [
            split_file,
            os.path.join(data_path, f"{prefix}_split.pt"),
            os.path.join(data_path, "crossdocked_pocket10_pose_split.pt"),
        ]
        possible_split = [path for path in possible_split if path]

        target_keys = []
        found_split = False
        for sp in possible_split:
            if os.path.exists(sp):
                print(f"Loading split: {os.path.basename(sp)}")
                splits = torch.load(sp)
                target_indices = _split_indices_for_split(splits, split)
                target_keys = [str(int(i)).encode() for i in target_indices]
                found_split = True
                break

        self.valid_keys = _filter_valid_lmdb_keys(self.env, target_keys, split, found_split, strict_split=True)
        self.length = len(self.valid_keys)
        print(f"CrossDockDataset ({split}) ready. {self.length} valid samples.")


class MOADDataset(Dataset):
    def __init__(self, data=None, data_path=None, prefix=None, device=None):
        assert (data is not None) or all(x is not None for x in (data_path, prefix, device))
        if data is not None:
            self.data = data
            self.prefix = prefix
            return

        self.prefix = prefix
        # Automatically detect the pocket mode, such as 4A or 6A.
        if '.' in prefix:
            prefix_base, pocket_mode = prefix.split('.')
        else:
            prefix_base = prefix
            pocket_mode = "full"

        dataset_path = os.path.join(data_path, f'{prefix_base}_{pocket_mode}_rigid.pt')

        if os.path.exists(dataset_path):
            print(f"📦 Loading preprocessed MOAD dataset: {dataset_path}")
            self.data = torch.load(dataset_path, map_location=device)
        else:
            print(f"🚀 Preprocessing MOAD dataset (Rigid-Fragment Mode)...")
            self.data = self.preprocess(data_path, prefix_base, device)
            torch.save(self.data, dataset_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        # Support both dictionary and list formats.
        if isinstance(self.data, dict):
            # Try to obtain the matching split (train/val/test) from the prefix.
            for key in ['train', 'val', 'test']:
                if key in self.prefix and key in self.data:
                    return self.data[key][item]
            # Fall back to the first available list.
            return next(iter(self.data.values()))[item]
        return self.data[item]

    @staticmethod
    def preprocess(data_path, prefix, device):
        """
        MOAD preprocessing: map all-atom PDB structures to 22 rigid-fragment classes.
        """
        lmdb_path = os.path.join(data_path, f'{prefix}.lmdb')
        if not os.path.exists(lmdb_path):
            raise FileNotFoundError(f"LMDB not found at: {lmdb_path}")

        data = []
        env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
        fragment_vocab = ensure_fragment_vocabulary(data_path, prefix)

        with env.begin(write=False) as txn:
            cursor = txn.cursor()
            for idx, (key, value) in enumerate(tqdm(cursor, desc="Rigid Fragmentizing MOAD")):
                if key is None: continue
                sample = pickle.loads(value)

                try:
                    # 1. Process the protein pocket (rigid frames).
                    protein_pos = np.asarray(sample['protein_pos'], dtype=np.float32)
                    if protein_pos.shape[0] > MAX_POCKET_ATOMS:
                        protein_pos = protein_pos[:MAX_POCKET_ATOMS]

                    pocket_pos = torch.as_tensor(protein_pos, dtype=const.TORCH_FLOAT)
                    n_pocket = pocket_pos.size(0)

                    # Initialize pocket frames with identity rotation matrices.
                    pocket_rot = torch.eye(3).view(1, 3, 3).repeat(n_pocket, 1, 1)
                    pocket_frames = _build_frames_from_pos_rot(pocket_pos, pocket_rot)
                    pocket_mask = torch.ones(n_pocket, dtype=const.TORCH_FLOAT).to(device)

                    # 2. Fragment the ligand (BRICS -> rigid nodes).
                    mol = build_ligand_mol_from_sample(sample)
                    if mol is None:
                        continue

                    rigid_frags, frag_edges, attachment_edges = fragment_graph_to_rigid_tensors(
                        mol,
                        fragment_vocab,
                    )

                    if len(rigid_frags) > MAX_FRAG_NODES:
                        rigid_frags = rigid_frags[:MAX_FRAG_NODES]
                        frag_edges = [
                            (u, v, bond_type)
                            for u, v, bond_type in frag_edges
                            if u < MAX_FRAG_NODES and v < MAX_FRAG_NODES
                        ]
                        attachment_edges = [
                            (u, v, attach_u, attach_v, bond_type)
                            for u, v, attach_u, attach_v, bond_type in attachment_edges
                            if u < MAX_FRAG_NODES and v < MAX_FRAG_NODES
                        ]

                    frag_centers, frag_rots, frag_types, frag_type_ids = [], [], [], []
                    for fragment_info in rigid_frags:
                        type_idx = int(fragment_info["type_idx"])
                        frag_centers.append(fragment_info["center"])
                        frag_rots.append(fragment_info["rotation"])
                        frag_type_ids.append(type_idx)

                        one_hot = np.zeros(const.GEOM_NUMBER_OF_FRAGMENT_TYPES, dtype=np.float32)
                        one_hot[type_idx] = 1.0
                        frag_types.append(one_hot)

                    n_frag = len(frag_centers)
                    if n_frag == 0:
                        continue

                    true_bonds = torch.zeros((n_frag, n_frag), dtype=torch.long)
                    true_attach_src = torch.full((n_frag, n_frag), -1, dtype=torch.long)
                    true_attach_dst = torch.full((n_frag, n_frag), -1, dtype=torch.long)
                    for u, v, bond_type in frag_edges:
                        true_bonds[u, v] = max(int(true_bonds[u, v].item()), int(bond_type))
                    for u, v, attach_u, attach_v, _bond_type in attachment_edges:
                        true_attach_src[u, v] = int(attach_u)
                        true_attach_dst[u, v] = int(attach_v)

                    item = {
                        'uuid': key.decode() if isinstance(key, bytes) else str(key),
                        'num_atoms': n_frag,
                        'positions': torch.stack(frag_centers),
                        'frag_rotations': torch.stack(frag_rots),
                        'one_hot': torch.tensor(np.stack(frag_types), dtype=const.TORCH_FLOAT),
                        'fragment_type_ids': torch.tensor(frag_type_ids, dtype=const.TORCH_INT),
                        'charges': torch.zeros(n_frag, dtype=const.TORCH_FLOAT),
                        'anchors': torch.zeros(n_frag, dtype=const.TORCH_FLOAT),
                        'fragment_mask': torch.ones(n_frag, dtype=const.TORCH_FLOAT),
                        'linker_mask': torch.ones(n_frag, dtype=const.TORCH_FLOAT),
                        'atom_mask': torch.ones(n_frag, dtype=const.TORCH_FLOAT),
                        'pocket_frames': pocket_frames,
                        'pocket_mask': pocket_mask,
                        'true_bonds': true_bonds,
                        'true_attach_src': true_attach_src,
                        'true_attach_dst': true_attach_dst,
                    }
                    data.append(item)
                    continue
                    frags, frag_edges = brics_fragment_atom_indices(mol)

                    if len(frags) > MAX_FRAG_NODES:
                        frags = frags[:MAX_FRAG_NODES]
                        # Keep only edges whose u and v indices are both within valid bounds.
                        frag_edges = [(u, v) for u, v in frag_edges if u < MAX_FRAG_NODES and v < MAX_FRAG_NODES]

                    ligand_pos_all = torch.tensor(sample['ligand_pos'], dtype=const.TORCH_FLOAT)
                    atom_names = [n.strip() for n in sample.get('ligand_atom_names', [])]
                    res_names = sample.get('residue_names', ['ALA'] * len(frags))

                    frag_centers, frag_rots, frag_types = [], [], []

                    for f_idx, atom_ids in enumerate(frags):
                        atom_ids = list(atom_ids)
                        pts = ligand_pos_all[atom_ids]

                        # Get the 22-class IDs defined in the paper.
                        aa_name = res_names[f_idx] if f_idx < len(res_names) else 'ALA'
                        aa_type_id = const.AA2NUM.get(aa_name, const.AA2NUM['UNK'])

                        # Use Kabsch alignment to obtain precise R and t.
                        ideal_info = const.IDEAL_COORDS[aa_type_id]
                        name2idx = const.AA_NAME2IDX[aa_type_id]

                        real_subset, ideal_subset = [], []
                        for global_idx in atom_ids:
                            # Try matching atom names to improve alignment accuracy.
                            name = atom_names[global_idx] if global_idx < len(atom_names) else ""
                            if name in name2idx:
                                real_subset.append(ligand_pos_all[global_idx])
                                ideal_subset.append(torch.tensor(ideal_info[name2idx[name]][2]))

                        if len(real_subset) >= 3:
                            t, R = kabsch_alignment(torch.stack(real_subset), torch.stack(ideal_subset))
                        else:
                            # Fall back to the center of mass and identity rotation.
                            t = pts.mean(dim=0)
                            R = torch.eye(3)

                        frag_centers.append(t)
                        frag_rots.append(R)

                        # Construct 22-dimensional one-hot features to match PDBbind and the model input.
                        oh = np.zeros(const.GEOM_NUMBER_OF_FRAGMENT_TYPES)
                        oh[aa_type_id] = 1.0
                        frag_types.append(oh)

                    # 3. Package the sample.
                    n_frag = len(frag_centers)
                    if n_frag == 0: continue
                    true_bonds = torch.zeros((n_frag, n_frag), dtype=torch.long)
                    for u, v, bond_type in frag_edges:
                        true_bonds[u, v] = max(int(true_bonds[u, v].item()), int(bond_type))

                    item = {
                        'uuid': key.decode() if isinstance(key, bytes) else str(key),
                        'num_atoms': n_frag,  # At fragment level, num_atoms is the number of nodes.
                        'positions': torch.stack(frag_centers),
                        'frag_rotations': torch.stack(frag_rots),
                        'one_hot': torch.tensor(np.stack(frag_types), dtype=const.TORCH_FLOAT),
                        'charges': torch.zeros(n_frag),
                        'anchors': torch.zeros(n_frag),
                        'fragment_mask': torch.ones(n_frag),  # MOAD evaluation typically generates the whole molecule.
                        'linker_mask': torch.zeros(n_frag),  # Set to 1 only for specific inpainting tasks.
                        'pocket_frames': pocket_frames,
                        'pocket_mask': pocket_mask,
                        'true_bonds': true_bonds
                    }
                    data.append(item)

                except Exception as e:
                    # Skip corrupted samples.
                    continue

        env.close()
        print(f"✅ Preprocessing finished. Total valid samples: {len(data)}")
        return data


def read_sdf(sdf_path):
    with Chem.SDMolSupplier(sdf_path, sanitize=False) as supplier:
        for molecule in supplier:
            yield molecule


def get_one_hot(atom, atoms_dict):
    one_hot = np.zeros(len(atoms_dict))
    one_hot[atoms_dict[atom]] = 1
    return one_hot


def parse_molecule(mol, is_geom):
    one_hot = []
    charges = []
    atom2idx = const.GEOM_ATOM2IDX if is_geom else const.ATOM2IDX
    charges_dict = const.GEOM_CHARGES if is_geom else const.CHARGES
    for atom in mol.GetAtoms():
        one_hot.append(get_one_hot(atom.GetSymbol(), atom2idx))
        charges.append(charges_dict[atom.GetSymbol()])
    positions = mol.GetConformer().GetPositions()
    return positions, np.array(one_hot), np.array(charges)


class ZincDataset(Dataset):
    def __init__(self, data_path, prefix, device):
        dataset_path = os.path.join(data_path, f'{prefix}.pt')
        if os.path.exists(dataset_path):
            self.data = torch.load(dataset_path, map_location=device)
        else:
            print(f'Preprocessing dataset with prefix {prefix}')
            self.data = ZincDataset.preprocess(data_path, prefix, device)
            torch.save(self.data, dataset_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if isinstance(self.data, dict):
            if not hasattr(self, '_keys_cache'):
                self._keys_cache = list(self.data.keys())
            real_key = self._keys_cache[item]
            return self.data[real_key]
        return self.data[item]

    @staticmethod
    def preprocess(data_path, prefix, device):
        data = []
        table_path = os.path.join(data_path, f'{prefix}_table.csv')
        fragments_path = os.path.join(data_path, f'{prefix}_frag.sdf')
        linkers_path = os.path.join(data_path, f'{prefix}_link.sdf')

        is_geom = ('geom' in prefix) or ('MOAD' in prefix)
        is_multifrag = 'multifrag' in prefix

        table = pd.read_csv(table_path)
        generator = tqdm(zip(table.iterrows(), read_sdf(fragments_path), read_sdf(linkers_path)), total=len(table))

        for (_, row), fragments, linker in generator:
            uuid = row['uuid']
            name = row['molecule']
            frag_pos, frag_one_hot, frag_charges = parse_molecule(fragments, is_geom=is_geom)
            link_pos, link_one_hot, link_charges = parse_molecule(linker, is_geom=is_geom)

            positions = np.concatenate([frag_pos, link_pos], axis=0)
            one_hot = np.concatenate([frag_one_hot, link_one_hot], axis=0)
            charges = np.concatenate([frag_charges, link_charges], axis=0)
            anchors = np.zeros_like(charges)

            if is_multifrag:
                for anchor_idx in map(int, row['anchors'].split('-')):
                    anchors[anchor_idx] = 1
            else:
                anchors[row['anchor_1']] = 1
                anchors[row['anchor_2']] = 1

            fragment_mask = np.concatenate([np.ones_like(frag_charges), np.zeros_like(link_charges)])
            linker_mask = np.concatenate([np.zeros_like(frag_charges), np.ones_like(link_charges)])

            data.append({
                'uuid': uuid,
                'name': name,
                'positions': torch.tensor(positions, dtype=const.TORCH_FLOAT, device=device),
                'one_hot': torch.tensor(one_hot, dtype=const.TORCH_FLOAT, device=device),
                'charges': torch.tensor(charges, dtype=const.TORCH_FLOAT, device=device),
                'anchors': torch.tensor(anchors, dtype=const.TORCH_FLOAT, device=device),
                'fragment_mask': torch.tensor(fragment_mask, dtype=const.TORCH_FLOAT, device=device),
                'linker_mask': torch.tensor(linker_mask, dtype=const.TORCH_FLOAT, device=device),
                'num_atoms': len(positions),
            })
        return data

def _pad_edge_matrix(m: torch.Tensor, max_len: int, fill_value: int = 0):
    """Pad an [N, N] edge matrix to [max_len, max_len]."""
    n = m.size(0)
    out = m.new_full((max_len, max_len), fill_value)
    out[:n, :n] = m
    return out


def _pad_frames_4x4(t: torch.Tensor, max_len: int):
    out = torch.eye(4, device=t.device, dtype=t.dtype).view(1, 4, 4).repeat(max_len, 1, 1)
    out[:t.shape[0]] = t
    return out


def _fit_1d(t: torch.Tensor, target_len: int, device=None, dtype=None, fill_value=0):
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, device=device, dtype=dtype or const.TORCH_FLOAT)
    if device is not None:
        t = t.to(device)
    if dtype is not None:
        t = t.to(dtype=dtype)
    t = t.reshape(-1)
    out = t.new_full((target_len,), fill_value)
    n = min(t.shape[0], target_len)
    if n > 0:
        out[:n] = t[:n]
    return out

# =========================================================
# Collate: adapt to the FrameDiff batch layout.
# =========================================================

def collate_with_fragment_edges(batch_list):
    batch = [s for s in batch_list if s is not None]
    if len(batch) == 0: return None

    max_lig = max([s['positions'].shape[0] for s in batch])

    lig_pos, lig_rot, lig_onehot, lig_fmask, lig_bonds = [], [], [], [], []

    for s in batch:
        lig_pos.append(_pad_2d(s['positions'], max_lig))
        lig_rot.append(_pad_rot(s['frag_rotations'], max_lig))
        lig_onehot.append(_pad_2d(s['one_hot'], max_lig))

        curr_mask = s['fragment_mask']
        padded_mask = torch.cat([curr_mask, torch.zeros(max_lig - curr_mask.shape[0], device=curr_mask.device)])
        lig_fmask.append(padded_mask)

        # Apply two-dimensional padding to true_bonds.
        if 'true_bonds' in s:
            lig_bonds.append(_pad_edge_matrix(s['true_bonds'], max_lig))

    out = {
        'positions': torch.stack(lig_pos),  # [B, max_lig, 3]
        'frag_rotations': torch.stack(lig_rot),  # [B, max_lig, 3, 3]
        'one_hot': torch.stack(lig_onehot),  # [B, max_lig, 22]
        'fragment_mask': torch.stack(lig_fmask).unsqueeze(-1),  # [B, max_lig, 1]
    }

    # Include topology information in the output when available.
    if len(lig_bonds) > 0:
        out['true_bonds'] = torch.stack(lig_bonds)  # [B, max_lig, max_lig]

    out['frag_frames'] = _build_frames_from_pos_rot(out['positions'], out['frag_rotations'])
    out['atom_mask'] = out['fragment_mask']

    return out


def collate_with_fragment_edges(batch_list):
    batch = [sample for sample in batch_list if sample is not None]
    if len(batch) == 0:
        return None

    max_lig = max(sample['positions'].shape[0] for sample in batch) + 2
    max_pocket = max([sample['pocket_frames'].shape[0] for sample in batch if 'pocket_frames' in sample] + [0])

    lig_pos, lig_rot, lig_onehot = [], [], []
    lig_fmask, lig_linkmask, lig_atommask = [], [], []
    lig_charges, lig_anchors, lig_type_ids = [], [], []
    lig_slotmask = []
    lig_bonds, lig_attach_src, lig_attach_dst = [], [], []
    pocket_frames, pocket_masks, pocket_elements = [], [], []
    pocket_aa_types, pocket_is_backbone = [], []

    for sample in batch:
        n_lig = sample['positions'].shape[0]
        device = sample['positions'].device

        lig_pos.append(_pad_2d(sample['positions'], max_lig))
        lig_rot.append(_pad_rot(sample['frag_rotations'], max_lig))
        lig_onehot.append(_pad_2d(sample['one_hot'], max_lig))

        fragment_mask = _fit_1d(sample.get('fragment_mask', torch.ones(n_lig, device=device, dtype=const.TORCH_FLOAT)), n_lig, device=device, dtype=const.TORCH_FLOAT)
        linker_mask = _fit_1d(sample.get('linker_mask', torch.ones(n_lig, device=device, dtype=const.TORCH_FLOAT)), n_lig, device=device, dtype=const.TORCH_FLOAT)
        atom_mask = _fit_1d(sample.get('atom_mask', torch.ones(n_lig, device=device, dtype=const.TORCH_FLOAT)), n_lig, device=device, dtype=const.TORCH_FLOAT)
        charges = _fit_1d(sample.get('charges', torch.zeros(n_lig, device=device, dtype=const.TORCH_FLOAT)), n_lig, device=device, dtype=const.TORCH_FLOAT)
        anchors = _fit_1d(sample.get('anchors', torch.zeros(n_lig, device=device, dtype=const.TORCH_FLOAT)), n_lig, device=device, dtype=const.TORCH_FLOAT)
        type_ids = _fit_1d(sample.get('fragment_type_ids', torch.argmax(sample['one_hot'], dim=-1).to(dtype=const.TORCH_INT)), n_lig, device=device, dtype=const.TORCH_INT, fill_value=-1)
        slot_mask = _fit_1d(sample.get('slot_mask', torch.ones(n_lig, device=device, dtype=const.TORCH_FLOAT)), n_lig, device=device, dtype=const.TORCH_FLOAT)

        lig_fmask.append(torch.cat([fragment_mask, torch.zeros(max_lig - n_lig, device=device, dtype=fragment_mask.dtype)]))
        lig_linkmask.append(torch.cat([linker_mask, torch.zeros(max_lig - n_lig, device=device, dtype=linker_mask.dtype)]))
        lig_atommask.append(torch.cat([atom_mask, torch.zeros(max_lig - n_lig, device=device, dtype=atom_mask.dtype)]))
        lig_slotmask.append(torch.cat([slot_mask, torch.ones(max_lig - n_lig, device=device, dtype=slot_mask.dtype)]))
        lig_charges.append(torch.cat([charges, torch.zeros(max_lig - n_lig, device=device, dtype=charges.dtype)]))
        lig_anchors.append(torch.cat([anchors, torch.zeros(max_lig - n_lig, device=device, dtype=anchors.dtype)]))
        lig_type_ids.append(torch.cat([type_ids, torch.full((max_lig - n_lig,), -1, device=device, dtype=const.TORCH_INT)]))

        if 'true_bonds' in sample:
            lig_bonds.append(_pad_edge_matrix(sample['true_bonds'], max_lig))
        if 'true_attach_src' in sample:
            lig_attach_src.append(_pad_edge_matrix(sample['true_attach_src'], max_lig, fill_value=-1))
        if 'true_attach_dst' in sample:
            lig_attach_dst.append(_pad_edge_matrix(sample['true_attach_dst'], max_lig, fill_value=-1))

        if max_pocket > 0:
            curr_frames = sample.get(
                'pocket_frames',
                torch.eye(4, device=device, dtype=sample['positions'].dtype).view(1, 4, 4),
            )
            curr_pocket_mask = sample.get(
                'pocket_mask',
                torch.ones(curr_frames.shape[0], device=curr_frames.device, dtype=const.TORCH_FLOAT),
            )
            pocket_frames.append(_pad_frames_4x4(curr_frames, max_pocket))
            pocket_masks.append(
                torch.cat([
                    curr_pocket_mask,
                    torch.zeros(max_pocket - curr_pocket_mask.shape[0], device=curr_pocket_mask.device, dtype=curr_pocket_mask.dtype),
                ])
            )
            curr_pocket_element = _fit_1d(
                sample.get(
                    'pocket_element',
                    torch.zeros(curr_frames.shape[0], device=curr_frames.device, dtype=const.TORCH_INT),
                ),
                curr_frames.shape[0],
                device=curr_frames.device,
                dtype=const.TORCH_INT,
            )
            pocket_elements.append(
                torch.cat([
                    curr_pocket_element,
                    torch.zeros(
                        max_pocket - curr_pocket_element.shape[0],
                        device=curr_pocket_element.device,
                        dtype=curr_pocket_element.dtype,
                    ),
                ])
            )
            curr_pocket_aa_type = _fit_1d(
                sample.get(
                    'pocket_aa_type',
                    torch.full(
                        (curr_frames.shape[0],),
                        20,
                        device=curr_frames.device,
                        dtype=const.TORCH_INT,
                    ),
                ),
                curr_frames.shape[0],
                device=curr_frames.device,
                dtype=const.TORCH_INT,
                fill_value=20,
            )
            pocket_aa_types.append(
                torch.cat([
                    curr_pocket_aa_type,
                    torch.full(
                        (max_pocket - curr_pocket_aa_type.shape[0],),
                        20,
                        device=curr_pocket_aa_type.device,
                        dtype=curr_pocket_aa_type.dtype,
                    ),
                ])
            )
            curr_pocket_is_backbone = _fit_1d(
                sample.get(
                    'pocket_is_backbone',
                    torch.zeros(
                        curr_frames.shape[0],
                        device=curr_frames.device,
                        dtype=torch.bool,
                    ),
                ),
                curr_frames.shape[0],
                device=curr_frames.device,
                dtype=torch.bool,
            )
            pocket_is_backbone.append(
                torch.cat([
                    curr_pocket_is_backbone,
                    torch.zeros(
                        max_pocket - curr_pocket_is_backbone.shape[0],
                        device=curr_pocket_is_backbone.device,
                        dtype=torch.bool,
                    ),
                ])
            )

    out = {
        'positions': torch.stack(lig_pos),
        'frag_rotations': torch.stack(lig_rot),
        'one_hot': torch.stack(lig_onehot),
        'fragment_type_ids': torch.stack(lig_type_ids),
        'charges': torch.stack(lig_charges),
        'anchors': torch.stack(lig_anchors),
        'fragment_mask': torch.stack(lig_fmask).unsqueeze(-1),
        'linker_mask': torch.stack(lig_linkmask).unsqueeze(-1),
        'atom_mask': torch.stack(lig_atommask).unsqueeze(-1),
        'slot_mask': torch.stack(lig_slotmask).unsqueeze(-1),
    }

    if len(lig_bonds) > 0:
        out['true_bonds'] = torch.stack(lig_bonds)
    if len(lig_attach_src) > 0:
        out['true_attach_src'] = torch.stack(lig_attach_src)
    if len(lig_attach_dst) > 0:
        out['true_attach_dst'] = torch.stack(lig_attach_dst)
    if len(pocket_frames) > 0:
        out['pocket_frames'] = torch.stack(pocket_frames)
        out['pocket_mask'] = torch.stack(pocket_masks)
        out['pocket_element'] = torch.stack(pocket_elements)
        out['pocket_aa_type'] = torch.stack(pocket_aa_types)
        out['pocket_is_backbone'] = torch.stack(pocket_is_backbone)

    metadata_keys = (
        'uuid',
        'name',
        'protein_filename',
        'ligand_filename',
        'dataset_index',
        'dataset_target_index',
        'generation_index_within_target',
    )
    for key in metadata_keys:
        if all(key in sample for sample in batch):
            out[key] = [sample[key] for sample in batch]

    for key in ('reference_box_center', 'reference_box_size'):
        if all(key in sample for sample in batch):
            out[key] = torch.stack([sample[key] for sample in batch])

    out['frag_frames'] = _build_frames_from_pos_rot(out['positions'], out['frag_rotations'])
    return out


def get_dataloader(
        dataset,
        batch_size,
        collate_fn=collate_with_fragment_edges,  # Default to rigid-fragment batching consistent with the paper.
        shuffle=False,
        num_workers=0,  # Default to 0 to avoid EOFError and encoding errors from multiprocess LMDB reads.
        pin_memory=False  # Default to False to avoid the 'Pin memory thread exited' error.
):

    num_workers = 0
    pin_memory = False

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,

        persistent_workers=False,  # Single-process loading does not support persistent_workers.
        prefetch_factor=None
    )
