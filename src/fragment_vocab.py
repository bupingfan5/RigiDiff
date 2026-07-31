import os
import pickle
from typing import Dict, List, Optional, Tuple

import lmdb
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import BRICS, Descriptors

from src import const


def _bondtype_from_int(value: int):
    value = int(value)
    if value == 1:
        return Chem.BondType.SINGLE
    if value == 2:
        return Chem.BondType.DOUBLE
    if value == 3:
        return Chem.BondType.TRIPLE
    if value == 4:
        return Chem.BondType.AROMATIC
    return Chem.BondType.SINGLE


def _bondtype_to_int(bond_type) -> int:
    if bond_type == Chem.BondType.SINGLE:
        return 1
    if bond_type == Chem.BondType.DOUBLE:
        return 2
    if bond_type == Chem.BondType.TRIPLE:
        return 3
    if bond_type == Chem.BondType.AROMATIC:
        return 4
    return 0


def _resolve_lmdb_path(data_path: str, prefix: str) -> str:
    prefix_lower = prefix.lower()
    prefixes = [prefix]
    if "." in prefix:
        prefixes.append(prefix.split(".")[0])
    if "_split" in prefix:
        prefixes.append(prefix.replace("_split", ""))

    candidates: List[str] = []
    for current_prefix in prefixes:
        candidates.append(os.path.join(data_path, f"{current_prefix}.lmdb"))

    if "pdbbind" in prefix_lower:
        candidates.append(os.path.join(data_path, "PDBbind_v2020_pocket10_processed_final.lmdb"))
    if prefix_lower == "crossdocked_pocket10":
        candidates = [os.path.join(data_path, "crossdocked_pocket10.lmdb")]
    elif "crossdock" in prefix_lower or "moad" in prefix_lower:
        candidates.append(os.path.join(data_path, "crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb"))

    seen = set()
    deduped_candidates = []
    for candidate in candidates:
        if candidate not in seen:
            deduped_candidates.append(candidate)
            seen.add(candidate)

    for candidate in deduped_candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(f"Unable to resolve LMDB for prefix '{prefix}'. Tried: {deduped_candidates}")


def get_fragment_vocab_path(data_path: str, prefix: str) -> str:
    lmdb_path = _resolve_lmdb_path(data_path, prefix)
    return lmdb_path[:-5] + "_fragment_vocab.pt"


def _build_ligand_mol_from_sample(sample) -> Optional[Chem.Mol]:
    elems = np.asarray(sample["ligand_element"])
    pos = np.asarray(sample["ligand_pos"], dtype=np.float32)
    bond_index = np.asarray(sample["ligand_bond_index"])
    bond_type = np.asarray(sample["ligand_bond_type"])

    bonds = bond_index.T if bond_index.ndim == 2 and bond_index.shape[0] == 2 else bond_index

    rw_mol = Chem.RWMol()
    for atomic_num in elems:
        rw_mol.AddAtom(Chem.Atom(int(atomic_num)))

    for i, (begin_idx, end_idx) in enumerate(bonds):
        begin_idx = int(begin_idx)
        end_idx = int(end_idx)
        if begin_idx == end_idx:
            continue
        if rw_mol.GetBondBetweenAtoms(begin_idx, end_idx) is not None:
            continue
        rw_mol.AddBond(begin_idx, end_idx, _bondtype_from_int(bond_type[i]))

    mol = rw_mol.GetMol()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for atom_idx, (x, y, z) in enumerate(pos):
        conf.SetAtomPosition(atom_idx, (float(x), float(y), float(z)))
    mol.AddConformer(conf, assignId=True)

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        try:
            mol.UpdatePropertyCache(strict=False)
        except Exception:
            return None

    try:
        Chem.GetSymmSSSR(mol)
    except Exception:
        return None
    return mol


def _copy_submol_with_conformer(parent_mol: Chem.Mol, atom_indices: Tuple[int, ...]):
    conf = parent_mol.GetConformer()
    atom_indices = tuple(int(idx) for idx in atom_indices)
    atom_set = set(atom_indices)

    rw_mol = Chem.RWMol()
    orig_to_local = {}
    for orig_idx in atom_indices:
        atom = Chem.Atom(parent_mol.GetAtomWithIdx(orig_idx))
        atom.SetNoImplicit(True)
        orig_to_local[orig_idx] = rw_mol.AddAtom(atom)

    for orig_idx in atom_indices:
        atom = parent_mol.GetAtomWithIdx(orig_idx)
        for bond in atom.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            if begin_idx in atom_set and end_idx in atom_set:
                local_begin = orig_to_local[begin_idx]
                local_end = orig_to_local[end_idx]
                if rw_mol.GetBondBetweenAtoms(local_begin, local_end) is None:
                    rw_mol.AddBond(local_begin, local_end, bond.GetBondType())

    fragment_mol = rw_mol.GetMol()
    frag_conf = Chem.Conformer(fragment_mol.GetNumAtoms())
    coords = []
    for orig_idx in atom_indices:
        pos = conf.GetAtomPosition(orig_idx)
        coords.append([float(pos.x), float(pos.y), float(pos.z)])
        frag_conf.SetAtomPosition(orig_to_local[orig_idx], pos)
    fragment_mol.AddConformer(frag_conf, assignId=True)
    return fragment_mol, np.asarray(coords, dtype=np.float32), orig_to_local


def _canonicalize_fragment(fragment_mol: Chem.Mol, coords: np.ndarray, attachment_local: List[int]):
    n_atoms = fragment_mol.GetNumAtoms()
    if n_atoms == 0:
        return fragment_mol, coords, attachment_local, {}

    ranks = list(Chem.CanonicalRankAtoms(fragment_mol, breakTies=True))
    order = sorted(range(n_atoms), key=lambda idx: (ranks[idx], fragment_mol.GetAtomWithIdx(idx).GetAtomicNum(), idx))
    inverse_order = {old_idx: new_idx for new_idx, old_idx in enumerate(order)}

    canonical_mol = Chem.RenumberAtoms(fragment_mol, order)
    canonical_coords = coords[order]
    canonical_attachment = sorted({inverse_order[idx] for idx in attachment_local})
    return canonical_mol, canonical_coords, canonical_attachment, inverse_order


def _rigid_frame_from_coords(coords: np.ndarray):
    center = coords.mean(axis=0, keepdims=True)
    centered = coords - center

    if coords.shape[0] < 3:
        rotation = np.eye(3, dtype=np.float32)
    else:
        covariance = centered.T @ centered
        U, _, _ = np.linalg.svd(covariance)
        rotation = U.astype(np.float32)
        for col in range(3):
            pivot = np.argmax(np.abs(rotation[:, col]))
            if rotation[pivot, col] < 0:
                rotation[:, col] *= -1.0
        if np.linalg.det(rotation) < 0:
            rotation[:, -1] *= -1.0

    local_coords = centered @ rotation
    return center.squeeze(0).astype(np.float32), rotation.astype(np.float32), local_coords.astype(np.float32)


def fit_fixed_template_pose(template_coords: np.ndarray, target_coords: np.ndarray):
    """Fit one vocabulary template to a fragment instance with a proper Kabsch rotation."""
    template = np.asarray(template_coords, dtype=np.float64)
    target = np.asarray(target_coords, dtype=np.float64)
    if template.shape != target.shape or template.ndim != 2 or template.shape[1] != 3:
        raise ValueError(
            f"Template/target coordinate shape mismatch: template={template.shape}, target={target.shape}"
        )
    if template.shape[0] == 0:
        raise ValueError("Cannot fit an empty fragment template")
    if not np.isfinite(template).all() or not np.isfinite(target).all():
        raise ValueError("Fragment template fitting received non-finite coordinates")

    template_center = template.mean(axis=0)
    target_center = target.mean(axis=0)
    template_centered = template - template_center
    target_centered = target - target_center

    if template.shape[0] == 1:
        row_rotation = np.eye(3, dtype=np.float64)
    else:
        covariance = template_centered.T @ target_centered
        u, _, vt = np.linalg.svd(covariance)
        row_rotation = u @ vt
        if np.linalg.det(row_rotation) < 0.0:
            u[:, -1] *= -1.0
            row_rotation = u @ vt

    # Decoder convention: global_column = rotation @ local_column + translation.
    rotation = row_rotation.T
    translation = target_center - rotation @ template_center
    reconstructed = template @ rotation.T + translation
    rmsd = float(np.sqrt(np.mean(np.sum((reconstructed - target) ** 2, axis=1))))
    return (
        translation.astype(np.float32),
        rotation.astype(np.float32),
        rmsd,
    )


def _fragment_instances_from_mol(mol: Chem.Mol):
    brics_bonds = list(BRICS.FindBRICSBonds(mol))
    if len(brics_bonds) == 0:
        brics_bonds = []

    broken_pairs = []
    bond_ids = []
    for (atom_a, atom_b), _labels in brics_bonds:
        bond = mol.GetBondBetweenAtoms(int(atom_a), int(atom_b))
        if bond is None:
            continue
        bond_ids.append(bond.GetIdx())
        broken_pairs.append((int(atom_a), int(atom_b), _bondtype_to_int(bond.GetBondType())))

    fragmented_mol = Chem.FragmentOnBonds(mol, bond_ids, addDummies=False) if bond_ids else mol
    fragments = Chem.GetMolFrags(fragmented_mol, asMols=False, sanitizeFrags=False)

    atom_to_fragment = {}
    for frag_idx, atom_indices in enumerate(fragments):
        for atom_idx in atom_indices:
            atom_to_fragment[int(atom_idx)] = frag_idx

    fragment_instances = []
    for atom_indices in fragments:
        atom_set = set(int(idx) for idx in atom_indices)
        heavy_atom_indices = [
            int(idx)
            for idx in atom_indices
            if mol.GetAtomWithIdx(int(idx)).GetAtomicNum() != 1
        ]
        if not heavy_atom_indices:
            continue
        attachment_orig = []
        for atom_a, atom_b, _bond_type in broken_pairs:
            if atom_a in atom_set and atom_b not in atom_set:
                attachment_orig.append(atom_a)
            elif atom_b in atom_set and atom_a not in atom_set:
                attachment_orig.append(atom_b)

        fragment_mol, coords, orig_to_local = _copy_submol_with_conformer(
            mol,
            heavy_atom_indices,
        )
        attachment_local = [orig_to_local[idx] for idx in attachment_orig if idx in orig_to_local]
        fragment_mol, coords, attachment_local, renumber_map = _canonicalize_fragment(
            fragment_mol,
            coords,
            attachment_local,
        )

        orig_to_local_final = {}
        for orig_idx, local_idx in orig_to_local.items():
            orig_to_local_final[orig_idx] = renumber_map[local_idx]

        center, rotation, local_coords = _rigid_frame_from_coords(coords)
        fragment_mol_no_h = Chem.RemoveHs(Chem.Mol(fragment_mol), sanitize=False)
        smiles = Chem.MolToSmiles(fragment_mol_no_h, canonical=True, isomericSmiles=True)

        atom_symbols = [atom.GetSymbol() for atom in fragment_mol.GetAtoms()]
        atom_names = [f"A{atom_idx:02d}" for atom_idx in range(fragment_mol.GetNumAtoms())]
        bonds = [
            (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), _bondtype_to_int(bond.GetBondType()))
            for bond in fragment_mol.GetBonds()
        ]

        fragment_instances.append(
            {
                "smiles": smiles,
                "center": center,
                "rotation": rotation,
                "local_coords": local_coords,
                "global_coords": coords.astype(np.float32),
                "atom_symbols": atom_symbols,
                "atom_names": atom_names,
                "bonds": bonds,
                "attachment_points": sorted(set(attachment_local)),
                "orig_to_local": orig_to_local_final,
                "mol_weight": float(Descriptors.MolWt(fragment_mol_no_h)),
            }
        )

    fragment_edges = []
    attachment_edges = []
    for atom_a, atom_b, bond_type in broken_pairs:
        frag_a = atom_to_fragment.get(atom_a)
        frag_b = atom_to_fragment.get(atom_b)
        if frag_a is None or frag_b is None or frag_a == frag_b:
            continue

        local_a = fragment_instances[frag_a]["orig_to_local"][atom_a]
        local_b = fragment_instances[frag_b]["orig_to_local"][atom_b]
        attach_points_a = fragment_instances[frag_a]["attachment_points"]
        attach_points_b = fragment_instances[frag_b]["attachment_points"]
        attach_idx_a = attach_points_a.index(local_a) if local_a in attach_points_a else -1
        attach_idx_b = attach_points_b.index(local_b) if local_b in attach_points_b else -1

        fragment_edges.append((frag_a, frag_b, bond_type))
        fragment_edges.append((frag_b, frag_a, bond_type))
        attachment_edges.append((frag_a, frag_b, attach_idx_a, attach_idx_b, bond_type))
        attachment_edges.append((frag_b, frag_a, attach_idx_b, attach_idx_a, bond_type))

    fragment_edges = list(set(fragment_edges))
    attachment_edges = list(set(attachment_edges))
    return fragment_instances, fragment_edges, attachment_edges


def _apply_vocabulary_to_const(entries: List[Dict]):
    names = [entry.get("smiles", f"frag_{idx}") for idx, entry in enumerate(entries)]
    const.NUM2AA[:] = names
    const.AA2NUM.clear()
    const.AA2NUM.update({name: idx for idx, name in enumerate(names)})
    const.GEOM_NUMBER_OF_FRAGMENT_TYPES = len(names)

    ideal_coords = []
    aa_bonds = []
    aa_name_to_idx = []
    frag_bond_types = []
    frag_attachment_points = []
    frag_atom_symbols = []

    for entry in entries:
        atom_names = entry["atom_names"]
        atom_symbols = entry["atom_symbols"]
        coords = entry["coords"]
        attachment_points = set(entry.get("attachment_points", []))
        bonds = entry["bonds"]

        ideal_coords.append(
            [
                [atom_names[idx], int(idx in attachment_points), tuple(float(v) for v in coords[idx])]
                for idx in range(len(atom_names))
            ]
        )
        aa_bonds.append(tuple((atom_names[begin_idx], atom_names[end_idx]) for begin_idx, end_idx, _ in bonds))
        frag_bond_types.append(tuple(int(bond_type) for _, _, bond_type in bonds))
        aa_name_to_idx.append({atom_name: idx for idx, atom_name in enumerate(atom_names)})
        frag_attachment_points.append(list(sorted(attachment_points)))
        frag_atom_symbols.append(list(atom_symbols))

    const.IDEAL_COORDS[:] = ideal_coords
    const.AA_BONDS[:] = aa_bonds
    const.AA_NAME2IDX[:] = aa_name_to_idx
    const.FRAGMENT_BOND_TYPES[:] = frag_bond_types
    const.FRAGMENT_ATTACHMENT_POINTS[:] = frag_attachment_points
    const.FRAGMENT_ATOM_SYMBOLS[:] = frag_atom_symbols
    const.AA_TIP_ATOM[:] = [
        (ideal_coords[idx][attachment_points[0]][0] if attachment_points else ideal_coords[idx][0][0])
        for idx, attachment_points in enumerate(const.FRAGMENT_ATTACHMENT_POINTS)
    ]


def _normalize_lmdb_path(path: str) -> str:
    normalized = str(path).replace('\\', '/')
    # Vocabulary files may be created under WSL and consumed by the Windows
    # trainer.  Treat /mnt/e/foo and E:/foo as the same provenance path.
    if os.name == 'nt' and normalized.lower().startswith('/mnt/') and len(normalized) > 6:
        drive = normalized[5]
        normalized = f'{drive}:{normalized[6:]}'
    return os.path.normcase(os.path.abspath(normalized))


def load_fragment_vocabulary(vocab_path: str, expected_lmdb_path: Optional[str] = None):
    payload = torch.load(vocab_path, map_location="cpu",weights_only=False)
    source_lmdb = payload.get("source_lmdb")
    if not source_lmdb:
        raise ValueError(f"Fragment vocabulary is missing 'source_lmdb': {vocab_path}")
    if expected_lmdb_path is not None:
        expected_path = _normalize_lmdb_path(expected_lmdb_path)
        source_path = _normalize_lmdb_path(source_lmdb)
        if source_path != expected_path:
            raise ValueError(
                "Fragment vocabulary LMDB mismatch. "
                f"vocab={vocab_path}, source_lmdb={source_lmdb}, expected_lmdb={expected_lmdb_path}"
            )
    entries = payload["entries"]
    _apply_vocabulary_to_const(entries)
    return payload


def build_fragment_vocabulary(data_path: str, prefix: str, max_samples: Optional[int] = None):
    lmdb_path = _resolve_lmdb_path(data_path, prefix)
    vocab_path = lmdb_path[:-5] + "_fragment_vocab.pt"

    env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False, readahead=False, meminit=False, max_readers=256)
    entries: List[Dict] = []
    smiles_to_idx: Dict[str, int] = {}

    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for sample_idx, (_key, value) in enumerate(cursor):
            if max_samples is not None and sample_idx >= max_samples:
                break
            try:
                sample = pickle.loads(value)
                mol = _build_ligand_mol_from_sample(sample)
                if mol is None:
                    continue

                fragment_instances, _, _ = _fragment_instances_from_mol(mol)
                for fragment_info in fragment_instances:
                    smiles = fragment_info["smiles"]
                    if smiles in smiles_to_idx:
                        continue

                    entry = {
                        "name": f"FRAG_{len(entries):04d}",
                        "smiles": smiles,
                        "atom_names": list(fragment_info["atom_names"]),
                        "atom_symbols": list(fragment_info["atom_symbols"]),
                        "coords": fragment_info["local_coords"].astype(np.float32),
                        "bonds": list(fragment_info["bonds"]),
                        "attachment_points": list(fragment_info["attachment_points"]),
                        "mol_weight": float(fragment_info["mol_weight"]),
                    }
                    smiles_to_idx[smiles] = len(entries)
                    entries.append(entry)
            except Exception:
                continue

    env.close()
    payload = {
        "source_lmdb": _normalize_lmdb_path(lmdb_path),
        "entries": entries,
        "smiles_to_idx": smiles_to_idx,
    }
    torch.save(payload, vocab_path)
    _apply_vocabulary_to_const(entries)
    return payload


def ensure_fragment_vocabulary(data_path: str, prefix: str, max_samples: Optional[int] = None):
    lmdb_path = _resolve_lmdb_path(data_path, prefix)
    vocab_path = get_fragment_vocab_path(data_path, prefix)
    if os.path.exists(vocab_path):
        return load_fragment_vocabulary(vocab_path, expected_lmdb_path=lmdb_path)
    return build_fragment_vocabulary(data_path, prefix, max_samples=max_samples)


def fragment_graph_to_rigid_tensors(mol: Chem.Mol, fragment_vocab: Dict):
    fragment_instances, fragment_edges, attachment_edges = _fragment_instances_from_mol(mol)
    smiles_to_idx = fragment_vocab["smiles_to_idx"]

    rigid_fragments = []
    for fragment_info in fragment_instances:
        smiles = fragment_info["smiles"]
        if smiles not in smiles_to_idx:
            raise KeyError(f"Fragment '{smiles}' is missing from the loaded fragment vocabulary.")

        type_idx = int(smiles_to_idx[smiles])
        vocabulary_entry = fragment_vocab["entries"][type_idx]
        center, rotation, frame_rmsd = fit_fixed_template_pose(
            vocabulary_entry["coords"],
            fragment_info["global_coords"],
        )

        rigid_fragments.append(
            {
                "type_idx": type_idx,
                "center": torch.tensor(center, dtype=const.TORCH_FLOAT),
                "rotation": torch.tensor(rotation, dtype=const.TORCH_FLOAT),
                "frame_rmsd": frame_rmsd,
                "attachment_points": list(fragment_info["attachment_points"]),
                "smiles": smiles,
            }
        )

    return rigid_fragments, fragment_edges, attachment_edges
