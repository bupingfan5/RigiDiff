import argparse
import csv
import os
import pickle
from collections import Counter
from pathlib import Path

import lmdb
import numpy as np
import torch
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.rdchem import BondType
from tqdm import tqdm


ATOM_FAMILIES = [
    "Acceptor",
    "Donor",
    "Aromatic",
    "Hydrophobe",
    "LumpedHydrophobe",
    "NegIonizable",
    "PosIonizable",
    "ZnBinder",
]
ATOM_FAMILIES_ID = {name: idx for idx, name in enumerate(ATOM_FAMILIES)}

BOND_TYPES = {
    BondType.UNSPECIFIED: 0,
    BondType.SINGLE: 1,
    BondType.DOUBLE: 2,
    BondType.TRIPLE: 3,
    BondType.AROMATIC: 4,
}

AA_NAME_SYM = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}
AA_NAME_NUMBER = {name: idx for idx, name in enumerate(AA_NAME_SYM)}
BACKBONE_NAMES = {"CA", "C", "N", "O"}


def torchify_dict(data):
    output = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            output[key] = torch.from_numpy(value)
        else:
            output[key] = value
    return output


def parse_pocket_pdb(path):
    ptable = Chem.GetPeriodicTable()
    title = None
    element = []
    pos = []
    atom_name = []
    is_backbone = []
    atom_to_aa_type = []

    with open(path, "r", errors="ignore") as handle:
        for line in handle:
            record = line[0:6].strip()
            if record == "HEADER":
                title = line[10:].strip().lower() or "pocket"
                continue
            if record == "ENDMDL":
                break
            if record != "ATOM":
                continue

            name = line[12:16].strip()
            residue_name = line[17:20].strip()
            element_symbol = line[76:78].strip().capitalize()
            if not element_symbol:
                element_symbol = line[13:14].strip().capitalize()
            if residue_name not in AA_NAME_NUMBER:
                raise ValueError(f"Unknown residue name: {residue_name}")

            atomic_number = ptable.GetAtomicNumber(element_symbol)
            if atomic_number <= 0:
                raise ValueError(f"Unknown atom element: {element_symbol}")

            element.append(atomic_number)
            pos.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            atom_name.append(name)
            is_backbone.append(name in BACKBONE_NAMES)
            atom_to_aa_type.append(AA_NAME_NUMBER[residue_name])

    if not element:
        raise ValueError("No ATOM records found in pocket PDB")

    return {
        "element": np.asarray(element, dtype=np.int64),
        "molecule_name": title or "pocket",
        "pos": np.asarray(pos, dtype=np.float32),
        "is_backbone": np.asarray(is_backbone, dtype=np.bool_),
        "atom_name": atom_name,
        "atom_to_aa_type": np.asarray(atom_to_aa_type, dtype=np.int64),
    }


def parse_ligand_sdf(path, feature_factory):
    mol = Chem.MolFromMolFile(str(path), sanitize=False, removeHs=False)
    if mol is None:
        raise ValueError("RDKit failed to read ligand")
    repair_method = "standard_sanitize"
    try:
        Chem.SanitizeMol(mol)
        mol = Chem.RemoveHs(mol)
    except Exception:
        # Some official CrossDock docked SDFs contain chemically inconsistent
        # bond orders. Preserve their atom graph and coordinates explicitly
        # instead of substituting another target.
        mol = Chem.RemoveHs(mol, sanitize=False)
        mol.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(mol)
        repair_method = "relaxed_property_cache_for_official_sdf"
    if mol.GetNumConformers() == 0:
        raise ValueError("Ligand has no conformer")

    num_atoms = mol.GetNumAtoms()
    atom_feature = np.zeros((num_atoms, len(ATOM_FAMILIES)), dtype=np.int64)
    for feature in feature_factory.GetFeaturesForMol(mol):
        family_id = ATOM_FAMILIES_ID.get(feature.GetFamily())
        if family_id is not None:
            atom_feature[list(feature.GetAtomIds()), family_id] = 1

    hybridization = []
    element = []
    ptable = Chem.GetPeriodicTable()
    pos = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float32)
    center_of_mass = np.zeros(3, dtype=np.float32)
    total_mass = 0.0
    for atom_idx in range(num_atoms):
        atom = mol.GetAtomWithIdx(atom_idx)
        hybridization.append((atom_idx, str(atom.GetHybridization())))
        atomic_number = atom.GetAtomicNum()
        element.append(atomic_number)
        atom_weight = ptable.GetAtomicWeight(atomic_number)
        center_of_mass += pos[atom_idx] * atom_weight
        total_mass += atom_weight
    if total_mass <= 0:
        raise ValueError("Ligand has invalid atomic mass")
    center_of_mass /= total_mass
    hybridization = [value for _, value in sorted(hybridization)]

    row, col, bond_type = [], [], []
    for bond in mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        encoded_type = BOND_TYPES.get(bond.GetBondType(), 0)
        row.extend([start, end])
        col.extend([end, start])
        bond_type.extend([encoded_type, encoded_type])

    bond_index = np.asarray([row, col], dtype=np.int64)
    bond_type = np.asarray(bond_type, dtype=np.int64)
    if bond_index.shape[1] > 0:
        order = (bond_index[0] * num_atoms + bond_index[1]).argsort()
        bond_index = bond_index[:, order]
        bond_type = bond_type[order]

    nbh_list = {idx: [] for idx in range(num_atoms)}
    for src, dst in zip(bond_index[0], bond_index[1]):
        nbh_list[int(src)].append(int(dst))

    return {
        "smiles": Chem.MolToSmiles(mol),
        "element": np.asarray(element, dtype=np.int64),
        "pos": pos,
        "bond_index": bond_index,
        "bond_type": bond_type,
        "center_of_mass": center_of_mass.astype(np.float32),
        "atom_feature": atom_feature,
        "hybridization": hybridization,
        "nbh_list": nbh_list,
        "repair_method": repair_method,
    }


def build_sample(raw_dir, pocket_rel, ligand_rel, feature_factory):
    pocket_path = raw_dir / pocket_rel
    ligand_path = raw_dir / ligand_rel
    if not pocket_path.exists():
        raise FileNotFoundError(f"Pocket file not found: {pocket_rel}")
    if not ligand_path.exists():
        raise FileNotFoundError(f"Ligand file not found: {ligand_rel}")

    protein = torchify_dict(parse_pocket_pdb(pocket_path))
    ligand = torchify_dict(parse_ligand_sdf(ligand_path, feature_factory))

    return {
        "protein_element": protein["element"],
        "protein_molecule_name": protein["molecule_name"],
        "protein_pos": protein["pos"],
        "protein_is_backbone": protein["is_backbone"],
        "protein_atom_name": protein["atom_name"],
        "protein_atom_to_aa_type": protein["atom_to_aa_type"],
        "ligand_smiles": ligand["smiles"],
        "ligand_element": ligand["element"],
        "ligand_pos": ligand["pos"],
        "ligand_bond_index": ligand["bond_index"],
        "ligand_bond_type": ligand["bond_type"],
        "ligand_center_of_mass": ligand["center_of_mass"],
        "ligand_atom_feature": ligand["atom_feature"],
        "ligand_hybridization": ligand["hybridization"],
        "ligand_nbh_list": ligand["nbh_list"],
        "ligand_parse_repair": ligand["repair_method"],
        "protein_filename": pocket_rel,
        "ligand_filename": ligand_rel,
    }


def remove_lmdb_file(path):
    for candidate in (path, Path(str(path) + "-lock")):
        if candidate.exists():
            candidate.unlink()


def first_lmdb_sample(path):
    env = lmdb.open(
        str(path),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=16,
    )
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        item = next(iter(cursor), None)
        if item is None:
            env.close()
            return None, None
        key, value = item
        sample = pickle.loads(value)
    env.close()
    return key, sample


def summarize_value(value):
    if torch.is_tensor(value):
        return f"Tensor(dtype={value.dtype}, shape={tuple(value.shape)})"
    if isinstance(value, list):
        return f"list(len={len(value)})"
    if isinstance(value, dict):
        return f"dict(len={len(value)})"
    return type(value).__name__


def print_schema(path, reference=None):
    key, sample = first_lmdb_sample(path)
    if sample is None:
        print(f"Schema check: {path} is empty")
        return

    print(f"Schema check: {path}")
    print(f"  first key: {key!r}")
    print(f"  fields ({len(sample)}): {sorted(sample)}")
    for field in sorted(sample):
        print(f"  {field}: {summarize_value(sample[field])}")

    if reference and reference.exists():
        _, reference_sample = first_lmdb_sample(reference)
        if reference_sample is not None:
            missing = sorted(set(reference_sample) - set(sample))
            extra = sorted(set(sample) - set(reference_sample))
            print(f"Reference schema: {reference}")
            print(f"  missing fields vs reference: {missing}")
            print(f"  extra fields vs reference: {extra}")


def write_manifest(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "lmdb_key", "status", "error_type", "error_message", "pocket_filename",
        "ligand_filename", "repair_method",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def required_split_keys(split_file, split_name):
    if split_file is None:
        return []
    payload = torch.load(split_file, map_location="cpu")
    if split_name not in payload:
        raise KeyError(f"Split {split_name!r} is absent from {split_file}")
    values = payload[split_name]
    if torch.is_tensor(values):
        values = values.detach().cpu().tolist()
    return [int(value) for value in values]


def convert(raw_dir, index_file, output, map_size_gb, limit, overwrite, commit_interval, reference,
            manifest_csv, required_split_file=None, required_split="test"):
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output} already exists. Pass --overwrite to replace it.")
        remove_lmdb_file(output)

    raw_dir = raw_dir.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(index_file, "rb") as handle:
        index = pickle.load(handle)

    fdef_name = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)

    env = lmdb.open(
        str(output),
        map_size=int(map_size_gb * 1024**3),
        create=True,
        subdir=False,
        readonly=False,
        lock=True,
        readahead=False,
        meminit=False,
        map_async=True,
    )

    total = len(index) if limit is None else min(limit, len(index))
    written = 0
    skipped = 0
    errors = Counter()
    manifest_rows = []
    txn = env.begin(write=True)
    try:
        for idx, item in enumerate(tqdm(index[:total], total=total, desc="CrossDock LMDB")):
            if item is None or len(item) < 2 or item[0] is None or item[1] is None:
                skipped += 1
                errors["missing_index_entry"] += 1
                manifest_rows.append({
                    "lmdb_key": idx,
                    "status": "failed",
                    "error_type": "missing_index_entry",
                    "error_message": "Official index entry is missing; restore this exact key from CrossDocked2020.",
                    "pocket_filename": item[0] if item is not None and len(item) > 0 and item[0] else "",
                    "ligand_filename": item[1] if item is not None and len(item) > 1 and item[1] else "",
                    "repair_method": "requires_official_source_restore",
                })
                continue

            pocket_rel, ligand_rel = item[0], item[1]
            try:
                sample = build_sample(raw_dir, pocket_rel, ligand_rel, feature_factory)
                txn.put(str(idx).encode(), pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL))
                written += 1
                manifest_rows.append({
                    "lmdb_key": idx,
                    "status": "success",
                    "error_type": "",
                    "error_message": "",
                    "pocket_filename": pocket_rel,
                    "ligand_filename": ligand_rel,
                    "repair_method": sample.get("ligand_parse_repair", "standard_sanitize"),
                })
            except Exception as exc:
                skipped += 1
                errors[type(exc).__name__] += 1
                manifest_rows.append({
                    "lmdb_key": idx,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "pocket_filename": pocket_rel,
                    "ligand_filename": ligand_rel,
                    "repair_method": "none",
                })
                continue

            if written > 0 and written % commit_interval == 0:
                txn.commit()
                txn = env.begin(write=True)
        txn.commit()
    except Exception:
        txn.abort()
        raise
    finally:
        env.sync()
        env.close()

    print(f"Done: processed={total}, written={written}, skipped={skipped}, output={output}")
    write_manifest(manifest_rows, manifest_csv)
    print(f"Build manifest: {manifest_csv}")
    if errors:
        print("Skip reasons:")
        for name, count in errors.most_common():
            print(f"  {name}: {count}")
    print_schema(output, reference=reference)

    required_keys = required_split_keys(required_split_file, required_split)
    if required_keys:
        env = lmdb.open(str(output), subdir=False, readonly=True, lock=False, readahead=False)
        with env.begin(write=False) as txn:
            missing_keys = [key for key in required_keys if txn.get(str(key).encode()) is None]
        env.close()
        if missing_keys:
            raise RuntimeError(
                f"LMDB is not complete for split {required_split!r}: missing {len(missing_keys)} keys: "
                f"{missing_keys}. Do not substitute other pockets."
            )
        print(f"Required split audit passed: {required_split} has {len(required_keys)}/{len(required_keys)} keys.")


def main():
    parser = argparse.ArgumentParser(description="Convert raw CrossDock pocket10 files to TargetDiff-style LMDB.")
    parser.add_argument("--raw_dir", default="data/crossdocked_pocket10", type=Path)
    parser.add_argument("--index_file", default=None, type=Path)
    parser.add_argument("--output", default="data/crossdocked_pocket10_v2.lmdb", type=Path)
    parser.add_argument("--manifest_csv", default="data/lmdb_build_manifest.csv", type=Path)
    parser.add_argument("--required_split_file", default="data/crossdocked_pocket10_pose_split.pt", type=Path)
    parser.add_argument("--required_split", default="test")
    parser.add_argument("--map_size_gb", default=64, type=float)
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--commit_interval", default=1000, type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--reference",
        default="data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb",
        type=Path,
    )
    args = parser.parse_args()

    index_file = args.index_file or (args.raw_dir / "index.pkl")
    convert(
        raw_dir=args.raw_dir,
        index_file=index_file,
        output=args.output,
        map_size_gb=args.map_size_gb,
        limit=args.limit,
        overwrite=args.overwrite,
        commit_interval=args.commit_interval,
        reference=args.reference,
        manifest_csv=args.manifest_csv,
        required_split_file=args.required_split_file,
        required_split=args.required_split,
    )


if __name__ == "__main__":
    main()
