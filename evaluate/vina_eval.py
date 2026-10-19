import argparse
import csv
import math
import os
import pickle
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import lmdb
import numpy as np
import torch
from rdkit import Chem, RDLogger
from tqdm import tqdm


RDLogger.DisableLog("rdApp.*")

TARGET_RE = re.compile(r"target_(\d+)(?:_|$)")
SAMPLE_RE = re.compile(r"_sample_(\d+)\.sdf$")
FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
AFFINITY_RE = re.compile(rf"Affinity:\s*({FLOAT_RE})")
SCORE_ONLY_RE = re.compile(rf"Estimated Free Energy of Binding\s*:\s*({FLOAT_RE})")
SUPPORTED_MODES = ("score", "min", "dock")
POSE_MODES = ("score", "min")


def resolve_path(path):
    if path is None:
        return None
    path = Path(path)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def find_executable(names):
    for name in names:
        path = shutil.which(name)
        if path:
            return path
    return None


def run_command(cmd, log_path=None):
    proc = subprocess.run(
        [str(x) for x in cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout.strip() or f"command returned {proc.returncode}")
    return proc.stdout


def parse_modes(value):
    modes = [m.strip().lower() for m in value.split(",") if m.strip()]
    bad = [m for m in modes if m not in SUPPORTED_MODES]
    if bad:
        raise ValueError(f"Unsupported modes: {bad}. Supported: {SUPPORTED_MODES}")
    return modes


def load_generated_rows(metrics_csv, sdf_dir, source_samples_per_target, key_to_target=None,
                        max_targets=None, max_per_target=None):
    candidates = []
    seen_paths = set()

    metrics_path = resolve_path(metrics_csv) if metrics_csv else None
    if metrics_path and metrics_path.exists():
        with metrics_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                sdf_path = row.get("sdf_path") or row.get("selected_sdf_path") or row.get("file")
                if not sdf_path:
                    continue
                path = resolve_path(sdf_path)
                if not path.exists():
                    continue
                key = str(path.resolve())
                if key in seen_paths:
                    continue
                seen_paths.add(key)
                row["sdf_path"] = key
                candidates.append(row)
    else:
        sdf_root = resolve_path(sdf_dir)
        for path in sorted(sdf_root.glob("*.sdf")):
            key = str(path.resolve())
            seen_paths.add(key)
            candidates.append({"sdf_path": key})

    grouped = defaultdict(list)
    for row in candidates:
        try:
            target_id = eval_target_id_from_row(row, source_samples_per_target, key_to_target=key_to_target)
        except ValueError:
            continue
        row["_eval_target_id"] = target_id
        grouped[target_id].append(row)

    selected = []
    for target_id in sorted(grouped):
        if max_targets is not None and target_id >= max_targets:
            continue
        rows = sorted(grouped[target_id], key=lambda r: Path(r["sdf_path"]).name)
        if max_per_target is not None:
            rows = rows[:max_per_target]
        selected.extend(rows)
    return selected


def target_id_from_sdf(path):
    match = TARGET_RE.search(Path(path).name)
    if not match:
        raise ValueError(f"Cannot parse target id from filename: {path}")
    return int(match.group(1))


def sample_idx_from_sdf(path):
    match = SAMPLE_RE.search(Path(path).name)
    if not match:
        raise ValueError(f"Cannot parse sample index from filename: {path}")
    return int(match.group(1))


def eval_target_id_from_row(row, source_samples_per_target, key_to_target=None):
    lmdb_key = row.get("lmdb_key")
    if lmdb_key not in ("", None) and key_to_target is not None:
        key_text = str(lmdb_key)
        if key_text not in key_to_target:
            raise ValueError(f"lmdb_key {key_text} is not part of the requested split")
        return key_to_target[key_text]
    dataset_target_index = row.get("dataset_target_index")
    if dataset_target_index not in ("", None):
        return int(dataset_target_index)
    target_name = row.get("target_name")
    if target_name:
        match = TARGET_RE.search(str(target_name))
        if match:
            return int(match.group(1))
    source_idx = row.get("source_sample_idx")
    if source_idx in ("", None):
        source_idx = sample_idx_from_sdf(row["sdf_path"])
    source_idx = int(source_idx)
    if source_samples_per_target <= 0:
        raise ValueError("--source_samples_per_target must be positive")
    return source_idx // source_samples_per_target


def valid_split_keys(split_file, lmdb_path, split_name):
    split = torch.load(str(split_file), map_location="cpu")
    if split_name not in split:
        raise KeyError(f"Split '{split_name}' not found in {split_file}. Available: {list(split.keys())}")
    target_indices = split[split_name]

    env = lmdb.open(str(lmdb_path), subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
    keys = [str(int(idx)).encode() for idx in target_indices]
    missing = []
    with env.begin(write=False) as txn:
        for key in keys:
            if txn.get(key) is None:
                missing.append(key.decode())
    env.close()
    if missing:
        raise RuntimeError(
            f"LMDB is incomplete for split {split_name!r}; missing keys: {missing}. "
            "Do not compress or renumber the target list."
        )
    return keys


def load_sample(env, key):
    with env.begin(write=False) as txn:
        value = txn.get(key)
    if value is None:
        raise KeyError(key)
    return pickle.loads(value)


def load_first_mol_sanitize_false(path):
    supplier = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)
    if len(supplier) == 0 or supplier[0] is None:
        raise ValueError(f"RDKit could not read molecule: {path}")
    return supplier[0]


def sanitize_generated_sdf(sdf_path):
    mol = load_first_mol_sanitize_false(sdf_path)
    Chem.SanitizeMol(mol)
    return mol


def heavy_atom_coords_and_masses(mol):
    if mol.GetNumConformers() == 0:
        raise ValueError("molecule has no conformer")
    conf = mol.GetConformer()
    coords = []
    masses = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords.append((float(pos.x), float(pos.y), float(pos.z)))
        masses.append(float(atom.GetMass()))
    if not coords:
        raise ValueError("molecule has no heavy atoms")
    return coords, masses


def weighted_center(coords, masses):
    total = sum(masses)
    if total <= 0:
        raise ValueError("heavy-atom mass sum is zero")
    return [
        sum(coord[axis] * mass for coord, mass in zip(coords, masses)) / total
        for axis in range(3)
    ]


def reference_box_from_ligand(ligand_path, margin, min_size=None, fixed_size=None):
    mol = load_first_mol_sanitize_false(ligand_path)
    coords, masses = heavy_atom_coords_and_masses(mol)
    center = weighted_center(coords, masses)
    mins = [min(c[i] for c in coords) for i in range(3)]
    maxs = [max(c[i] for c in coords) for i in range(3)]
    if fixed_size is not None:
        size = [float(fixed_size)] * 3
    else:
        size = [max(maxs[i] - mins[i] + margin, 1.0) for i in range(3)]
        if min_size is not None:
            size = [max(s, float(min_size)) for s in size]
    return center, size


def molecule_out_of_box(mol, center, size):
    coords, _masses = heavy_atom_coords_and_masses(mol)
    half = [s / 2.0 for s in size]
    for coord in coords:
        for axis in range(3):
            if coord[axis] < center[axis] - half[axis] or coord[axis] > center[axis] + half[axis]:
                return True
    return False


def translate_mol(mol, shift):
    moved = Chem.Mol(mol)
    if moved.GetNumConformers() == 0:
        raise ValueError("molecule has no conformer")
    conf = moved.GetConformer()
    for atom_idx in range(moved.GetNumAtoms()):
        pos = conf.GetAtomPosition(atom_idx)
        conf.SetAtomPosition(
            atom_idx,
            (float(pos.x) + shift[0], float(pos.y) + shift[1], float(pos.z) + shift[2]),
        )
    return moved


def recenter_mol_to_reference(mol, reference_center):
    coords, masses = heavy_atom_coords_and_masses(mol)
    generated_center = weighted_center(coords, masses)
    shift = [float(reference_center[i] - generated_center[i]) for i in range(3)]
    return translate_mol(mol, shift), generated_center, shift


def distance(a, b):
    return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(3)))


def write_explicit_h_sdf(mol, sdf_path):
    sdf_path.parent.mkdir(parents=True, exist_ok=True)
    mol_h = Chem.AddHs(Chem.Mol(mol), addCoords=True)
    writer = Chem.SDWriter(str(sdf_path))
    try:
        writer.write(mol_h)
    finally:
        writer.close()
    if not sdf_path.exists() or sdf_path.stat().st_size == 0:
        raise RuntimeError(f"failed to write explicit-H ligand SDF: {sdf_path}")
    return sdf_path


def prepare_ligand_meeko(mol, explicit_h_sdf_path, pdbqt_path, ligand_tool, log_path=None):
    pdbqt_path.parent.mkdir(parents=True, exist_ok=True)
    if pdbqt_path.exists():
        return
    write_explicit_h_sdf(mol, explicit_h_sdf_path)
    run_command([ligand_tool, "-i", explicit_h_sdf_path, "-o", pdbqt_path], log_path=log_path)


def prepare_receptor_meeko(pdb_path, pdbqt_path, receptor_tool, default_altloc,
                           allow_bad_res=True, log_path=None):
    pdbqt_path.parent.mkdir(parents=True, exist_ok=True)
    if pdbqt_path.exists():
        return
    out_prefix = str(pdbqt_path.with_suffix(""))
    cmd = [receptor_tool, "-i", pdb_path, "-o", out_prefix, "-p"]
    if default_altloc:
        cmd.extend(["--default_altloc", default_altloc])
    if allow_bad_res:
        cmd.append("--allow_bad_res")
    output = run_command(cmd, log_path=log_path)
    generated = Path(out_prefix + ".pdbqt")
    if generated.exists() and generated != pdbqt_path:
        generated.replace(pdbqt_path)
    if not pdbqt_path.exists():
        raise RuntimeError(f"Meeko did not create receptor PDBQT: {pdbqt_path}")
    return output


def parse_vina_affinity(output):
    match = AFFINITY_RE.search(output)
    if match:
        return float(match.group(1))
    match = SCORE_ONLY_RE.search(output)
    if match:
        return float(match.group(1))
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "1":
            try:
                return float(parts[1])
            except ValueError:
                pass
    raise ValueError(f"Could not parse affinity from Vina output:\n{output}")


def common_vina_args(vina, receptor_pdbqt, ligand_pdbqt, center, size, cpu):
    return [
        vina,
        "--receptor", receptor_pdbqt,
        "--ligand", ligand_pdbqt,
        "--center_x", center[0],
        "--center_y", center[1],
        "--center_z", center[2],
        "--size_x", size[0],
        "--size_y", size[1],
        "--size_z", size[2],
        "--cpu", cpu,
    ]


def run_vina_mode(mode, vina, receptor_pdbqt, ligand_pdbqt, center, size, args, out_dir, log_dir, stem):
    cmd = common_vina_args(vina, receptor_pdbqt, ligand_pdbqt, center, size, args.cpu)
    if mode == "score":
        cmd.append("--score_only")
    elif mode == "min":
        out_path = out_dir / f"{stem}_vina_min.pdbqt"
        cmd.extend(["--local_only", "--seed", args.seed, "--out", out_path])
    elif mode == "dock":
        out_path = out_dir / f"{stem}_vina_dock.pdbqt"
        cmd.extend([
            "--exhaustiveness", args.exhaustiveness,
            "--num_modes", args.num_modes,
            "--seed", args.seed,
            "--out", out_path,
        ])
    else:
        raise ValueError(mode)

    log_path = log_dir / f"{stem}_{mode}.log"
    output = run_command(cmd, log_path=log_path)
    score = parse_vina_affinity(output)
    if not math.isfinite(score):
        raise ValueError(f"Non-finite Vina score: {score}")
    return score


def safe_float(value):
    try:
        if value in ("", None):
            return None
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def mean(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def median(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def std(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    mu = mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / len(values))


def bootstrap_mean_ci(values, confidence=0.95, n_boot=2000, seed=2025):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    samples = rng.choice(array, size=(int(n_boot), len(array)), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(samples, alpha)), float(np.quantile(samples, 1.0 - alpha))


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return value


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True) if path.parent != Path(".") else None
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_generation_failures(output_rows, attempts_csv, key_to_target, modes):
    if attempts_csv is None or not attempts_csv.exists():
        return output_rows
    evaluated = {
        (str(row.get("lmdb_key", "")), str(row.get("generation_index_within_target", "")))
        for row in output_rows
    }
    with attempts_csv.open(newline="", encoding="utf-8") as handle:
        for attempt in csv.DictReader(handle):
            identity = (str(attempt.get("lmdb_key", "")), str(attempt.get("generation_index_within_target", "")))
            if identity in evaluated:
                continue
            lmdb_key = identity[0]
            target_id = key_to_target.get(lmdb_key)
            if target_id is None:
                continue
            error = attempt.get("assembly_error") or "generation_failure"
            row = dict(attempt)
            row.update({
                "target_id": target_id,
                "rdkit_valid": "False",
                "rdkit_error": error,
                "prep_success": "False",
                "prep_error": error,
                "out_of_box": str(attempt.get("out_of_box", "")).title() if attempt.get("out_of_box") != "" else "",
            })
            for mode in modes:
                row[f"vina_{mode}"] = ""
                row[f"vina_{mode}_error"] = error
            output_rows.append(row)
    return output_rows


def target_group_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["target_id"])].append(row)
    return grouped


def per_target_stats(rows, modes):
    stats_rows = []
    for target_id, group in sorted(target_group_rows(rows).items()):
        raw_n = len(group)
        valid_n = sum(row.get("rdkit_valid") == "True" for row in group)
        prep_n = sum(row.get("prep_success") == "True" for row in group)
        base = {
            "target_id": target_id,
            "target_name": f"target_{target_id}",
            "raw_n": raw_n,
            "valid_n": valid_n,
            "prep_success_n": prep_n,
            "rdkit_validity": valid_n / raw_n if raw_n else "",
            "prep_success_rate": prep_n / valid_n if valid_n else "",
            "out_of_box_rate": sum(row.get("out_of_box") == "True" for row in group) / valid_n if valid_n else "",
            "generation_success_rate": sum(str(row.get("assembly_success")) == "True" for row in group) / raw_n if raw_n else "",
        }
        qed_values = [safe_float(row.get("qed")) for row in group]
        sa_values = [safe_float(row.get("sa")) for row in group]
        base["mean_qed"] = fmt(mean(qed_values))
        base["mean_sa"] = fmt(mean(sa_values))
        valid_smiles = [row.get("smiles") for row in group if row.get("rdkit_valid") == "True" and row.get("smiles")]
        base["uniqueness"] = len(set(valid_smiles)) / len(valid_smiles) if valid_smiles else ""
        for mode in modes:
            key = f"vina_{mode}"
            values = [safe_float(row.get(key)) for row in group]
            success_n = sum(v is not None for v in values)
            base[f"{key}_success_n"] = success_n
            base[f"{key}_success_rate"] = success_n / valid_n if valid_n else ""
            base[f"{key}_mean"] = fmt(mean(values))
            base[f"{key}_median"] = fmt(median(values))
            base[f"{key}_std"] = fmt(std(values))
            sorted_values = sorted(v for v in values if v is not None)
            negative_n = sum(v < 0 for v in sorted_values)
            base[f"{key}_negative_n"] = negative_n
            base[f"{key}_negative_rate"] = negative_n / success_n if success_n else ""
            base[f"{key}_raw_negative_yield"] = negative_n / raw_n if raw_n else ""
            base[f"{key}_median_below_zero"] = float(median(sorted_values) < 0) if sorted_values else ""
            base[f"{key}_top1"] = fmt(sorted_values[0]) if sorted_values else ""
            base[f"{key}_top5_mean"] = fmt(mean(sorted_values[:5])) if sorted_values else ""
            base[f"{key}_top10_mean"] = fmt(mean(sorted_values[:10])) if sorted_values else ""
            unique_values = []
            seen_smiles = set()
            for row, value in zip(group, values):
                if value is None:
                    continue
                smiles = row.get("smiles") or f"__row_{len(unique_values)}"
                if smiles in seen_smiles:
                    continue
                seen_smiles.add(smiles)
                unique_values.append(value)
            unique_values.sort()
            base[f"{key}_unique_top1"] = fmt(unique_values[0]) if unique_values else ""
            base[f"{key}_unique_top5_mean"] = fmt(mean(unique_values[:5])) if unique_values else ""
            base[f"{key}_unique_top10_mean"] = fmt(mean(unique_values[:10])) if unique_values else ""
        stats_rows.append(base)
    return stats_rows


def macro_summary(per_target_rows, modes):
    summary = []
    scalar_cols = [
        "generation_success_rate", "rdkit_validity", "prep_success_rate",
        "out_of_box_rate", "uniqueness", "mean_qed", "mean_sa",
    ]
    for mode in modes:
        key = f"vina_{mode}"
        scalar_cols.extend([
            f"{key}_success_rate",
            f"{key}_mean",
            f"{key}_median",
            f"{key}_negative_rate",
            f"{key}_raw_negative_yield",
            f"{key}_median_below_zero",
            f"{key}_top1",
            f"{key}_top5_mean",
            f"{key}_top10_mean",
            f"{key}_unique_top1",
            f"{key}_unique_top5_mean",
            f"{key}_unique_top10_mean",
        ])

    for col in scalar_cols:
        values = [safe_float(row.get(col)) for row in per_target_rows]
        ci_low, ci_high = bootstrap_mean_ci(values)
        summary.append({
            "level": "per_target_macro",
            "metric": col,
            "mean": fmt(mean(values)),
            "median": fmt(median(values)),
            "std": fmt(std(values)),
            "ci95_low": fmt(ci_low),
            "ci95_high": fmt(ci_high),
            "n_targets": sum(v is not None for v in values),
        })
    return summary


def pooled_summary(rows, modes):
    output = []
    raw_n = len(rows)
    valid_n = sum(row.get("rdkit_valid") == "True" for row in rows)
    prep_n = sum(row.get("prep_success") == "True" for row in rows)
    output.extend([
        {"level": "pooled_molecule", "metric": "raw_n", "mean": raw_n},
        {"level": "pooled_molecule", "metric": "rdkit_validity", "mean": fmt(valid_n / raw_n if raw_n else None)},
        {"level": "pooled_molecule", "metric": "prep_success_rate", "mean": fmt(prep_n / valid_n if valid_n else None)},
    ])
    for mode in modes:
        key = f"vina_{mode}"
        values = [safe_float(row.get(key)) for row in rows]
        valid_values = [value for value in values if value is not None]
        negative_n = sum(value < 0 for value in valid_values)
        output.append({
            "level": "pooled_molecule",
            "metric": key,
            "mean": fmt(mean(values)),
            "median": fmt(median(values)),
            "std": fmt(std(values)),
            "n": sum(v is not None for v in values),
        })
        output.extend([
            {
                "level": "pooled_molecule",
                "metric": f"{key}_negative_rate",
                "mean": fmt(negative_n / len(valid_values) if valid_values else None),
                "n": len(valid_values),
            },
            {
                "level": "pooled_molecule",
                "metric": f"{key}_raw_negative_yield",
                "mean": fmt(negative_n / raw_n if raw_n else None),
                "n": raw_n,
            },
        ])
    return output


def topk_table(per_target_rows, modes):
    rows = []
    for row in per_target_rows:
        out = {"level": "per_target", "target_id": row["target_id"], "target_name": row["target_name"]}
        for mode in modes:
            key = f"vina_{mode}"
            out[f"{key}_top1"] = row.get(f"{key}_top1", "")
            out[f"{key}_top5_mean"] = row.get(f"{key}_top5_mean", "")
            out[f"{key}_top10_mean"] = row.get(f"{key}_top10_mean", "")
            out[f"{key}_unique_top1"] = row.get(f"{key}_unique_top1", "")
            out[f"{key}_unique_top5_mean"] = row.get(f"{key}_unique_top5_mean", "")
            out[f"{key}_unique_top10_mean"] = row.get(f"{key}_unique_top10_mean", "")
        rows.append(out)

    macro = {"level": "per_target_macro", "target_id": "", "target_name": "macro_average"}
    for mode in modes:
        key = f"vina_{mode}"
        for suffix in (
            "top1", "top5_mean", "top10_mean",
            "unique_top1", "unique_top5_mean", "unique_top10_mean",
        ):
            col = f"{key}_{suffix}"
            macro[col] = fmt(mean([safe_float(row.get(col)) for row in per_target_rows]))
    rows.append(macro)
    return rows


def require_main_tools(prep_backend):
    vina = find_executable(["vina.exe", "vina"])
    if vina is None:
        raise RuntimeError("AutoDock Vina is not in PATH. Install with: conda install -c conda-forge vina")

    if prep_backend != "meeko":
        raise RuntimeError("Main experiment only supports --prep_backend meeko.")

    ligand_tool = find_executable(["mk_prepare_ligand.py", "mk_prepare_ligand"])
    receptor_tool = find_executable(["mk_prepare_receptor.py", "mk_prepare_receptor"])
    if ligand_tool is None or receptor_tool is None:
        raise RuntimeError(
            "Meeko tools are required in PATH: mk_prepare_ligand.py and mk_prepare_receptor.py. "
            "Install with: conda install -c conda-forge meeko"
        )
    return vina, ligand_tool, receptor_tool


def main():
    parser = argparse.ArgumentParser(description="Unified AutoDock Vina evaluation for generated CrossDocked molecules.")
    parser.add_argument("--metrics_csv", default="samples/crossdock_paper_compare1/metrics.csv")
    parser.add_argument("--generation_attempts_csv", default=None)
    parser.add_argument("--sdf_dir", default="samples/crossdock_paper_compare1/candidates")
    parser.add_argument("--data_root", default="data/crossdocked_pocket10")
    parser.add_argument("--lmdb", default="data/crossdocked_pocket10.lmdb")
    parser.add_argument("--split_file", default="data/crossdocked_pocket10_pose_split.pt")
    parser.add_argument("--split", default="test")
    parser.add_argument("--modes", default="score")
    parser.add_argument("--box_mode", default="ligand_heavy_bbox_margin", choices=["ligand_heavy_bbox_margin"])
    parser.add_argument("--box_margin", type=float, default=10.0)
    parser.add_argument("--box_min_size", type=float, default=20.0)
    parser.add_argument("--box_size", type=float, default=None)
    parser.add_argument(
        "--pose_out_of_box_policy",
        default="fail",
        choices=["fail", "recenter"],
        help=(
            "Policy for Vina Score/Min when the generated pose is outside the fixed reference box. "
            "Use 'fail' for the main paper-comparable experiment. 'recenter' is diagnostic only."
        ),
    )
    parser.add_argument(
        "--dock_out_of_box_policy",
        default="translate_to_box",
        choices=["translate_to_box", "fail"],
        help=(
            "Policy for Vina Dock when the input conformer is outside the fixed reference box. "
            "translate_to_box only changes the starting conformer for redocking and is recorded in the CSV."
        ),
    )
    parser.add_argument("--recenter_generated_to_reference", action="store_true")
    parser.add_argument("--prep_backend", default="meeko", choices=["meeko"])
    parser.add_argument("--default_altloc", default="A")
    parser.add_argument("--allow_bad_res", dest="allow_bad_res", action="store_true", default=True)
    parser.add_argument("--no_allow_bad_res", dest="allow_bad_res", action="store_false")
    parser.add_argument("--exhaustiveness", type=int, default=32)
    parser.add_argument("--num_modes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--cpu", type=int, default=8)
    parser.add_argument("--max_targets", type=int, default=None)
    parser.add_argument("--max_per_target", type=int, default=100)
    parser.add_argument("--samples_per_target", type=int, default=100)
    parser.add_argument("--oversample_factor", type=int, default=8)
    parser.add_argument("--source_samples_per_target", type=int, default=None)
    parser.add_argument("--work_dir", default="work_vina_own_full")
    parser.add_argument("--output_csv", default="vina_own_full.csv")
    parser.add_argument("--summary_csv", default="vina_own_summary.csv")
    parser.add_argument("--per_target_csv", default="vina_own_per_target.csv")
    parser.add_argument("--topk_csv", default="vina_own_topk.csv")
    args = parser.parse_args()

    modes = parse_modes(args.modes)
    if args.recenter_generated_to_reference:
        args.pose_out_of_box_policy = "recenter"
        print(
            "WARNING: --recenter_generated_to_reference / --pose_out_of_box_policy recenter "
            "is diagnostic only and should not be used for the main paper-comparable Vina Score/Min table."
        )
    vina, ligand_tool, receptor_tool = require_main_tools(args.prep_backend)
    vina_version = run_command([vina, "--version"]).strip().replace("\n", " | ")

    work_dir = resolve_path(args.work_dir)
    ligand_dir = work_dir / "ligands"
    receptor_dir = work_dir / "receptors"
    vina_out_dir = work_dir / "vina_outputs"
    log_dir = work_dir / "logs"
    for directory in (ligand_dir, receptor_dir, vina_out_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)

    data_root = resolve_path(args.data_root)
    lmdb_path = resolve_path(args.lmdb)
    split_file = resolve_path(args.split_file)

    keys = valid_split_keys(split_file, lmdb_path, args.split)
    key_to_target = {key.decode(): target_id for target_id, key in enumerate(keys)}

    source_samples_per_target = args.source_samples_per_target
    if source_samples_per_target is None:
        source_samples_per_target = args.samples_per_target * args.oversample_factor

    rows = load_generated_rows(
        args.metrics_csv,
        args.sdf_dir,
        source_samples_per_target,
        key_to_target,
        args.max_targets,
        args.max_per_target,
    )
    env = lmdb.open(str(lmdb_path), subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
    target_cache = {}
    output_rows = []

    for row in tqdm(rows, desc="Unified Vina evaluation"):
        sdf_path = resolve_path(row["sdf_path"])
        target_id = row.get("_eval_target_id")
        if target_id is None:
            target_id = eval_target_id_from_row(
                row, source_samples_per_target, key_to_target=key_to_target
            )
        target_id = int(target_id)
        result = dict(row)
        result.pop("_eval_target_id", None)
        result.update({
            "sdf_path": str(sdf_path),
            "target_id": target_id,
            "source_samples_per_target": source_samples_per_target,
            "prep_backend": args.prep_backend,
            "vina_version": vina_version,
            "rdkit_valid": "False",
            "prep_success": "False",
            "recenter_generated_to_reference": str(bool(args.recenter_generated_to_reference)),
            "pose_out_of_box_policy": args.pose_out_of_box_policy,
            "dock_out_of_box_policy": args.dock_out_of_box_policy,
            "out_of_box_before_recenter": "",
            "out_of_box": "",
            "generated_center_x": "",
            "generated_center_y": "",
            "generated_center_z": "",
            "generated_center_distance_to_box": "",
            "recenter_shift_x": "",
            "recenter_shift_y": "",
            "recenter_shift_z": "",
            "vina_score_input_recentered": "False",
            "vina_min_input_recentered": "False",
            "vina_dock_input_recentered": "False",
            "vina_dock_recenter_shift_x": "",
            "vina_dock_recenter_shift_y": "",
            "vina_dock_recenter_shift_z": "",
            "box_center_x": "",
            "box_center_y": "",
            "box_center_z": "",
            "box_size_x": "",
            "box_size_y": "",
            "box_size_z": "",
        })
        for mode in modes:
            result[f"vina_{mode}"] = ""
            result[f"vina_{mode}_error"] = ""

        try:
            if target_id >= len(keys):
                raise RuntimeError(f"target_{target_id} exceeds split length {len(keys)}")

            try:
                mol = sanitize_generated_sdf(sdf_path)
                result["rdkit_valid"] = "True"
            except Exception as exc:
                result["rdkit_error"] = f"invalid_rdkit: {exc}"
                for mode in modes:
                    result[f"vina_{mode}_error"] = "invalid_rdkit"
                output_rows.append(result)
                continue

            if target_id not in target_cache:
                sample = load_sample(env, keys[target_id])
                protein_path = data_root / sample["protein_filename"]
                ligand_path = data_root / sample["ligand_filename"]
                if not protein_path.exists():
                    raise RuntimeError(f"protein file not found: {protein_path}")
                if not ligand_path.exists():
                    raise RuntimeError(f"reference ligand file not found: {ligand_path}")
                center, size = reference_box_from_ligand(
                    ligand_path,
                    args.box_margin,
                    min_size=args.box_min_size,
                    fixed_size=args.box_size,
                )
                receptor_pdbqt = receptor_dir / f"target_{target_id}.pdbqt"
                prepare_receptor_meeko(
                    protein_path,
                    receptor_pdbqt,
                    receptor_tool,
                    args.default_altloc,
                    allow_bad_res=args.allow_bad_res,
                    log_path=log_dir / f"target_{target_id}_receptor_prep.log",
                )
                target_cache[target_id] = {
                    "lmdb_key": keys[target_id].decode(),
                    "protein_filename": sample["protein_filename"],
                    "ligand_filename": sample["ligand_filename"],
                    "center": center,
                    "size": size,
                    "receptor_pdbqt": receptor_pdbqt,
                }

            target = target_cache[target_id]
            result["lmdb_key"] = target["lmdb_key"]
            result["protein_filename"] = target["protein_filename"]
            result["ligand_filename"] = target["ligand_filename"]
            center = target["center"]
            size = target["size"]
            result["box_center_x"], result["box_center_y"], result["box_center_z"] = [fmt(x) for x in center]
            result["box_size_x"], result["box_size_y"], result["box_size_z"] = [fmt(x) for x in size]

            gen_coords, gen_masses = heavy_atom_coords_and_masses(mol)
            generated_center = weighted_center(gen_coords, gen_masses)
            result["generated_center_x"], result["generated_center_y"], result["generated_center_z"] = [
                fmt(x) for x in generated_center
            ]
            result["generated_center_distance_to_box"] = fmt(distance(generated_center, center))
            out_of_box_before_recenter = molecule_out_of_box(mol, center, size)
            result["out_of_box_before_recenter"] = str(out_of_box_before_recenter)

            pose_mol = mol
            if out_of_box_before_recenter and args.pose_out_of_box_policy == "recenter":
                pose_mol, _generated_center, shift = recenter_mol_to_reference(mol, center)
                result["recenter_shift_x"], result["recenter_shift_y"], result["recenter_shift_z"] = [
                    fmt(x) for x in shift
                ]
                for mode in POSE_MODES:
                    if mode in modes:
                        result[f"vina_{mode}_input_recentered"] = "True"
            pose_out_of_box = molecule_out_of_box(pose_mol, center, size)
            result["out_of_box"] = str(pose_out_of_box)

            pose_ligand_pdbqt = None
            needs_pose_pdbqt = any(mode in modes for mode in POSE_MODES)
            if needs_pose_pdbqt:
                pose_ligand_explicit_h_sdf = ligand_dir / f"{sdf_path.stem}_pose_explicitH.sdf"
                pose_ligand_pdbqt = ligand_dir / f"{sdf_path.stem}_pose.pdbqt"
                prepare_ligand_meeko(
                    pose_mol,
                    pose_ligand_explicit_h_sdf,
                    pose_ligand_pdbqt,
                    ligand_tool,
                    log_path=log_dir / f"{sdf_path.stem}_pose_ligand_prep.log",
                )

            dock_ligand_pdbqt = pose_ligand_pdbqt
            if "dock" in modes:
                dock_mol = mol
                if out_of_box_before_recenter:
                    if args.dock_out_of_box_policy == "translate_to_box":
                        dock_mol, _generated_center, dock_shift = recenter_mol_to_reference(mol, center)
                        result["vina_dock_input_recentered"] = "True"
                        result["vina_dock_recenter_shift_x"], result["vina_dock_recenter_shift_y"], result[
                            "vina_dock_recenter_shift_z"
                        ] = [fmt(x) for x in dock_shift]
                    else:
                        dock_mol = None
                if dock_mol is not None:
                    dock_ligand_explicit_h_sdf = ligand_dir / f"{sdf_path.stem}_dock_explicitH.sdf"
                    dock_ligand_pdbqt = ligand_dir / f"{sdf_path.stem}_dock.pdbqt"
                    prepare_ligand_meeko(
                        dock_mol,
                        dock_ligand_explicit_h_sdf,
                        dock_ligand_pdbqt,
                        ligand_tool,
                        log_path=log_dir / f"{sdf_path.stem}_dock_ligand_prep.log",
                    )

            result["prep_success"] = "True"

            for mode in modes:
                if mode == "dock" and dock_ligand_pdbqt is None:
                    result[f"vina_{mode}_error"] = "dock_input_out_of_box"
                    continue
                ligand_pdbqt = dock_ligand_pdbqt if mode == "dock" else pose_ligand_pdbqt
                try:
                    score = run_vina_mode(
                        mode,
                        vina,
                        target["receptor_pdbqt"],
                        ligand_pdbqt,
                        center,
                        size,
                        args,
                        vina_out_dir,
                        log_dir,
                        sdf_path.stem,
                    )
                    result[f"vina_{mode}"] = fmt(score)
                except Exception as exc:
                    result[f"vina_{mode}_error"] = f"vina_{mode}_failure: {exc}"

        except Exception as exc:
            message = str(exc)
            if result["rdkit_valid"] == "False":
                result["rdkit_error"] = message
            elif result["prep_success"] == "False":
                result["prep_error"] = f"preparation_failure: {message}"
            for mode in modes:
                if not result[f"vina_{mode}_error"]:
                    result[f"vina_{mode}_error"] = message

        output_rows.append(result)

    env.close()

    attempts_path = resolve_path(args.generation_attempts_csv) if args.generation_attempts_csv else None
    if attempts_path is None and args.metrics_csv:
        inferred_attempts = resolve_path(args.metrics_csv).parent / "generation_attempts.csv"
        if inferred_attempts.exists():
            attempts_path = inferred_attempts
    output_rows = append_generation_failures(output_rows, attempts_path, key_to_target, modes)

    write_csv(args.output_csv, output_rows)
    per_target = per_target_stats(output_rows, modes)
    write_csv(args.per_target_csv, per_target)
    summary = macro_summary(per_target, modes) + pooled_summary(output_rows, modes)
    write_csv(args.summary_csv, summary)
    write_csv(args.topk_csv, topk_table(per_target, modes))

    print(f"Rows: {len(output_rows)}")
    print(f"Results: {args.output_csv}")
    print(f"Per-target macro summary: {args.summary_csv}")
    print(f"Per-target table: {args.per_target_csv}")
    print(f"Top-k table: {args.topk_csv}")


if __name__ == "__main__":
    main()
