import argparse
import json
import math
import pickle
import re
from collections import Counter
from pathlib import Path

import lmdb
from tqdm import tqdm

from vina_eval import (
    append_generation_failures,
    find_executable,
    fmt,
    load_generated_rows,
    macro_summary,
    molecule_out_of_box,
    parse_vina_affinity,
    per_target_stats,
    pooled_summary,
    prepare_ligand_meeko,
    recenter_mol_to_reference,
    resolve_path,
    run_command,
    sanitize_generated_sdf,
    topk_table,
    valid_split_keys,
    write_csv,
)


RECEPTOR_RE = re.compile(r"^(?P<pdb_id>[0-9A-Za-z]{4})_(?P<chain>[^_]+)_rec_", re.IGNORECASE)
WATER_NAMES = {"HOH", "WAT", "H2O", "DOD"}
STANDARD_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "ASH", "GLH", "HID", "HIE", "HIP", "LYN", "CYX", "CYM",
    "A", "C", "G", "U", "DA", "DC", "DG", "DT", "DU",
}


def load_sample(env, key):
    with env.begin(write=False) as txn:
        value = txn.get(key)
    if value is None:
        raise KeyError(key)
    return pickle.loads(value)


def parse_identity(protein_filename):
    name = Path(protein_filename).name
    match = RECEPTOR_RE.match(name)
    if match is None:
        raise ValueError(f"Cannot parse receptor PDB ID and chain from {name}")
    return match.group("pdb_id").upper(), match.group("chain")


def parse_chain_ids(value):
    return [item.strip() for item in re.split(r"[;,]+", str(value)) if item.strip()]


def heavy_atom_box_from_pdb(pdb_path, padding):
    coords = []
    for line in Path(pdb_path).read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        element = line[76:78].strip().upper() if len(line) >= 78 else ""
        atom_name = line[12:16].strip().upper() if len(line) >= 16 else ""
        if element in {"H", "D"} or (not element and atom_name.startswith(("H", "D"))):
            continue
        try:
            coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
        except (ValueError, IndexError):
            continue
    if not coords:
        raise ValueError(f"No heavy-atom coordinates found in {pdb_path}")
    mins = [min(coord[axis] for coord in coords) for axis in range(3)]
    maxs = [max(coord[axis] for coord in coords) for axis in range(3)]
    center = [(mins[axis] + maxs[axis]) / 2.0 for axis in range(3)]
    size = [maxs[axis] - mins[axis] + 2.0 * float(padding) for axis in range(3)]
    return center, size, len(coords)


def write_meeko_input_without_hydrogens(source_pdb, output_pdb):
    kept = []
    removed_hydrogens = 0
    for line in Path(source_pdb).read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            element = line[76:78].strip().upper() if len(line) >= 78 else ""
            atom_name = line[12:16].strip().upper() if len(line) >= 16 else ""
            is_hydrogen = element in {"H", "D"} or (
                not element and atom_name.startswith(("H", "D"))
            )
            if is_hydrogen:
                removed_hydrogens += 1
                continue
        kept.append(line)
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    output_pdb.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return removed_hydrogens


def _residue_counts(topology):
    return Counter(residue.name.strip().upper() for residue in topology.residues())


def correct_suspicious_terminal_oxt(pdb_path, min_ca_oxt_distance=2.1, c_oxt_length=1.25):
    lines = Path(pdb_path).read_text(encoding="utf-8", errors="replace").splitlines()
    residues = {}
    for line_idx, line in enumerate(lines):
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 54:
            continue
        key = (line[21:22], line[22:26], line[26:27])
        atom_name = line[12:16].strip().upper()
        if atom_name not in {"CA", "C", "O", "OXT"}:
            continue
        try:
            coord = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        except ValueError:
            continue
        residues.setdefault(key, {})[atom_name] = (line_idx, coord)

    corrected = 0
    for atoms in residues.values():
        if not {"CA", "C", "O", "OXT"}.issubset(atoms):
            continue
        ca = atoms["CA"][1]
        carbon = atoms["C"][1]
        oxygen = atoms["O"][1]
        oxt = atoms["OXT"][1]
        ca_oxt = math.sqrt(sum((ca[i] - oxt[i]) ** 2 for i in range(3)))
        if ca_oxt >= float(min_ca_oxt_distance):
            continue

        def unit(vector):
            norm = math.sqrt(sum(value * value for value in vector))
            if norm < 1e-8:
                raise ValueError("zero-length terminal bond vector")
            return [value / norm for value in vector]

        c_to_o = unit([oxygen[i] - carbon[i] for i in range(3)])
        c_to_ca = unit([ca[i] - carbon[i] for i in range(3)])
        direction = unit([-(c_to_o[i] + c_to_ca[i]) for i in range(3)])
        new_oxt = [carbon[i] + float(c_oxt_length) * direction[i] for i in range(3)]
        line_idx = atoms["OXT"][0]
        old_line = lines[line_idx]
        lines[line_idx] = (
            old_line[:30]
            + f"{new_oxt[0]:8.3f}{new_oxt[1]:8.3f}{new_oxt[2]:8.3f}"
            + old_line[54:]
        )
        corrected += 1

    if corrected:
        Path(pdb_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return corrected


def repair_full_receptor(
    input_pdb,
    output_pdb,
    chain_mode,
    mapped_chains,
    keep_heterogens,
    ph,
    replace_nonstandard,
):
    try:
        from pdbfixer import PDBFixer
        from openmm.app import Modeller, PDBFile
    except ImportError as exc:
        raise RuntimeError(
            "PDBFixer/OpenMM are required for full-protein repair. Install with: "
            "conda install -c conda-forge pdbfixer openmm"
        ) from exc

    fixer = PDBFixer(filename=str(input_pdb))
    input_counts = _residue_counts(fixer.topology)

    removed_chains = []
    if chain_mode == "mapped":
        wanted = set(mapped_chains)
        chains = list(fixer.topology.chains())
        delete_indices = [idx for idx, chain in enumerate(chains) if chain.id not in wanted]
        removed_chains = [chains[idx].id for idx in delete_indices]
        if len(delete_indices) == len(chains):
            available = [chain.id for chain in chains]
            raise ValueError(f"Mapped chains {sorted(wanted)} not found; available chains: {available}")
        if delete_indices:
            fixer.removeChains(delete_indices)

    nonstandard_count = 0
    if replace_nonstandard:
        fixer.findNonstandardResidues()
        nonstandard_count = len(fixer.nonstandardResidues)
        fixer.replaceNonstandardResidues()

    keep = {name.upper() for name in keep_heterogens}
    modeller = Modeller(fixer.topology, fixer.positions)
    deleted_residues = []
    for residue in list(modeller.topology.residues()):
        name = residue.name.strip().upper()
        if name in STANDARD_RESIDUES:
            continue
        if name in keep and name not in WATER_NAMES:
            continue
        deleted_residues.append(residue)
    deleted_counts = Counter(residue.name.strip().upper() for residue in deleted_residues)
    if deleted_residues:
        modeller.delete(deleted_residues)
    fixer.topology = modeller.topology
    fixer.positions = modeller.positions

    # Do not build missing loops: only repair missing atoms in residues that are present.
    fixer.findMissingResidues()
    missing_residue_blocks = len(fixer.missingResidues)
    fixer.missingResidues = {}

    # Do not invent poorly constrained side chains in a consecutive unresolved
    # terminal tail. Trim only terminal residues with missing non-terminal heavy
    # atoms; unresolved internal residues remain subject to strict preparation.
    trimmed_terminal_residues = []
    for _round in range(20):
        fixer.findMissingAtoms()
        residues_with_missing_atoms = set(fixer.missingAtoms)
        to_delete = []
        for chain in fixer.topology.chains():
            residues = list(chain.residues())
            if not residues:
                continue
            if residues[0] in residues_with_missing_atoms:
                to_delete.append(residues[0])
            if residues[-1] in residues_with_missing_atoms and residues[-1] not in to_delete:
                to_delete.append(residues[-1])
        if not to_delete:
            break
        trimmed_terminal_residues.extend(
            f"{residue.chain.id}:{residue.id}:{residue.name}" for residue in to_delete
        )
        modeller = Modeller(fixer.topology, fixer.positions)
        modeller.delete(to_delete)
        fixer.topology = modeller.topology
        fixer.positions = modeller.positions
    else:
        raise RuntimeError("More than 20 unresolved terminal residues would need trimming")

    fixer.findMissingAtoms()
    missing_heavy_atoms = sum(len(atoms) for atoms in fixer.missingAtoms.values())
    missing_terminal_atoms = sum(len(atoms) for atoms in fixer.missingTerminals.values())
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(float(ph))

    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    with output_pdb.open("w", encoding="utf-8") as handle:
        PDBFile.writeFile(fixer.topology, fixer.positions, handle, keepIds=True)
    terminal_oxt_positions_corrected = correct_suspicious_terminal_oxt(output_pdb)

    output_counts = _residue_counts(fixer.topology)
    return {
        "input_residue_count": sum(input_counts.values()),
        "output_residue_count": sum(output_counts.values()),
        "removed_chains": ";".join(removed_chains),
        "deleted_heterogen_residues": sum(deleted_counts.values()),
        "deleted_heterogen_names": ";".join(f"{name}:{count}" for name, count in sorted(deleted_counts.items())),
        "kept_heterogens": ";".join(sorted(keep)),
        "nonstandard_residues_replaced": nonstandard_count,
        "missing_residue_blocks_not_built": missing_residue_blocks,
        "trimmed_unresolved_terminal_residues": ";".join(trimmed_terminal_residues),
        "trimmed_unresolved_terminal_residue_count": len(trimmed_terminal_residues),
        "missing_heavy_atoms_added": missing_heavy_atoms,
        "missing_terminal_atoms_added": missing_terminal_atoms,
        "terminal_oxt_positions_corrected": terminal_oxt_positions_corrected,
    }


def prepare_receptor_meeko(prepared_pdb, receptor_pdbqt, receptor_tool, default_altloc, allow_bad_res, log_path):
    prefix = receptor_pdbqt.with_suffix("")
    cmd = [
        receptor_tool,
        "-i", prepared_pdb,
        "-o", prefix,
        "-p",
        "--charge_model", "gasteiger",
    ]
    if default_altloc:
        cmd.extend(["--default_altloc", default_altloc])
    if allow_bad_res:
        cmd.append("--allow_bad_res")
    output = run_command(cmd, log_path=log_path)
    generated = Path(str(prefix) + ".pdbqt")
    if not generated.exists():
        raise RuntimeError(f"Meeko did not create receptor PDBQT: {generated}")
    if generated.resolve() != receptor_pdbqt.resolve():
        generated.replace(receptor_pdbqt)
    return output


def validate_receptor_pdbqt(receptor_pdbqt):
    atom_count = 0
    for line_number, line in enumerate(
        Path(receptor_pdbqt).read_text(encoding="utf-8", errors="replace").splitlines(),
        start=1,
    ):
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        atom_count += 1
        try:
            coordinates = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            charge = float(line[70:76])
        except (ValueError, IndexError) as exc:
            raise ValueError(
                f"Malformed receptor PDBQT atom record at line {line_number}: {line}"
            ) from exc
        if not all(math.isfinite(value) for value in (*coordinates, charge)):
            raise ValueError(f"Non-finite receptor PDBQT value at line {line_number}")
    if atom_count == 0:
        raise ValueError(f"No receptor atoms found in {receptor_pdbqt}")
    return atom_count


def prepare_receptor_openbabel(
    prepared_pdb,
    receptor_pdbqt,
    receptor_tool,
    ph,
    log_path,
):
    cmd = [
        receptor_tool,
        prepared_pdb,
        "-O", receptor_pdbqt,
        "-xr",
        "-p", str(float(ph)),
        "--partialcharge", "gasteiger",
    ]
    output = run_command(cmd, log_path=log_path)
    if not receptor_pdbqt.exists():
        raise RuntimeError(f"OpenBabel did not create receptor PDBQT: {receptor_pdbqt}")
    return output


def run_blind_docking(vina, receptor_pdbqt, ligand_pdbqt, center, size, args, out_path, log_path):
    if args.resume and Path(out_path).exists() and Path(log_path).exists():
        cached_output = Path(log_path).read_text(encoding="utf-8", errors="replace")
        try:
            cached_score = parse_vina_affinity(cached_output)
            if math.isfinite(cached_score):
                return cached_score
        except (ValueError, TypeError):
            pass
    cmd = [
        vina,
        "--receptor", receptor_pdbqt,
        "--ligand", ligand_pdbqt,
        "--center_x", center[0],
        "--center_y", center[1],
        "--center_z", center[2],
        "--size_x", size[0],
        "--size_y", size[1],
        "--size_z", size[2],
        "--cpu", args.cpu,
        "--exhaustiveness", args.exhaustiveness,
        "--num_modes", args.num_modes,
        "--seed", args.seed,
        "--out", out_path,
    ]
    output = run_command(cmd, log_path=log_path)
    score = parse_vina_affinity(output)
    if not math.isfinite(score):
        raise ValueError(f"Non-finite Vina docking score: {score}")
    return score


def target_metadata(data_root, lmdb_path, split_file, split_name, max_targets):
    keys = valid_split_keys(split_file, lmdb_path, split_name)
    if max_targets is not None:
        keys = keys[:max_targets]
    env = lmdb.open(str(lmdb_path), subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
    targets = []
    try:
        for target_id, key in enumerate(keys):
            sample = load_sample(env, key)
            pocket_rel = Path(sample["protein_filename"])
            pdb_id, chain_text = parse_identity(pocket_rel)
            full_pdb = data_root / pocket_rel.parent / f"{pdb_id}_full.pdb"
            targets.append({
                "target_id": target_id,
                "lmdb_key": key.decode(),
                "pocket_protein_filename": str(pocket_rel),
                "reference_ligand_filename": str(sample["ligand_filename"]),
                "pdb_id": pdb_id,
                "mapped_chains": parse_chain_ids(chain_text),
                "full_pdb": full_pdb,
            })
    finally:
        env.close()
    return keys, targets


def prepare_targets(targets, args, receptor_tool, work_dir):
    prepared_dir = work_dir / "prepared_receptors"
    pdbqt_dir = work_dir / "receptors_pdbqt"
    log_dir = work_dir / "logs"
    for directory in (prepared_dir, pdbqt_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)

    keep_heterogens = [name.strip().upper() for name in args.keep_heterogens.split(",") if name.strip()]
    cache = {}
    manifest = []
    for target in tqdm(targets, desc="Preparing full receptors"):
        row = {
            "target_id": target["target_id"],
            "lmdb_key": target["lmdb_key"],
            "pdb_id": target["pdb_id"],
            "mapped_chains": ";".join(target["mapped_chains"]),
            "chain_mode": args.chain_mode,
            "pocket_protein_filename": target["pocket_protein_filename"],
            "reference_ligand_filename": target["reference_ligand_filename"],
            "full_protein_filename": str(target["full_pdb"]),
            "receptor_prep_success": "False",
            "receptor_prep_error": "",
            "receptor_prep_backend": args.receptor_prep_backend,
            "ph": args.ph,
            "charge_model": "gasteiger",
            "box_padding_per_side": args.protein_box_padding,
        }
        try:
            if not target["full_pdb"].exists():
                raise FileNotFoundError(f"Full PDB not found: {target['full_pdb']}")
            stem = f"target_{target['target_id']}_{target['pdb_id']}"
            prepared_pdb = prepared_dir / f"{stem}_prepared.pdb"
            meeko_input_pdb = prepared_dir / f"{stem}_meeko_input_noH.pdb"
            repair_json = prepared_dir / f"{stem}_repair.json"
            receptor_pdbqt = pdbqt_dir / f"{stem}_{args.receptor_prep_backend}.pdbqt"
            if args.overwrite:
                prepared_pdb.unlink(missing_ok=True)
                meeko_input_pdb.unlink(missing_ok=True)
                repair_json.unlink(missing_ok=True)
                receptor_pdbqt.unlink(missing_ok=True)

            repair_info = {}
            if not prepared_pdb.exists():
                repair_info = repair_full_receptor(
                    target["full_pdb"],
                    prepared_pdb,
                    args.chain_mode,
                    target["mapped_chains"],
                    keep_heterogens,
                    args.ph,
                    args.replace_nonstandard,
                )
                repair_json.write_text(json.dumps(repair_info, indent=2), encoding="utf-8")
            elif repair_json.exists():
                repair_info = json.loads(repair_json.read_text(encoding="utf-8"))
            removed_input_hydrogens = write_meeko_input_without_hydrogens(
                prepared_pdb, meeko_input_pdb
            )
            if not receptor_pdbqt.exists():
                if args.receptor_prep_backend == "meeko":
                    prepare_receptor_meeko(
                        meeko_input_pdb,
                        receptor_pdbqt,
                        receptor_tool,
                        args.default_altloc,
                        args.allow_bad_res,
                        log_dir / f"{stem}_meeko.log",
                    )
                else:
                    prepare_receptor_openbabel(
                        meeko_input_pdb,
                        receptor_pdbqt,
                        receptor_tool,
                        args.ph,
                        log_dir / f"{stem}_openbabel.log",
                    )
            receptor_pdbqt_atom_count = validate_receptor_pdbqt(receptor_pdbqt)
            center, size, heavy_atom_count = heavy_atom_box_from_pdb(prepared_pdb, args.protein_box_padding)
            box_volume = size[0] * size[1] * size[2]
            row.update(repair_info)
            row.update({
                "prepared_protein_filename": str(prepared_pdb),
                "meeko_input_protein_filename": str(meeko_input_pdb),
                "meeko_input_hydrogens_removed": removed_input_hydrogens,
                "final_hydrogen_and_charge_provider": (
                    "Meeko" if args.receptor_prep_backend == "meeko" else "OpenBabel"
                ),
                "receptor_pdbqt": str(receptor_pdbqt),
                "receptor_pdbqt_atom_count": receptor_pdbqt_atom_count,
                "receptor_heavy_atom_count": heavy_atom_count,
                "box_center_x": fmt(center[0]),
                "box_center_y": fmt(center[1]),
                "box_center_z": fmt(center[2]),
                "box_size_x": fmt(size[0]),
                "box_size_y": fmt(size[1]),
                "box_size_z": fmt(size[2]),
                "box_volume": fmt(box_volume),
                "large_box_warning": str(box_volume > 27000.0 or max(size) > 30.0),
                "receptor_prep_success": "True",
            })
            cache[target["target_id"]] = {
                **target,
                "prepared_pdb": prepared_pdb,
                "receptor_pdbqt": receptor_pdbqt,
                "center": center,
                "size": size,
                "box_volume": box_volume,
            }
        except Exception as exc:
            row["receptor_prep_error"] = str(exc)
        manifest.append(row)
    return cache, manifest


def check_tools(prepare_only, receptor_prep_backend):
    if receptor_prep_backend == "meeko":
        receptor_tool = find_executable(["mk_prepare_receptor.py", "mk_prepare_receptor"])
        receptor_install = "conda install -c conda-forge meeko"
    else:
        receptor_tool = find_executable(["obabel", "obabel.exe"])
        receptor_install = "conda install -c conda-forge openbabel"
    if receptor_tool is None:
        raise RuntimeError(
            f"{receptor_prep_backend} receptor tool is missing: {receptor_install}"
        )
    if prepare_only:
        return None, None, receptor_tool
    vina = find_executable(["vina", "vina.exe"])
    ligand_tool = find_executable(["mk_prepare_ligand.py", "mk_prepare_ligand"])
    if vina is None or ligand_tool is None:
        raise RuntimeError("Vina/Meeko ligand tool missing: conda install -c conda-forge vina meeko")
    return vina, ligand_tool, receptor_tool


def main():
    parser = argparse.ArgumentParser(
        description="Prepare complete CrossDocked receptor chains and run whole-protein blind AutoDock Vina docking."
    )
    parser.add_argument("--metrics_csv", default="samples/run8_formal_test99_100perpocket_seed2025_vocabfix/metrics.csv")
    parser.add_argument("--generation_attempts_csv", default=None)
    parser.add_argument("--sdf_dir", default="samples/run8_formal_test99_100perpocket_seed2025_vocabfix/candidates")
    parser.add_argument("--data_root", default="data/crossdocked_pocket10")
    parser.add_argument("--lmdb", default="data/crossdocked_pocket10_test99_v2.lmdb")
    parser.add_argument("--split_file", default="data/crossdocked_pocket10_test99_split.pt")
    parser.add_argument("--split", default="test")
    parser.add_argument("--chain_mode", choices=["mapped", "all"], default="mapped")
    parser.add_argument("--keep_heterogens", default="")
    parser.add_argument("--ph", type=float, default=7.4)
    parser.add_argument("--replace_nonstandard", action="store_true", default=True)
    parser.add_argument("--no_replace_nonstandard", dest="replace_nonstandard", action="store_false")
    parser.add_argument("--protein_box_padding", type=float, default=6.0)
    parser.add_argument("--default_altloc", default="A")
    parser.add_argument("--allow_bad_res", action="store_true", default=False)
    parser.add_argument(
        "--receptor_prep_backend",
        choices=["meeko", "openbabel"],
        default="meeko",
        help=(
            "Use one receptor preparation backend for every target. OpenBabel is "
            "provided for complete proteins that fail Meeko/RDKit proximity-bond perception."
        ),
    )
    parser.add_argument("--exhaustiveness", type=int, default=64)
    parser.add_argument("--num_modes", type=int, default=9)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--cpu", type=int, default=8)
    parser.add_argument("--max_targets", type=int, default=None)
    parser.add_argument("--max_per_target", type=int, default=100)
    parser.add_argument("--samples_per_target", type=int, default=100)
    parser.add_argument("--source_samples_per_target", type=int, default=100)
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse a completed docking pose and parseable Vina log from the work directory.",
    )
    parser.add_argument("--append_generation_failures", action="store_true", default=True)
    parser.add_argument(
        "--no_append_generation_failures",
        dest="append_generation_failures",
        action="store_false",
        help="Use for a small smoke test so unevaluated raw attempts are not appended to its CSV.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--work_dir", default="work_vina_full_protein_test99_seed2025")
    parser.add_argument("--receptor_manifest_csv", default="full_protein_receptor_manifest.csv")
    parser.add_argument("--output_csv", default="vina_full_protein_test99_seed2025.csv")
    parser.add_argument("--summary_csv", default="vina_full_protein_test99_seed2025_summary.csv")
    parser.add_argument("--per_target_csv", default="vina_full_protein_test99_seed2025_per_target.csv")
    parser.add_argument("--topk_csv", default="vina_full_protein_test99_seed2025_topk.csv")
    args = parser.parse_args()

    if args.protein_box_padding < 0:
        raise ValueError("--protein_box_padding must be non-negative")
    if args.receptor_prep_backend != "meeko" and args.allow_bad_res:
        raise ValueError("--allow_bad_res is only supported by the Meeko receptor backend")
    vina, ligand_tool, receptor_tool = check_tools(
        args.prepare_only, args.receptor_prep_backend
    )
    work_dir = resolve_path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    data_root = resolve_path(args.data_root)
    lmdb_path = resolve_path(args.lmdb)
    split_file = resolve_path(args.split_file)
    keys, targets = target_metadata(data_root, lmdb_path, split_file, args.split, args.max_targets)
    key_to_target = {key.decode(): target_id for target_id, key in enumerate(keys)}

    target_cache, manifest = prepare_targets(targets, args, receptor_tool, work_dir)
    write_csv(args.receptor_manifest_csv, manifest)
    failed_receptors = len(targets) - len(target_cache)
    print(f"Prepared full receptors: {len(target_cache)}/{len(targets)}")
    print(f"Receptor manifest: {args.receptor_manifest_csv}")
    if args.prepare_only:
        if failed_receptors:
            raise RuntimeError(f"{failed_receptors} full receptors failed preparation; inspect the manifest and logs")
        return

    ligand_dir = work_dir / "ligands_pdbqt"
    vina_out_dir = work_dir / "vina_outputs"
    log_dir = work_dir / "logs"
    for directory in (ligand_dir, vina_out_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rows = load_generated_rows(
        args.metrics_csv,
        args.sdf_dir,
        args.source_samples_per_target,
        key_to_target=key_to_target,
        max_targets=args.max_targets,
        max_per_target=args.max_per_target,
    )
    output_rows = []
    vina_version = run_command([vina, "--version"]).strip().replace("\n", " | ")
    for row in tqdm(rows, desc="Whole-protein blind Vina docking"):
        result = dict(row)
        target_id = int(result.pop("_eval_target_id"))
        sdf_path = resolve_path(result["sdf_path"])
        result.update({
            "sdf_path": str(sdf_path),
            "target_id": target_id,
            "evaluation_scope": "full_protein_blind_docking",
            "chain_mode": args.chain_mode,
            "rdkit_valid": "",
            "prep_success": "False",
            "receptor_prep_success": "False",
            "out_of_box": "",
            "blind_dock_input_recentered": "False",
            "vina_version": vina_version,
            "vina_seed": args.seed,
            "vina_exhaustiveness": args.exhaustiveness,
            "vina_num_modes": args.num_modes,
            "vina_cpu": args.cpu,
            "protein_box_padding": args.protein_box_padding,
            "vina_dock": "",
            "full_protein_vina_score": "",
            "vina_dock_error": "",
        })
        try:
            target = target_cache.get(target_id)
            if target is None:
                manifest_row = next((item for item in manifest if item["target_id"] == target_id), None)
                error = manifest_row.get("receptor_prep_error", "unknown receptor preparation failure") if manifest_row else "target missing"
                raise RuntimeError(f"receptor_prep_failure: {error}")
            result.update({
                "lmdb_key": target["lmdb_key"],
                "pdb_id": target["pdb_id"],
                "mapped_chains": ";".join(target["mapped_chains"]),
                "protein_filename": target["pocket_protein_filename"],
                "full_protein_filename": str(target["full_pdb"]),
                "prepared_protein_filename": str(target["prepared_pdb"]),
                "receptor_pdbqt": str(target["receptor_pdbqt"]),
                "box_center_x": fmt(target["center"][0]),
                "box_center_y": fmt(target["center"][1]),
                "box_center_z": fmt(target["center"][2]),
                "box_size_x": fmt(target["size"][0]),
                "box_size_y": fmt(target["size"][1]),
                "box_size_z": fmt(target["size"][2]),
                "box_volume": fmt(target["box_volume"]),
                "receptor_prep_success": "True",
            })
            mol = sanitize_generated_sdf(sdf_path)
            result["rdkit_valid"] = "True"
            out_of_box = molecule_out_of_box(mol, target["center"], target["size"])
            result["out_of_box"] = str(out_of_box)
            if out_of_box:
                mol, _old_center, _shift = recenter_mol_to_reference(mol, target["center"])
                result["blind_dock_input_recentered"] = "True"

            stem = sdf_path.stem
            ligand_sdf_h = ligand_dir / f"{stem}_explicitH.sdf"
            ligand_pdbqt = ligand_dir / f"{stem}.pdbqt"
            prepare_ligand_meeko(
                mol,
                ligand_sdf_h,
                ligand_pdbqt,
                ligand_tool,
                log_path=log_dir / f"{stem}_ligand_prep.log",
            )
            result["prep_success"] = "True"
            score = run_blind_docking(
                vina,
                target["receptor_pdbqt"],
                ligand_pdbqt,
                target["center"],
                target["size"],
                args,
                vina_out_dir / f"{stem}_full_dock.pdbqt",
                log_dir / f"{stem}_full_dock.log",
            )
            result["vina_dock"] = fmt(score)
            result["full_protein_vina_score"] = fmt(score)
        except Exception as exc:
            message = str(exc)
            if message.startswith("receptor_prep_failure:"):
                result["receptor_prep_error"] = message
            elif result["rdkit_valid"] == "":
                result["rdkit_valid"] = "False"
                result["rdkit_error"] = message
            elif result["prep_success"] == "False":
                result["prep_error"] = message
            result["vina_dock_error"] = message
        output_rows.append(result)

    if args.append_generation_failures:
        attempts_path = resolve_path(args.generation_attempts_csv) if args.generation_attempts_csv else None
        if attempts_path is None:
            inferred = resolve_path(args.metrics_csv).parent / "generation_attempts.csv"
            attempts_path = inferred if inferred.exists() else None
        output_rows = append_generation_failures(output_rows, attempts_path, key_to_target, ["dock"])
    write_csv(args.output_csv, output_rows)
    per_target = per_target_stats(output_rows, ["dock"])
    write_csv(args.per_target_csv, per_target)
    write_csv(args.summary_csv, macro_summary(per_target, ["dock"]) + pooled_summary(output_rows, ["dock"]))
    write_csv(args.topk_csv, topk_table(per_target, ["dock"]))
    print(f"Rows: {len(output_rows)}")
    print(f"Results: {args.output_csv}")
    print(f"Summary: {args.summary_csv}")
    print(f"Per-target: {args.per_target_csv}")
    print(f"Top-k: {args.topk_csv}")


if __name__ == "__main__":
    main()
