from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs

from case_utils import (
    CASE_ROOT,
    copy_file,
    fingerprint,
    load_config,
    load_vina,
    molecular_properties,
    numeric,
    parse_bool,
    project_path,
    read_first_sdf,
    sample_stem,
    tanimoto,
)


def target_dir(config: dict, target_id: int) -> Path:
    if target_id == int(config["primary_target_id"]):
        return CASE_ROOT / config["primary_target_name"]
    return CASE_ROOT / f"target_{target_id}"


def load_training_fingerprints(path: Path):
    fingerprints = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            smiles = line.strip().split()[0] if line.strip() else ""
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fingerprints.append(fingerprint(mol))
    return fingerprints


def training_novelty(mol: Chem.Mol, training_fps) -> tuple[float, float]:
    fp = fingerprint(mol)
    maximum = 0.0
    chunk_size = 10000
    for start in range(0, len(training_fps), chunk_size):
        similarities = DataStructs.BulkTanimotoSimilarity(
            fp, training_fps[start : start + chunk_size]
        )
        if similarities:
            maximum = max(maximum, max(similarities))
    return float(1.0 - maximum), float(maximum)


def pareto_mask(values: np.ndarray) -> np.ndarray:
    """Return nondominated rows for objectives where larger is better."""
    keep = np.ones(values.shape[0], dtype=bool)
    for index, row in enumerate(values):
        if not keep[index]:
            continue
        dominated_by_other = np.any(
            np.all(values >= row, axis=1)
            & np.any(values > row, axis=1)
        )
        if dominated_by_other:
            keep[index] = False
    return keep


def rank_good(series: pd.Series) -> pd.Series:
    return series.rank(pct=True, ascending=True, method="average")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-id", type=int, default=48)
    parser.add_argument("--skip-training-novelty", action="store_true")
    args = parser.parse_args()

    config = load_config()
    directory = target_dir(config, args.target_id)
    manifest = pd.read_csv(directory / "properties" / "case_pose_manifest.csv")
    clashes = pd.read_csv(directory / "properties" / "case_clashes.csv")
    vina = load_vina(config)
    vina = vina[
        vina["target_id"].eq(args.target_id)
        & vina["sdf_path"].notna()
        & vina["source_sample_idx"].notna()
    ].copy()
    vina["stem"] = vina.apply(sample_stem, axis=1)

    standard = pd.read_csv(project_path(config, "standard_metrics_csv"))
    standard["stem"] = standard["file"].str.extract(r"(target_\d+_sample_\d+)")
    standard = standard[["stem", "SA_Score", "Lipinski"]].copy()
    standard = standard.rename(columns={"Lipinski": "Lipinski_paper"})

    frame = vina.merge(manifest, on=["target_id", "source_sample_idx", "stem"], how="inner")
    frame = frame.merge(clashes, on=["target_id", "source_sample_idx", "stem"], how="inner")
    frame = frame.merge(standard, on="stem", how="left", suffixes=("", "_paper"))
    frame["global_vina"] = numeric(frame["full_protein_vina_score"])
    if "local_vina" not in frame:
        raise RuntimeError(
            "Local pocket docking results are missing. Run redock_case_target.py "
            "and then analyze_clashes.py before candidate selection."
        )
    frame["vina"] = numeric(frame["local_vina"])

    records = []
    for _, row in frame.iterrows():
        mol = read_first_sdf(Path(row["generated_sdf"]), remove_hs=False)
        props = molecular_properties(
            mol,
            existing_qed=float(row["qed"]) if pd.notna(row["qed"]) else None,
            existing_raw_sa=float(row["sa"]) if pd.notna(row["sa"]) else None,
            existing_logp=float(row["logp"]) if pd.notna(row["logp"]) else None,
            existing_mw=float(row["mw"]) if pd.notna(row["mw"]) else None,
        )
        props["sa"] = (
            float(row["SA_Score"]) if pd.notna(row.get("SA_Score")) else props["sa"]
        )
        props["lipinski"] = (
            int(row["Lipinski_paper"])
            if pd.notna(row.get("Lipinski_paper"))
            else props["lipinski"]
        )
        records.append({**row.to_dict(), **props})

    candidates = pd.DataFrame(records)
    native = read_first_sdf(directory / "reference" / "native_MA9.sdf")
    crossdocked = read_first_sdf(directory / "reference" / "crossdocked_reference.sdf")
    candidates["similarity_to_native_MA9"] = [
        tanimoto(read_first_sdf(Path(path)), native)
        for path in candidates["generated_sdf"]
    ]
    candidates["similarity_to_crossdocked_524"] = [
        tanimoto(read_first_sdf(Path(path)), crossdocked)
        for path in candidates["generated_sdf"]
    ]

    if args.skip_training_novelty:
        candidates["training_novelty"] = np.nan
        candidates["max_training_similarity"] = np.nan
    else:
        training_fps = load_training_fingerprints(project_path(config, "training_smiles"))
        novelty = [
            training_novelty(read_first_sdf(Path(path)), training_fps)
            for path in candidates["generated_sdf"]
        ]
        candidates["training_novelty"] = [item[0] for item in novelty]
        candidates["max_training_similarity"] = [item[1] for item in novelty]

    mandatory = (
        candidates["rdkit_valid"].map(parse_bool)
        & candidates["assembly_success"].map(parse_bool)
        & numeric(candidates["connected_components"]).eq(1)
        & candidates["vina"].notna()
        & ~candidates["out_of_box"].map(parse_bool)
        & candidates["mw"].between(
            float(config["minimum_mw"]), float(config["maximum_mw"])
        )
        & numeric(candidates["severe_vdw_clash_count_local_docked"]).eq(0)
        & numeric(candidates["pocket_occupancy_fraction_local_docked"]).ge(
            float(config["minimum_pocket_occupancy"])
        )
    )
    candidates["mandatory_pass"] = mandatory
    audit_path = directory / "properties" / "case_molecule_selection_audit.csv"
    candidates.to_csv(audit_path, index=False)
    eligible = candidates[mandatory].copy()
    if len(eligible) < 2:
        raise RuntimeError(
            f"Only {len(eligible)} molecules passed mandatory filters. "
            "Inspect case_molecule_selection_audit.csv; do not silently relax thresholds."
        )

    eligible["vina_rank"] = rank_good(-eligible["vina"])
    eligible["qed_rank"] = rank_good(eligible["qed"])
    eligible["sa_rank"] = rank_good(eligible["sa"])
    eligible["lipinski_rank"] = rank_good(eligible["lipinski"])
    eligible["clash_rank"] = rank_good(-eligible["vdw_overlap_max_local_docked"])
    if eligible["training_novelty"].notna().any():
        eligible["novelty_rank"] = rank_good(eligible["training_novelty"])
        novelty_weight = 0.10
    else:
        eligible["novelty_rank"] = 0.5
        novelty_weight = 0.0

    eligible["balanced_score"] = (
        0.35 * eligible["vina_rank"]
        + 0.20 * eligible["qed_rank"]
        + 0.15 * eligible["sa_rank"]
        + 0.10 * eligible["lipinski_rank"]
        + (0.20 - novelty_weight) * eligible["clash_rank"]
        + novelty_weight * eligible["novelty_rank"]
    )
    objective_columns = [
        "vina_rank",
        "qed_rank",
        "sa_rank",
        "lipinski_rank",
        "clash_rank",
        "novelty_rank",
    ]
    eligible["pareto_front"] = pareto_mask(
        eligible[objective_columns].to_numpy(dtype=float)
    )

    pareto = eligible[eligible["pareto_front"]]
    candidate_a = pareto.sort_values(
        ["balanced_score", "vina", "source_sample_idx"],
        ascending=[False, True, True],
    ).iloc[0]
    mol_a = read_first_sdf(Path(candidate_a["generated_sdf"]))

    pool = eligible.drop(index=candidate_a.name).copy()
    target_median = float(eligible["vina"].median())
    vina_cutoff = min(
        target_median, float(candidate_a["vina"]) + float(config["candidate_b_vina_margin"])
    )
    competitive = pool[pool["vina"].le(vina_cutoff)].copy()
    if competitive.empty:
        competitive = pool.copy()
    competitive["similarity_to_A"] = [
        tanimoto(read_first_sdf(Path(path)), mol_a)
        for path in competitive["generated_sdf"]
    ]
    distinct = competitive[
        competitive["similarity_to_A"].le(
            float(config["candidate_b_max_similarity"])
        )
    ]
    if not distinct.empty:
        competitive = distinct
    candidate_b = competitive.sort_values(
        ["balanced_score", "similarity_to_A", "vina", "source_sample_idx"],
        ascending=[False, True, True, True],
    ).iloc[0]

    candidates["selection_role"] = ""
    candidates.loc[candidate_a.name, "selection_role"] = "Candidate A"
    candidates.loc[candidate_b.name, "selection_role"] = "Candidate B"
    for column in eligible.columns:
        candidates.loc[eligible.index, column] = eligible[column]

    audit_path = directory / "properties" / "case_molecule_selection_audit.csv"
    candidates.sort_values(
        ["mandatory_pass", "balanced_score"], ascending=[False, False]
    ).to_csv(audit_path, index=False)

    selected_rows = []
    for role, row in (("Candidate A", candidate_a), ("Candidate B", candidate_b)):
        role_dir = directory / role.lower().replace(" ", "_")
        role_dir.mkdir(parents=True, exist_ok=True)
        generated_dest = role_dir / f"{role.lower().replace(' ', '_')}_generated.sdf"
        docked_sdf_dest = role_dir / f"{role.lower().replace(' ', '_')}_docked.sdf"
        docked_pdbqt_dest = role_dir / f"{role.lower().replace(' ', '_')}_docked.pdbqt"
        copy_file(Path(row["generated_sdf"]), generated_dest)
        copy_file(Path(row["local_docked_sdf"]), docked_sdf_dest)
        copy_file(Path(row["local_docked_pdbqt"]), docked_pdbqt_dest)
        selected_rows.append(
            {
                "role": role,
                **row.to_dict(),
                "selected_generated_sdf": str(generated_dest),
                "selected_docked_sdf": str(docked_sdf_dest),
                "selected_docked_pdbqt": str(docked_pdbqt_dest),
            }
        )

    selected = pd.DataFrame(selected_rows)
    mol_b = read_first_sdf(Path(candidate_b["generated_sdf"]))
    selected["similarity_A_to_B"] = tanimoto(mol_a, mol_b)
    selected_path = directory / "properties" / "case_molecules.csv"
    selected.to_csv(selected_path, index=False)
    print(f"Eligible molecules: {len(eligible)}/{len(candidates)}")
    print(f"Candidate A: {candidate_a['stem']} | Vina={candidate_a['vina']:.3f}")
    print(
        f"Candidate B: {candidate_b['stem']} | Vina={candidate_b['vina']:.3f} "
        f"| sim(A,B)={tanimoto(mol_a, mol_b):.3f}"
    )
    print(f"Wrote: {selected_path}")


if __name__ == "__main__":
    main()
