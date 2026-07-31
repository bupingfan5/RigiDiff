from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from case_utils import (
    CASE_ROOT,
    crossdocked_reference_path,
    diversity_from_smiles,
    load_config,
    load_vina,
    numeric,
    parse_bool,
    parse_chain_from_pocket,
    project_path,
    rank_percentile,
    sa_paper_from_raw,
)


def first_nonempty(series: pd.Series, default=""):
    values = series.dropna().astype(str)
    values = values[values.str.len() > 0]
    return values.iloc[0] if not values.empty else default


def target_summary(config: dict) -> pd.DataFrame:
    vina = load_vina(config)
    standard_path = project_path(config, "standard_metrics_csv")
    standard = pd.read_csv(standard_path) if standard_path.exists() else pd.DataFrame()
    if not standard.empty:
        standard["stem"] = standard["file"].str.extract(r"(target_\d+_sample_\d+)")
        standard["target_id"] = numeric(
            standard["stem"].str.extract(r"target_(\d+)_")[0]
        ).astype("Int64")

    rows = []
    for target_id, group in vina.groupby("target_id", dropna=True):
        target_id = int(target_id)
        docked = group[group["vina"].notna()].copy()
        valid = group[group["rdkit_valid"].map(parse_bool)].copy()
        pocket_file = first_nonempty(group["protein_filename"])
        pdb_id = first_nonempty(group["pdb_id"]).upper()
        reference_path = (
            crossdocked_reference_path(config, pocket_file)
            if pocket_file
            else Path()
        )

        matching_standard = (
            standard[standard["target_id"].eq(target_id)]
            if not standard.empty
            else pd.DataFrame()
        )
        if not matching_standard.empty:
            mean_sa = numeric(matching_standard["SA_Score"]).mean()
            mean_lipinski = numeric(matching_standard["Lipinski"]).mean()
        else:
            mean_sa = valid["sa"].dropna().map(sa_paper_from_raw).mean()
            mean_lipinski = np.nan

        rows.append(
            {
                "target_id": target_id,
                "target_name": first_nonempty(group["target_name"], f"target_{target_id}"),
                "pdb_id": pdb_id,
                "chain": parse_chain_from_pocket(pocket_file),
                "lmdb_key": first_nonempty(group["lmdb_key"]),
                "pocket_file": pocket_file,
                "reference_ligand": str(reference_path) if pocket_file else "",
                "generated_count": int(len(group)),
                "valid_count": int(len(valid)),
                "docking_success_count": int(len(docked)),
                "docking_success_rate": float(len(docked) / max(len(group), 1)),
                "median_vina": float(docked["vina"].median()) if not docked.empty else np.nan,
                "best_vina": float(docked["vina"].min()) if not docked.empty else np.nan,
                "mean_qed": float(valid["qed"].mean()) if not valid.empty else np.nan,
                "mean_sa": float(mean_sa) if pd.notna(mean_sa) else np.nan,
                "mean_raw_sa": float(valid["sa"].mean()) if not valid.empty else np.nan,
                "mean_logp": float(valid["logp"].mean()) if not valid.empty else np.nan,
                "mean_lipinski": float(mean_lipinski) if pd.notna(mean_lipinski) else np.nan,
                "diversity": diversity_from_smiles(valid["smiles"]),
                "uniqueness": float(valid["smiles"].nunique() / max(len(valid), 1)),
            }
        )

    summary = pd.DataFrame(rows)
    metadata = pd.read_csv(project_path(config, "metadata_csv"))
    metadata["pdb_id"] = metadata["pdb_id"].astype(str).str.upper()
    summary = summary.merge(
        metadata,
        on=["target_id", "pdb_id", "chain"],
        how="left",
        suffixes=("", "_metadata"),
    )

    for column in ("disease_score", "structure_score", "residue_evidence_score"):
        summary[column] = numeric(summary[column]).fillna(0.0)

    summary["vina_quality"] = rank_percentile(-summary["median_vina"])
    summary["representative_score"] = (
        1.0 - 2.0 * (summary["vina_quality"] - 0.5).abs()
    ).clip(0.0, 1.0)
    summary["success_score"] = rank_percentile(summary["docking_success_rate"])
    summary["property_score"] = rank_percentile(summary["mean_qed"])
    summary["eligible"] = (
        summary["disease"].notna()
        & summary["protein_target"].notna()
        & summary["native_ligand"].notna()
        & summary["known_key_residues"].notna()
        & summary["docking_success_count"].ge(30)
    )
    summary["target_score"] = (
        0.30 * summary["disease_score"]
        + 0.25 * summary["structure_score"]
        + 0.15 * summary["residue_evidence_score"]
        + 0.10 * summary["success_score"]
        + 0.10 * summary["representative_score"]
        + 0.10 * summary["property_score"]
    )
    summary.loc[~summary["eligible"], "target_score"] = np.nan
    return summary.sort_values(
        ["eligible", "target_score", "target_id"],
        ascending=[False, False, True],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(CASE_ROOT / "target_selection" / "case_target_candidates.csv"),
    )
    args = parser.parse_args()
    config = load_config()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = target_summary(config)
    summary.to_csv(output, index=False)
    print(f"Wrote {len(summary)} targets to {output}")
    print(
        summary.loc[
            summary["eligible"],
            [
                "target_id",
                "pdb_id",
                "protein_target",
                "disease",
                "valid_count",
                "median_vina",
                "best_vina",
                "mean_qed",
                "mean_sa",
                "diversity",
                "target_score",
            ],
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
