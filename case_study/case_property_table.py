from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from case_utils import CASE_ROOT, load_config, molecular_properties, read_first_sdf, tanimoto


def target_dir(config: dict, target_id: int) -> Path:
    if target_id == int(config["primary_target_id"]):
        return CASE_ROOT / config["primary_target_name"]
    return CASE_ROOT / f"target_{target_id}"


def fmt(value, digits=3):
    if value is None or pd.isna(value):
        return "--"
    return f"{float(value):.{digits}f}"


def count_interactions(frame, ligand, pattern):
    subset = frame[
        frame["ligand"].eq(ligand)
        & frame["interaction_type"].str.contains(pattern, regex=True)
    ]
    return int(len(subset))


def key_contacts(frame, ligand, known):
    subset = frame[frame["ligand"].eq(ligand)]
    contacts = {
        int(value)
        for value in pd.to_numeric(subset["residue_number"], errors="coerce").dropna()
    }
    return "; ".join(map(str, sorted(contacts.intersection(known)))) or "--"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-id", type=int, default=48)
    args = parser.parse_args()
    config = load_config()
    directory = target_dir(config, args.target_id)
    selected = pd.read_csv(directory / "properties" / "case_molecules.csv")
    interactions = pd.read_csv(directory / "interactions" / "case_interactions.csv")
    clashes = pd.read_csv(directory / "properties" / "case_clashes_long.csv")

    reference_path = directory / "reference" / "crossdocked_reference.sdf"
    reference = read_first_sdf(reference_path)
    reference_props = molecular_properties(reference)
    known_residues = {
        125, 205, 206, 357, 547, 630, 631, 656, 659, 662, 666, 708, 710, 740
    }

    ligands = {
        "Reference ligand": {
            **reference_props,
            "vina": np.nan,
            "similarity_to_crossdocked_524": 1.0,
            "clash_role": "crossdocked_reference",
            "interaction_name": "Reference 524",
            "source_id": "2QT9:524",
        }
    }
    for role in ("Candidate A", "Candidate B"):
        row = selected[selected["role"].eq(role)].iloc[0].to_dict()
        ligands[role] = {
            **row,
            "clash_role": row["stem"],
            "interaction_name": role,
            "source_id": row["stem"],
        }

    table_rows = []
    for label, info in ligands.items():
        clash = clashes[clashes["stem"].eq(info["clash_role"])]
        if label != "Reference ligand":
            clash = clash[clash["pose_kind"].eq("local_docked")]
        clash_row = clash.iloc[0] if not clash.empty else {}
        table_rows.append(
            {
                "ligand": label,
                "source_id": info["source_id"],
                "vina": info.get("vina", np.nan),
                "qed": info.get("qed", np.nan),
                "sa": info.get("sa", np.nan),
                "sa_raw": info.get("sa_raw", np.nan),
                "logp": info.get("logp", np.nan),
                "lipinski": info.get("lipinski", np.nan),
                "mw": info.get("mw", np.nan),
                "tpsa": info.get("tpsa", np.nan),
                "hbd": info.get("hbd", np.nan),
                "hba": info.get("hba", np.nan),
                "rotatable_bonds": info.get("rotatable_bonds", np.nan),
                "ring_count": info.get("ring_count", np.nan),
                "aromatic_ring_count": info.get("aromatic_ring_count", np.nan),
                "hydrogen_bonds": count_interactions(
                    interactions, info["interaction_name"], "hydrogen_bond"
                ),
                "hydrophobic_contacts": count_interactions(
                    interactions, info["interaction_name"], "^hydrophobic$"
                ),
                "pi_related_interactions": count_interactions(
                    interactions, info["interaction_name"], "^pi_"
                ),
                "key_residue_contacts": key_contacts(
                    interactions, info["interaction_name"], known_residues
                ),
                "vdw_clash_count": clash_row.get("vdw_clash_count", np.nan),
                "severe_vdw_clash_count": clash_row.get(
                    "severe_vdw_clash_count", np.nan
                ),
                "vdw_overlap_max": clash_row.get("vdw_overlap_max", np.nan),
                "minimum_relative_distance": clash_row.get(
                    "minimum_relative_distance", np.nan
                ),
                "pocket_occupancy_fraction": clash_row.get(
                    "pocket_occupancy_fraction", np.nan
                ),
                "similarity_to_crossdocked_524": info.get(
                    "similarity_to_crossdocked_524", np.nan
                ),
            }
        )

    results = pd.DataFrame(table_rows)
    table_dir = directory / "tables"
    table_dir.mkdir(exist_ok=True)
    results.to_csv(table_dir / "case_study_table.csv", index=False)

    by_ligand = results.set_index("ligand")
    columns = ["Reference ligand", "Candidate A", "Candidate B"]
    display_rows = [
        ("Vina Score (kcal/mol)", "vina", 3),
        ("QED", "qed", 3),
        ("SA $\\uparrow$", "sa", 3),
        ("LogP", "logp", 3),
        ("Lipinski", "lipinski", 0),
        ("MW (Da)", "mw", 1),
        ("TPSA ($\\AA^2$)", "tpsa", 1),
        ("HBD", "hbd", 0),
        ("HBA", "hba", 0),
        ("Rotatable bonds", "rotatable_bonds", 0),
        ("Rings", "ring_count", 0),
        ("H-bond candidates", "hydrogen_bonds", 0),
        ("Hydrophobic contacts", "hydrophobic_contacts", 0),
        ("$\\pi$-related contacts", "pi_related_interactions", 0),
        ("vdW clashes", "vdw_clash_count", 0),
        ("Severe clashes", "severe_vdw_clash_count", 0),
        ("Similarity to ref. 524", "similarity_to_crossdocked_524", 3),
    ]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Molecular and structural properties for the DPP4 case study. "
        "SA is normalized to $[0,1]$ (higher is better).}",
        "\\label{tab:dpp4_case}",
        "\\small",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Property & Reference & Cand. A & Cand. B \\\\",
        "\\midrule",
    ]
    for label, key, digits in display_rows:
        values = [fmt(by_ligand.loc[column, key], digits) for column in columns]
        lines.append(f"{label} & " + " & ".join(values) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    (table_dir / "case_study_table.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    def json_value(value):
        if pd.isna(value):
            return None
        return value.item() if isinstance(value, np.generic) else value

    result_json = {
        row["ligand"]: {
            key: json_value(value)
            for key, value in row.items()
            if key != "ligand"
        }
        for row in table_rows
    }
    (directory / "case_study_results.json").write_text(
        json.dumps(result_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(results.to_string(index=False))
    print(f"Wrote table files to {table_dir}")


if __name__ == "__main__":
    main()
