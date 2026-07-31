#!/usr/bin/env python3
"""Summarize ligand descriptors and their relationship to Vina/PoseBusters."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from tqdm import tqdm


PROPERTIES = {
    "molecular_weight": [0, 150, 200, 250, 300, 400, 500, np.inf],
    "hbond_acceptors": [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 10.5, np.inf],
    "hbond_donors": [-0.5, 0.5, 1.5, 2.5, 3.5, 5.5, np.inf],
    "rotatable_bonds": [-0.5, 0.5, 1.5, 2.5, 3.5, 5.5, 7.5, 10.5, np.inf],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vina_csv", required=True)
    parser.add_argument("--posebusters_csv")
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--summary_csv", required=True)
    parser.add_argument("--distribution_csv", required=True)
    return parser.parse_args()


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().eq("true")


def read_molecule(sdf_path: str, smiles: str | None) -> Chem.Mol | None:
    path = Path(str(sdf_path)) if pd.notna(sdf_path) and str(sdf_path).strip() else None
    mol = None
    if path is not None and path.exists():
        supplier = Chem.SDMolSupplier(str(path), sanitize=True, removeHs=False)
        mol = next((candidate for candidate in supplier if candidate is not None), None)
    if mol is None and smiles is not None and pd.notna(smiles) and str(smiles).strip():
        mol = Chem.MolFromSmiles(str(smiles), sanitize=True)
    if mol is None:
        return None
    try:
        return Chem.RemoveHs(mol)
    except Exception:
        return mol


def descriptor_row(row: pd.Series) -> dict[str, object]:
    result = row.to_dict()
    try:
        mol = read_molecule(row.get("sdf_path", ""), row.get("smiles"))
        if mol is None:
            raise ValueError("RDKit could not read SDF or SMILES")
        result.update(
            descriptor_success=True,
            descriptor_error="",
            molecular_weight=float(Descriptors.MolWt(mol)),
            hbond_acceptors=int(rdMolDescriptors.CalcNumHBA(mol)),
            hbond_donors=int(rdMolDescriptors.CalcNumHBD(mol)),
            rotatable_bonds=int(rdMolDescriptors.CalcNumRotatableBonds(mol, strict=True)),
            heavy_atom_count=int(mol.GetNumHeavyAtoms()),
        )
    except Exception as exc:
        result.update(
            descriptor_success=False,
            descriptor_error=f"{type(exc).__name__}: {exc}",
            molecular_weight=np.nan,
            hbond_acceptors=np.nan,
            hbond_donors=np.nan,
            rotatable_bonds=np.nan,
            heavy_atom_count=np.nan,
        )
    return result


def add_posebusters_columns(df: pd.DataFrame, path: str | None) -> pd.DataFrame:
    if not path:
        return df
    pb = pd.read_csv(path, low_memory=False)
    columns = [
        "sdf_path",
        "minimum_distance_to_protein",
        "volume_overlap_with_protein",
        "posebusters_error",
    ]
    available = [column for column in columns if column in pb.columns]
    pb = pb[available].drop_duplicates("sdf_path")
    rename = {
        "minimum_distance_to_protein": "pb_minimum_distance_pass",
        "volume_overlap_with_protein": "pb_volume_overlap_pass",
        "posebusters_error": "pb_error",
    }
    return df.merge(pb.rename(columns=rename), on="sdf_path", how="left")


def make_groups(df: pd.DataFrame) -> dict[str, pd.Series]:
    score = pd.to_numeric(df.get("vina_score"), errors="coerce")
    success = bool_series(df["descriptor_success"])
    groups = {
        "all_descriptor_valid": success,
        "vina_scored": success & score.notna(),
        "vina_negative": success & score.notna() & score.lt(0),
        "vina_nonnegative": success & score.notna() & score.ge(0),
        "vina_extreme_positive_ge20": success & score.notna() & score.ge(20),
    }
    if "pb_minimum_distance_pass" in df.columns:
        groups["posebusters_distance_pass"] = success & bool_series(
            df["pb_minimum_distance_pass"]
        )
        groups["posebusters_distance_fail"] = success & df[
            "pb_minimum_distance_pass"
        ].notna() & ~bool_series(df["pb_minimum_distance_pass"])
    if "pb_volume_overlap_pass" in df.columns:
        groups["posebusters_overlap_pass"] = success & bool_series(
            df["pb_volume_overlap_pass"]
        )
        groups["posebusters_overlap_fail"] = success & df[
            "pb_volume_overlap_pass"
        ].notna() & ~bool_series(df["pb_volume_overlap_pass"])
    return groups


def summarize(df: pd.DataFrame, groups: dict[str, pd.Series]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_name, mask in groups.items():
        part = df.loc[mask]
        for prop in PROPERTIES:
            values = pd.to_numeric(part[prop], errors="coerce").dropna()
            row = {
                "group": group_name,
                "property": prop,
                "count": len(values),
                "mean": values.mean(),
                "std": values.std(ddof=1),
                "min": values.min(),
                "p25": values.quantile(0.25),
                "median": values.median(),
                "p75": values.quantile(0.75),
                "max": values.max(),
            }
            score = pd.to_numeric(part.get("vina_score"), errors="coerce")
            paired = pd.DataFrame({"property": part[prop], "vina_score": score}).dropna()
            row["spearman_with_vina"] = (
                paired["property"].corr(paired["vina_score"], method="spearman")
                if len(paired) >= 3
                else np.nan
            )
            rows.append(row)
    return pd.DataFrame(rows)


def distributions(df: pd.DataFrame, groups: dict[str, pd.Series]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_name, mask in groups.items():
        part = df.loc[mask]
        for prop, bins in PROPERTIES.items():
            values = pd.to_numeric(part[prop], errors="coerce").dropna()
            labels = [
                f"[{bins[index]:g}, {bins[index + 1]:g})"
                for index in range(len(bins) - 1)
            ]
            cut = pd.cut(values, bins=bins, labels=labels, right=False, include_lowest=True)
            counts = cut.value_counts(sort=False)
            for bin_name, count in counts.items():
                rows.append(
                    {
                        "group": group_name,
                        "property": prop,
                        "bin": str(bin_name),
                        "count": int(count),
                        "fraction": float(count / len(values)) if len(values) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    raw = pd.read_csv(args.vina_csv, low_memory=False)
    tqdm.pandas(desc="RDKit descriptors")
    records = raw.progress_apply(descriptor_row, axis=1).tolist()
    result = pd.DataFrame(records)
    result["vina_score"] = pd.to_numeric(result.get("vina_score"), errors="coerce")
    result = add_posebusters_columns(result, args.posebusters_csv)

    groups = make_groups(result)
    summary = summarize(result, groups)
    distribution = distributions(result, groups)

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)
    distribution.to_csv(args.distribution_csv, index=False)

    valid = bool_series(result["descriptor_success"])
    scored = valid & result["vina_score"].notna()
    print(f"Rows: {len(result)}")
    print(f"Descriptor success: {valid.sum()} ({valid.mean():.2%})")
    print(f"Vina-scored descriptor rows: {scored.sum()}")
    print(f"Molecule-level output: {args.output_csv}")
    print(f"Summary: {args.summary_csv}")
    print(f"Distributions: {args.distribution_csv}")


if __name__ == "__main__":
    main()
