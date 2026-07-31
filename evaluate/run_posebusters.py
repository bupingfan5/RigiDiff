import argparse
from pathlib import Path

import pandas as pd
from posebusters import PoseBusters


def as_true(series):
    return series.astype(str).str.lower().eq("true")


def main():
    parser = argparse.ArgumentParser(description="Run PoseBusters on scored generated poses.")
    parser.add_argument("--vina_csv", required=True)
    parser.add_argument("--data_root", default="data/crossdocked_pocket10")
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--project_root", default=".")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = root / data_root

    vina = pd.read_csv(args.vina_csv)
    selected = vina[as_true(vina["rdkit_valid"]) & vina["vina_score"].notna()].copy()
    buster = PoseBusters(config="dock")
    reports = []

    for _, row in selected.iterrows():
        sdf = Path(row["sdf_path"])
        if not sdf.is_absolute():
            sdf = root / sdf
        protein = data_root / row["protein_filename"]
        metadata = {
            "target_id": row.get("target_id"),
            "lmdb_key": row.get("lmdb_key"),
            "vina_score": row.get("vina_score"),
            "sdf_path": str(sdf),
            "protein_filename": row.get("protein_filename"),
        }
        try:
            report = buster.bust(
                mol_pred=[sdf],
                mol_true=None,
                mol_cond=protein,
            ).reset_index()
            for key, value in metadata.items():
                report[key] = value
            report["posebusters_error"] = ""
        except Exception as exc:
            report = pd.DataFrame([{**metadata, "posebusters_error": str(exc)}])
        reports.append(report)

    if not reports:
        raise RuntimeError("No RDKit-valid, Vina-scored molecules were available for PoseBusters")

    result = pd.concat(reports, ignore_index=True, sort=False)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    failures = result["posebusters_error"].fillna("").astype(str).str.len().gt(0).sum()
    print(f"PoseBusters checked: {len(result)}")
    print(f"PoseBusters errors: {int(failures)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
