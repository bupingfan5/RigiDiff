from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from case_utils import CASE_ROOT, load_config


def target_dir(config: dict, target_id: int) -> Path:
    if target_id == int(config["primary_target_id"]):
        return CASE_ROOT / config["primary_target_name"]
    return CASE_ROOT / f"target_{target_id}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-id", type=int, default=48)
    args = parser.parse_args()
    config = load_config()
    directory = target_dir(config, args.target_id)

    checks = []

    def check(name, passed, detail, required=True):
        checks.append(
            {
                "check": name,
                "status": "PASS" if passed else ("FAIL" if required else "WARN"),
                "detail": detail,
            }
        )

    expected = [
        directory / "properties" / "case_pose_manifest.csv",
        directory / "properties" / "case_local_docking.csv",
        directory / "properties" / "case_local_docking_protocol.json",
        directory / "properties" / "case_clashes.csv",
        directory / "properties" / "case_molecules.csv",
        directory / "interactions" / "case_interactions.csv",
        directory / "tables" / "case_study_table.csv",
        directory / "tables" / "case_study_table.tex",
        directory / "figures" / "case_study.pml",
        directory / "case_study_results.json",
    ]
    for path in expected:
        check(f"file:{path.name}", path.exists() and path.stat().st_size > 0, str(path))

    selected_path = directory / "properties" / "case_molecules.csv"
    if selected_path.exists():
        selected = pd.read_csv(selected_path)
        check("two_selected_molecules", len(selected) == 2, f"rows={len(selected)}")
        check(
            "selection_roles",
            set(selected["role"]) == {"Candidate A", "Candidate B"},
            str(sorted(selected["role"].astype(str).tolist())),
        )
        check(
            "distinct_candidates",
            selected["stem"].nunique() == 2,
            ";".join(selected["stem"].astype(str)),
        )
        check(
            "successful_vina",
            pd.to_numeric(selected["vina"], errors="coerce").notna().all(),
            ";".join(selected["vina"].astype(str)),
        )
        check(
            "no_severe_local_docked_clash",
            pd.to_numeric(
                selected["severe_vdw_clash_count_local_docked"], errors="coerce"
            ).fillna(1).eq(0).all(),
            ";".join(selected["severe_vdw_clash_count_local_docked"].astype(str)),
        )
        check(
            "local_docked_pocket_occupancy",
            pd.to_numeric(
                selected["pocket_occupancy_fraction_local_docked"], errors="coerce"
            ).ge(float(config["minimum_pocket_occupancy"])).all(),
            ";".join(selected["pocket_occupancy_fraction_local_docked"].astype(str)),
        )

    interaction_path = directory / "interactions" / "case_interactions.csv"
    if interaction_path.exists():
        interactions = pd.read_csv(interaction_path)
        check(
            "interaction_records",
            len(interactions) > 0,
            f"rows={len(interactions)}; backend=internal_geometry_v1",
        )

    fragment_status = directory / "fragment_mapping_status.json"
    check(
        "fragment_provenance",
        False,
        "Atom-to-fragment provenance is not available; fragment panel remains pending.",
        required=False,
    )
    rendered = list((directory / "figures").glob("panel_*.png"))
    check(
        "rendered_figures",
        len(rendered) >= 3,
        f"rendered={len(rendered)}",
        required=False,
    )

    failed = [item for item in checks if item["status"] == "FAIL"]
    report = {
        "target_id": args.target_id,
        "verification_status": "VERIFIED" if not failed else "INCOMPLETE",
        "checks": checks,
        "required_failures": len(failed),
    }
    (directory / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    lines = [
        "# Case Study Validation Report",
        "",
        f"- Target ID: {args.target_id}",
        f"- Verification status: {report['verification_status']}",
        f"- Required failures: {len(failed)}",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    lines.extend(
        f"| {item['check']} | {item['status']} | {item['detail']} |"
        for item in checks
    )
    (directory / "validation_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
