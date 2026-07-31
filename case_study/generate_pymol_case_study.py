from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from case_utils import CASE_ROOT, load_config


def q(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


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
    selected = pd.read_csv(directory / "properties" / "case_molecules.csv")
    interactions = pd.read_csv(directory / "interactions" / "case_interactions.csv")

    candidate_a = Path(
        selected.loc[selected["role"].eq("Candidate A"), "selected_docked_sdf"].iloc[0]
    )
    candidate_b = Path(
        selected.loc[selected["role"].eq("Candidate B"), "selected_docked_sdf"].iloc[0]
    )
    a_contacts = interactions[interactions["ligand"].eq("Candidate A")]
    important = a_contacts[
        ~a_contacts["interaction_type"].eq("hydrophobic")
    ].sort_values("distance")
    all_contact_terms = []
    for row in important.itertuples():
        key = (str(row.chain), int(row.residue_number))
        if key not in all_contact_terms:
            all_contact_terms.append(key)
    preferred_residues = (205, 662)
    contact_terms = [
        item for item in all_contact_terms if item[1] in preferred_residues
    ][:2]
    if len(contact_terms) < 2:
        contact_terms.extend(
            item for item in all_contact_terms
            if item not in contact_terms
        )
        contact_terms = contact_terms[:2]
    selection = " or ".join(
        f"(chain {chain} and resi {resi})" for chain, resi in contact_terms
    ) or "none"
    label_offsets = [(2.0, 2.0, 0.0), (-2.0, -1.5, 0.0)]
    label_commands = []
    for index, ((chain, residue), offset) in enumerate(
        zip(contact_terms, label_offsets)
    ):
        label_commands.extend(
            [
                f"select contact_label_{index}, protein and chain {chain} and resi {residue} and name CA",
                f'label contact_label_{index}, "%s%s" % (resn,resi)',
                f"set label_position, [{offset[0]},{offset[1]},{offset[2]}], contact_label_{index}",
            ]
        )
    label_block = "\n".join(label_commands)

    figure_dir = directory / "figures"
    figure_dir.mkdir(exist_ok=True)
    script = f"""reinitialize
bg_color white
set ray_opaque_background, off
set antialias, 2
set cartoon_transparency, 0.10
set surface_transparency, 0.65
set stick_radius, 0.18
set dash_radius, 0.07
set dash_gap, 0.25
set label_size, 11
set label_color, black

load {q(directory / 'reference' / 'pocket.pdb')}, protein
load {q(directory / 'reference' / 'crossdocked_reference.sdf')}, reference
load {q(candidate_a)}, candidate_A
load {q(candidate_b)}, candidate_B

hide everything
show cartoon, protein
color gray80, protein
show sticks, reference or candidate_A or candidate_B
color marine, reference and elem C
color orange, candidate_A and elem C
color green, candidate_B and elem C
util.cnc reference
util.cnc candidate_A
util.cnc candidate_B

select pocket_A, byres (protein within 4.5 of candidate_A)
select contacts_A, protein and ({selection})
show sticks, pocket_A
color gray60, pocket_A and elem C
show sticks, contacts_A
{label_block}

disable candidate_A
disable candidate_B
enable reference
select pocket_ref, byres (protein within 4.5 of reference)
show sticks, pocket_ref
orient reference
zoom reference, 12
ray 2400, 1800
png {q(figure_dir / 'panel_reference.png')}, dpi=300

disable reference
enable candidate_A
show sticks, pocket_A or contacts_A
dist polar_A, candidate_A, contacts_A, mode=2
color yellow, polar_A
orient candidate_A
zoom candidate_A, 12
ray 2400, 1800
png {q(figure_dir / 'panel_candidate_A.png')}, dpi=300

enable reference
disable polar_A
hide labels, protein
orient reference or candidate_A
zoom reference or candidate_A, 10
ray 2400, 1800
png {q(figure_dir / 'panel_overlay.png')}, dpi=300

disable reference
disable candidate_A
enable candidate_B
select pocket_B, byres (protein within 4.5 of candidate_B)
show sticks, pocket_B
orient candidate_B
zoom candidate_B, 12
ray 2400, 1800
png {q(figure_dir / 'panel_candidate_B.png')}, dpi=300

save {q(figure_dir / 'case_study_session.pse')}
quit
"""
    pml = figure_dir / "case_study.pml"
    pml.write_text(script, encoding="utf-8")

    fragment_status = {
        "status": "missing_generation_provenance",
        "reason": (
            "Existing SDF and generation_attempts.csv do not contain atom-to-fragment "
            "membership, vocabulary IDs, centers, rotations, or attachment atom mappings."
        ),
        "required_action": (
            "Re-run the selected generation indices with unchanged checkpoint, seed, "
            "temperature, and guidance settings while serializing fragment provenance."
        ),
        "do_not_use": "Retrospective BRICS decomposition as if it were RigiDiff provenance.",
    }
    (directory / "fragment_mapping_status.json").write_text(
        json.dumps(fragment_status, indent=2), encoding="utf-8"
    )
    print(f"Wrote PyMOL script: {pml}")


if __name__ == "__main__":
    main()
