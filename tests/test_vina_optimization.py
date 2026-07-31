import tempfile
import unittest
from pathlib import Path

import lmdb
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from evaluate.vina_eval import valid_split_keys
from src.fragment_vocab import _fragment_instances_from_mol, fit_fixed_template_pose
from src.generation_utils import (
    FragmentCountSampler,
    attachment_constrained_tree_plans,
    rotate_points_about_axis,
    rotation_matrix_align_vectors,
)
from src.pocket_geometry import (
    bound_generated_center_shift,
    center_ligand_and_pocket,
    pocket_clash_terms,
    retain_global_clash_gradient,
    update_guided_center_offset,
    late_clash_guidance_weight,
    rigid_body_vdw_relax,
)
from src.pocket_pose_value_net import PocketPoseValueNetwork


class CoordinateFrameTests(unittest.TestCase):
    def test_joint_translation_keeps_centered_inputs_identical(self):
        ligand = torch.tensor([[[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]])
        ligand_mask = torch.ones(1, 2, 1)
        pocket = torch.tensor([[[5.0, 4.0, 3.0], [6.0, 4.0, 2.0]]])
        pocket_mask = torch.ones(1, 2, 1)
        shift = torch.tensor([[[20.0, -7.0, 11.0]]])

        ligand_a, pocket_a, _ = center_ligand_and_pocket(
            ligand, ligand_mask, pocket, pocket_mask
        )
        ligand_b, pocket_b, _ = center_ligand_and_pocket(
            ligand + shift, ligand_mask, pocket + shift, pocket_mask
        )
        torch.testing.assert_close(ligand_a, ligand_b)
        torch.testing.assert_close(pocket_a, pocket_b)


class PocketClashTests(unittest.TestCase):
    def test_rigid_body_relaxation_reduces_clashes_without_internal_distortion(self):
        ligand = np.asarray([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
        pocket = np.asarray([[0.2, 0.0, 0.0]])
        relaxed, diagnostics = rigid_body_vdw_relax(
            ligand,
            [6, 6],
            pocket,
            [6],
            steps=100,
            max_com_shift=4.0,
        )
        self.assertLess(
            diagnostics["after"]["clash_count"],
            diagnostics["before"]["clash_count"],
        )
        self.assertAlmostEqual(
            float(np.linalg.norm(relaxed[0] - relaxed[1])),
            float(np.linalg.norm(ligand[0] - ligand[1])),
            places=6,
        )
        self.assertLessEqual(diagnostics["com_shift"], 4.0 + 1e-6)

    def test_multistart_relaxation_is_deterministic_and_rigid(self):
        ligand = np.asarray(
            [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [0.0, 1.4, 0.0]],
            dtype=np.float64,
        )
        pocket = np.asarray(
            [[0.2, 0.1, 0.0], [1.1, 0.2, 0.0], [0.1, 1.1, 0.0]],
            dtype=np.float64,
        )
        kwargs = dict(
            steps=40,
            max_com_shift=3.0,
            multistart_orientations=6,
            multistart_refine=4,
        )
        first, first_diag = rigid_body_vdw_relax(
            ligand, [6, 6, 6], pocket, [6, 6, 6], **kwargs
        )
        second, second_diag = rigid_body_vdw_relax(
            ligand, [6, 6, 6], pocket, [6, 6, 6], **kwargs
        )
        np.testing.assert_allclose(first, second, atol=1e-10)
        self.assertEqual(first_diag["orientation_count"], 6)
        self.assertEqual(first_diag["after"], second_diag["after"])
        self.assertLessEqual(
            first_diag["after"]["overlap_energy"],
            first_diag["before"]["overlap_energy"],
        )
        np.testing.assert_allclose(
            np.linalg.norm(first[:, None, :] - first[None, :, :], axis=-1),
            np.linalg.norm(ligand[:, None, :] - ligand[None, :, :], axis=-1),
            atol=1e-6,
        )
        self.assertLessEqual(first_diag["com_shift"], 3.0 + 1e-6)

    def test_clash_guidance_decays_over_reverse_sampling(self):
        self.assertAlmostEqual(late_clash_guidance_weight(100, 0, 100), 1.0)
        self.assertAlmostEqual(late_clash_guidance_weight(50, 0, 100), 0.5)
        self.assertAlmostEqual(late_clash_guidance_weight(0, 0, 100), 0.1)
        self.assertAlmostEqual(late_clash_guidance_weight(101, 0, 100), 0.0)

    def test_overlap_has_positive_loss_and_finite_gradient(self):
        ligand = torch.tensor([[[0.0, 0.0, 0.0]]], requires_grad=True)
        ligand_mask = torch.ones(1, 1, dtype=torch.bool)
        ligand_radii = torch.full((1, 1), 1.7)
        pocket = torch.tensor([[[0.2, 0.0, 0.0]]])
        pocket_mask = torch.ones(1, 1, 1)
        pocket_element = torch.tensor([[6]])

        loss, clash_count, min_distance = pocket_clash_terms(
            ligand, ligand_mask, ligand_radii, pocket, pocket_mask, pocket_element
        )
        self.assertGreater(float(loss), 0.0)
        self.assertEqual(float(clash_count[0]), 1.0)
        self.assertAlmostEqual(float(min_distance[0]), 0.2, places=5)
        loss.backward()
        self.assertTrue(torch.isfinite(ligand.grad).all())

    def test_non_overlap_has_zero_loss(self):
        ligand = torch.tensor([[[0.0, 0.0, 0.0]]])
        loss, clash_count, _ = pocket_clash_terms(
            ligand,
            torch.ones(1, 1, dtype=torch.bool),
            torch.full((1, 1), 1.7),
            torch.tensor([[[10.0, 0.0, 0.0]]]),
            torch.ones(1, 1, 1),
            torch.tensor([[6]]),
        )
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(float(clash_count[0]), 0.0)

    def test_guidance_gradient_can_be_made_batch_size_invariant(self):
        def gradient_for_batch(batch_size):
            ligand = torch.tensor([[[0.0, 0.0, 0.0]]]).repeat(batch_size, 1, 1)
            ligand.requires_grad_(True)
            loss, _, _ = pocket_clash_terms(
                ligand,
                torch.ones(batch_size, 1, dtype=torch.bool),
                torch.full((batch_size, 1), 1.7),
                torch.tensor([[[0.2, 0.0, 0.0]]]).repeat(batch_size, 1, 1),
                torch.ones(batch_size, 1, 1),
                torch.full((batch_size, 1), 6, dtype=torch.long),
            )
            return torch.autograd.grad(loss * batch_size, ligand)[0][0]

        torch.testing.assert_close(gradient_for_batch(1), gradient_for_batch(4))

    def test_single_fragment_retains_global_translation_gradient(self):
        gradient = torch.tensor([[[2.0, -4.0, 1.0]]])
        mask = torch.ones(1, 1, 1)
        retained = retain_global_clash_gradient(gradient, mask, global_fraction=0.25)
        self.assertTrue(torch.allclose(retained, gradient * 0.25))

    def test_multi_fragment_global_mean_is_scaled_not_removed(self):
        gradient = torch.tensor([[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]])
        mask = torch.ones(1, 2, 1)
        retained = retain_global_clash_gradient(gradient, mask, global_fraction=0.25)
        self.assertTrue(torch.allclose(retained.mean(dim=1), gradient.mean(dim=1) * 0.25))

    def test_generated_center_shift_is_bounded_not_deleted(self):
        coords = torch.tensor([[[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]]])
        mask = torch.ones(1, 2, 1)
        bounded, center = bound_generated_center_shift(coords, mask, max_shift=1.0)
        self.assertAlmostEqual(float(center.norm()), 1.0, places=6)
        self.assertTrue(torch.allclose(bounded.mean(dim=1), center.squeeze(1)))

    def test_zero_center_limit_preserves_legacy_projection(self):
        coords = torch.tensor([[[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]]])
        bounded, _ = bound_generated_center_shift(coords, torch.ones(1, 2, 1), max_shift=0.0)
        self.assertTrue(torch.allclose(bounded.mean(dim=1), torch.zeros(1, 3)))

    def test_guided_center_offset_descends_energy_and_is_bounded(self):
        offset = torch.zeros(1, 1, 3)
        gradient = torch.tensor([[[2.0, 0.0, 0.0]]])
        updated = update_guided_center_offset(offset, gradient, step_size=0.4, max_shift=0.5)
        self.assertTrue(torch.allclose(updated, torch.tensor([[[-0.5, 0.0, 0.0]]])))


class RigidAssemblyGeometryTests(unittest.TestCase):
    def test_attachment_constrained_tree_avoids_single_site_bridge(self):
        edges = [
            {"frag_i": 0, "frag_j": 1, "edge_probability": 0.90,
             "choices": [{"atom_i": 0, "atom_j": 2, "bond_label": 1}]},
            {"frag_i": 1, "frag_j": 2, "edge_probability": 0.85,
             "choices": [{"atom_i": 2, "atom_j": 3, "bond_label": 1}]},
            {"frag_i": 2, "frag_j": 3, "edge_probability": 0.80,
             "choices": [{"atom_i": 4, "atom_j": 5, "bond_label": 1}]},
            {"frag_i": 0, "frag_j": 2, "edge_probability": 0.70,
             "choices": [{"atom_i": 1, "atom_j": 3, "bond_label": 1}]},
        ]
        first = attachment_constrained_tree_plans(4, edges, beam_width=32, max_plans=4)
        second = attachment_constrained_tree_plans(4, edges, beam_width=32, max_plans=4)
        self.assertTrue(first)
        self.assertEqual(first, second)
        plan = first[0]
        self.assertEqual(len(plan), 3)
        used_atoms = [
            atom
            for edge in plan
            for atom in (edge["atom_i"], edge["atom_j"])
        ]
        self.assertEqual(len(used_atoms), len(set(used_atoms)))
        selected_pairs = {
            tuple(sorted((edge["frag_i"], edge["frag_j"]))) for edge in plan
        }
        self.assertNotIn((1, 2), selected_pairs)

    def test_fragment_instances_drop_explicit_hydrogens(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("NC(N)=N"))
        self.assertEqual(AllChem.EmbedMolecule(mol, randomSeed=42), 0)
        fragments, _, _ = _fragment_instances_from_mol(mol)
        self.assertTrue(fragments)
        self.assertTrue(all("H" not in fragment["atom_symbols"] for fragment in fragments))

    def test_fixed_template_pose_recovers_known_rigid_transform(self):
        template = np.asarray(
            [[0.0, 0.0, 0.0], [1.2, 0.1, 0.0], [0.2, 1.1, 0.3], [-0.3, 0.2, 0.9]],
            dtype=np.float64,
        )
        angle = np.deg2rad(63.0)
        rotation = np.asarray(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        translation = np.asarray([4.0, -2.5, 1.3], dtype=np.float64)
        target = template @ rotation.T + translation

        fitted_translation, fitted_rotation, rmsd = fit_fixed_template_pose(template, target)
        reconstructed = template @ fitted_rotation.T + fitted_translation

        self.assertLess(rmsd, 1e-6)
        self.assertAlmostEqual(float(np.linalg.det(fitted_rotation)), 1.0, places=6)
        np.testing.assert_allclose(reconstructed, target, atol=1e-6)

    def test_fixed_template_pose_handles_single_atom(self):
        translation, rotation, rmsd = fit_fixed_template_pose(
            np.asarray([[0.5, -0.2, 1.0]]),
            np.asarray([[3.0, 4.0, 5.0]]),
        )
        np.testing.assert_allclose(rotation, np.eye(3), atol=1e-7)
        np.testing.assert_allclose(translation, [2.5, 4.2, 4.0], atol=1e-7)
        self.assertAlmostEqual(rmsd, 0.0)

    def test_vector_alignment_handles_opposite_vectors(self):
        rotation = rotation_matrix_align_vectors([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0])
        aligned = rotation @ torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64).numpy()
        self.assertTrue(torch.allclose(
            torch.as_tensor(aligned),
            torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float64),
            atol=1e-7,
        ))

    def test_dihedral_rotation_preserves_axis_points(self):
        points = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=torch.float64).numpy()
        rotated = rotate_points_about_axis(
            points,
            origin=[0.0, 0.0, 0.0],
            axis=[1.0, 0.0, 0.0],
            angle_radians=torch.pi / 2,
        )
        self.assertTrue(torch.allclose(
            torch.as_tensor(rotated[0]),
            torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
            atol=1e-7,
        ))
        self.assertTrue(torch.allclose(
            torch.as_tensor(rotated[1]),
            torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64),
            atol=1e-7,
        ))


class FragmentCountTests(unittest.TestCase):
    def test_range_mode_samples_the_range(self):
        sampler = FragmentCountSampler(mode="range", legacy_value="3,7", max_count=10, seed=42)
        values = sampler.sample(100)
        self.assertGreater(len(set(values)), 1)
        self.assertGreaterEqual(min(values), 3)
        self.assertLessEqual(max(values), 7)

    def test_empirical_mode_is_reproducible_and_bounded(self):
        first = FragmentCountSampler(mode="empirical", max_count=10, seed=2025).sample(50)
        second = FragmentCountSampler(mode="empirical", max_count=10, seed=2025).sample(50)
        self.assertEqual(first, second)
        self.assertTrue(all(1 <= value <= 10 for value in first))


class PocketValueNetworkTests(unittest.TestCase):
    def test_score_is_differentiable_with_respect_to_pose(self):
        model = PocketPoseValueNetwork(ligand_feature_dim=4, hidden_dim=16, num_rbf=8)
        ligand_pos = torch.randn(2, 3, 3, requires_grad=True)
        score, negative_logit = model(
            ligand_pos,
            torch.softmax(torch.randn(2, 3, 4), dim=-1),
            torch.ones(2, 3, 1),
            torch.randn(2, 5, 3),
            torch.full((2, 5), 6),
            torch.ones(2, 5, 1),
        )
        (score.sum() + negative_logit.sum()).backward()
        self.assertEqual(tuple(score.shape), (2,))
        self.assertTrue(torch.isfinite(ligand_pos.grad).all())


class SplitIdentityTests(unittest.TestCase):
    def test_missing_split_key_is_not_silently_compressed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lmdb_path = root / "test.lmdb"
            env = lmdb.open(str(lmdb_path), subdir=False, map_size=1024 * 1024)
            with env.begin(write=True) as txn:
                txn.put(b"10", b"present")
            env.close()
            split_path = root / "split.pt"
            torch.save({"test": [10, 20]}, split_path)
            with self.assertRaisesRegex(RuntimeError, "missing keys:.*20"):
                valid_split_keys(split_path, lmdb_path, "test")


if __name__ == "__main__":
    unittest.main()
