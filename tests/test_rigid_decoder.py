import unittest

import numpy as np
from rdkit import Chem

from generate_fragments import RigidAssemblyDecoder


class RigidDecoderIntegrationTests(unittest.TestCase):
    def test_far_fragments_are_aligned_before_bonding(self):
        mol = Chem.RWMol()
        first = mol.AddAtom(Chem.Atom("C"))
        second = mol.AddAtom(Chem.Atom("C"))
        conformer = Chem.Conformer(2)
        conformer.SetAtomPosition(first, (0.0, 0.0, 0.0))
        conformer.SetAtomPosition(second, (10.0, 0.0, 0.0))
        mol.AddConformer(conformer)

        bond_probabilities = np.zeros((2, 2, 5), dtype=np.float64)
        bond_probabilities[:, :, 0] = 0.1
        bond_probabilities[:, :, 1] = 0.9
        connected = RigidAssemblyDecoder()._apply_predicted_bonds(
            mol,
            mol.GetConformer(),
            frag_attachment_atoms={0: [0], 1: [1]},
            frag_count=2,
            final_bonds_b=np.zeros((2, 2), dtype=np.int64),
            final_bond_probs_b=bond_probabilities,
            frag_atom_indices={0: [0], 1: [1]},
            atom_to_frag={0: 0, 1: 1},
        )

        coords = np.asarray(
            [np.asarray(mol.GetConformer().GetAtomPosition(idx)) for idx in range(2)]
        )
        self.assertTrue(connected)
        self.assertEqual(mol.GetNumBonds(), 1)
        self.assertAlmostEqual(float(np.linalg.norm(coords[0] - coords[1])), 1.52, places=5)


if __name__ == "__main__":
    unittest.main()
