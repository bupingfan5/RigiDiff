import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "case_study"))
import case_utils


class CaseStudyConfigTests(unittest.TestCase):
    def test_root_is_independent_of_working_directory(self):
        config = case_utils.load_config()
        self.assertEqual(config["project_root"], case_utils.PROJECT_ROOT)
        with tempfile.TemporaryDirectory() as directory:
            original = Path.cwd()
            try:
                os.chdir(directory)
                self.assertEqual(case_utils.load_config()["project_root"], config["project_root"])
            finally:
                os.chdir(original)

    def test_new_experiment_paths(self):
        config = case_utils.load_config()
        self.assertEqual(config["vina_csv"], "vina_openbabel_test99_seed3407.csv")
        self.assertEqual(config["vina_work_dir"], "work_vina_openbabel_test99_seed3407")
        self.assertEqual(config["generation_seed"], 3407)
        self.assertEqual(config["local_docking_seed"], 3407)
        self.assertTrue(case_utils.project_path(config, "metadata_csv").is_file())

    @unittest.skipIf(os.name == "nt", "WSL path translation test")
    def test_windows_drive_paths_under_linux(self):
        for value in ("E:/RigiDiff/file.csv", r"E:\RigiDiff\file.csv"):
            self.assertEqual(
                case_utils.resolve_project_path(value, Path("/tmp/project")),
                Path("/mnt/e/RigiDiff/file.csv"),
            )

    def test_missing_csv_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {"project_root": Path(directory), "vina_csv": "new_run.csv"}
            with self.assertRaisesRegex(FileNotFoundError, "New-experiment Vina CSV"):
                case_utils.load_vina(config)

    def test_named_executable_uses_path(self):
        with patch.object(case_utils.shutil, "which", return_value=sys.executable):
            self.assertEqual(case_utils.executable_path({"vina_executable": "vina"}, "vina_executable"), Path(sys.executable))

    def test_missing_executable_is_not_joined_silently(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {"project_root": Path(directory), "vina_executable": "missing-vina-test"}
            with patch.object(case_utils.shutil, "which", return_value=None):
                with self.assertRaisesRegex(FileNotFoundError, "Activate the environment"):
                    case_utils.executable_path(config, "vina_executable")

    def test_prepared_receptor_uses_configured_experiment(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {"project_root": Path(directory), "vina_work_dir": "new_work"}
            expected = Path(directory) / "new_work/prepared_receptors/target_48_2OQV_prepared.pdb"
            expected.parent.mkdir(parents=True)
            expected.touch()
            self.assertEqual(
                case_utils.prepared_protein_path(config, "E:/old/work/prepared_receptors/target_48_2OQV_prepared.pdb"),
                expected,
            )
            expected.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "configured docking run"):
                case_utils.prepared_protein_path(config, str(expected))


if __name__ == "__main__":
    unittest.main()
