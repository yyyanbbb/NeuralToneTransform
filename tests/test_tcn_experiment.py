from __future__ import annotations

import argparse
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf
import torch

from src.ntt.evaluation.compare_models import model_rows
from src.ntt.tcn.model import GatedTCN
from src.ntt.tcn.utils import repository_root, sanitize_paths_for_json, to_repo_relative


class TCNExperimentConfigTests(unittest.TestCase):
    def test_paths_saved_to_json_are_repository_relative(self) -> None:
        root = repository_root()
        absolute_config = root / "configs" / "tcn_gated" / "medium.json"
        absolute_output = root / "outputs" / "tcn_gated" / "medium"

        self.assertEqual(to_repo_relative(absolute_config, root), "configs/tcn_gated/medium.json")
        sanitized = sanitize_paths_for_json(
            {
                "metadata_path": str(root / "data" / "chunks" / "metadata.json"),
                "output_dir": str(absolute_output),
                "nested": {"checkpoint_path": str(absolute_output / "checkpoints" / "best.pt")},
            },
            root,
        )

        payload = json.dumps(sanitized)
        self.assertIn("configs/tcn_gated", to_repo_relative(absolute_config, root))
        self.assertEqual(sanitized["metadata_path"], "data/chunks/metadata.json")
        self.assertEqual(sanitized["output_dir"], "outputs/tcn_gated/medium")
        self.assertEqual(sanitized["nested"]["checkpoint_path"], "outputs/tcn_gated/medium/checkpoints/best.pt")
        self.assertNotIn("C:\\", payload)
        self.assertNotIn("/Users/", payload)
        self.assertNotIn("/home/", payload)

    def test_large_config_receptive_field_exceeds_a2_reference(self) -> None:
        config_path = repository_root() / "configs" / "tcn_gated" / "large.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        model = GatedTCN.from_config(config)

        self.assertEqual(config["model_name"], "GatedTCN-Large")
        self.assertGreaterEqual(model.receptive_field(), 6350)
        self.assertLess(model.count_parameters(), 1_000_000)
        self.assertEqual(tuple(model(torch.zeros(1, 1, 2048)).shape), (1, 1, 2048))


class TCNComparisonTests(unittest.TestCase):
    def test_comparison_rows_include_all_tcn_variants_with_tbd_when_missing(self) -> None:
        args = argparse.Namespace(
            target=None,
            a1_pred=None,
            a2_lite_pred=None,
            a2_full_pred=None,
            custom_tcn_pred=None,
            custom_tcn_name="GatedTCN-Medium",
            custom_tcn_checkpoint=None,
            tcn_small_pred=None,
            tcn_small_checkpoint=None,
            tcn_medium_pred=None,
            tcn_medium_checkpoint=None,
            tcn_large_pred=None,
            tcn_large_checkpoint=None,
        )

        rows = model_rows(args)
        names = [row["Model"] for row in rows]

        self.assertIn("GatedTCN-Small", names)
        self.assertIn("GatedTCN-Medium", names)
        self.assertIn("GatedTCN-Large", names)
        medium = next(row for row in rows if row["Model"] == "GatedTCN-Medium")
        self.assertEqual(medium["MSE"], "TBD")
        self.assertEqual(medium["RTF"], "TBD")


class TCNEvaluationUtilityTests(unittest.TestCase):
    def test_benchmark_writes_repository_relative_metadata(self) -> None:
        from src.ntt.tcn.benchmark import write_benchmark_json

        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "benchmark.json"
            payload = write_benchmark_json(
                out,
                {
                    "model_name": "GatedTCN-Test",
                    "checkpoint_path": repository_root() / "outputs" / "tcn_gated" / "medium" / "checkpoints" / "best.pt",
                    "input_path": repository_root() / "data" / "aligned" / "aligned_dry.wav",
                    "device": "cpu",
                },
            )

            saved = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["checkpoint_path"], "outputs/tcn_gated/medium/checkpoints/best.pt")
            self.assertEqual(saved["input_path"], "data/aligned/aligned_dry.wav")

    def test_plot_comparison_skips_missing_predictions_and_writes_analysis(self) -> None:
        from src.ntt.evaluation.plot_comparison import create_comparison_figures

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_rate = 8000
            target_path = root / "target.wav"
            pred_path = root / "pred.wav"
            samples = np.sin(np.linspace(0, 2 * np.pi, sample_rate, dtype=np.float32))
            sf.write(target_path, samples, sample_rate)
            sf.write(pred_path, samples * 0.9, sample_rate)

            result = create_comparison_figures(
                target_path=target_path,
                predictions={"TCN Medium": pred_path, "TCN Large": root / "missing.wav"},
                out_dir=root / "figures",
                report_path=root / "FIGURE_ANALYSIS.md",
                start_seconds=0.0,
                duration_seconds=0.2,
            )

            self.assertTrue((root / "figures" / "waveform_overlay.png").is_file())
            self.assertTrue((root / "figures" / "spectrogram_target.png").is_file())
            self.assertIn("TCN Large", result["skipped"])
            self.assertTrue((root / "FIGURE_ANALYSIS.md").is_file())


if __name__ == "__main__":
    unittest.main()
