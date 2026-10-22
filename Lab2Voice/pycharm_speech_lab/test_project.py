"""Testy niewymagające mikrofonu ani prawdziwych nagrań mowy."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import soundfile as sf

from dataset_report import analyze_dataset
from speech_commands import (
    DURATION,
    N_SAMPLES,
    SR,
    WORDS,
    add_noise,
    audio_to_features,
    model_parameter_counts,
    predict,
    split_by_speaker,
    time_shift,
    train,
)
from visualize_audio import create_visualizations


def synthetic_signal(frequency: float, speaker_offset: float = 0.0) -> np.ndarray:
    time_axis = np.arange(N_SAMPLES, dtype=np.float32) / SR
    envelope = np.sin(np.pi * np.arange(N_SAMPLES, dtype=np.float32) / N_SAMPLES) ** 2
    signal = 0.6 * np.sin(2 * np.pi * (frequency + speaker_offset) * time_axis)
    return (signal * envelope).astype(np.float32)


def create_synthetic_dataset(root: Path) -> None:
    for speaker_number, speaker in enumerate(("osoba_a", "osoba_b", "osoba_c")):
        for label, word in enumerate(WORDS):
            destination = root / speaker / word
            destination.mkdir(parents=True, exist_ok=True)
            signal = synthetic_signal(220 + 80 * label, 3 * speaker_number)
            sf.write(destination / "000.wav", signal, SR)


class ProjectTests(unittest.TestCase):
    def test_features_and_augmentations_keep_expected_shapes(self) -> None:
        signal = synthetic_signal(440)
        self.assertEqual(audio_to_features(signal, 13).shape[0], 13)
        self.assertEqual(audio_to_features(signal, 40).shape[0], 40)
        noisy = add_noise(signal, rng=np.random.default_rng(1))
        shifted = time_shift(signal, rng=np.random.default_rng(2))
        self.assertEqual(noisy.shape, (N_SAMPLES,))
        self.assertEqual(shifted.shape, (N_SAMPLES,))
        self.assertTrue(np.all(np.abs(noisy) <= 1.0))

    def test_speaker_splits_are_disjoint(self) -> None:
        labels = np.tile(np.arange(len(WORDS)), 3)
        groups = np.repeat(np.array(["a", "b", "c"]), len(WORDS))
        train_idx, val_idx, test_idx = split_by_speaker(labels, groups)
        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        test_groups = set(groups[test_idx])
        self.assertTrue(train_groups.isdisjoint(val_groups))
        self.assertTrue(train_groups.isdisjoint(test_groups))
        self.assertTrue(val_groups.isdisjoint(test_groups))

    def test_extended_model_has_more_parameters(self) -> None:
        counts = model_parameter_counts()
        self.assertGreater(counts["extended"], counts["base"])

    def test_full_pipeline_on_synthetic_non_speech_audio(self) -> None:
        with TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            data = temporary_path / "data"
            create_synthetic_dataset(data)
            report = analyze_dataset(data)
            self.assertEqual(report["file_count"], 3 * len(WORDS))
            self.assertFalse(report["problems"])

            plots = create_visualizations(
                [data / "osoba_a" / "start" / "000.wav"],
                temporary_path / "plots",
            )
            self.assertGreaterEqual(len(plots), 2)
            self.assertTrue(all(path.is_file() for path in plots))

            model_path = temporary_path / "speech_model.pt"
            summary = train(
                data,
                model_path,
                epochs=1,
                batch_size=8,
                n_mfcc=13,
                augment=True,
                results_root=temporary_path / "results",
            )
            self.assertTrue(model_path.is_file())
            self.assertEqual(
                summary["train_examples_after_augmentation"],
                3 * summary["train_examples_before_augmentation"],
            )
            self.assertTrue((Path(summary["run_dir"]) / "summary.json").is_file())
            prediction = predict(data / "osoba_a" / "start" / "000.wav", model_path)
            self.assertIn(prediction["predicted"], WORDS)
            self.assertAlmostEqual(sum(prediction["probabilities"].values()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
