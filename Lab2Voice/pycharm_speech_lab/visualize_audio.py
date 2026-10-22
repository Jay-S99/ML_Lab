"""Zapisuj wykresy amplitudy i MFCC dla istniejących plików WAV."""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from speech_commands import MFCC_OPTIONS, SR, audio_to_features, load_audio


def safe_stem(path: Path) -> str:
    parts = path.with_suffix("").parts[-3:]
    return "_".join(part.replace(" ", "_") for part in parts)


def plot_signal_and_mfcc(path: Path, output_dir: Path, n_mfcc: int = 20) -> Path:
    audio = load_audio(path)
    mfcc = audio_to_features(audio, n_mfcc=n_mfcc)
    time_axis = np.arange(len(audio)) / SR
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    axes[0].plot(time_axis, audio, linewidth=0.8)
    axes[0].set(title=f"Sygnał audio: {path.name}", xlabel="Czas [s]", ylabel="Amplituda")
    image = librosa.display.specshow(
        mfcc, x_axis="time", sr=SR, hop_length=160, ax=axes[1], cmap="magma"
    )
    axes[1].set(title=f"Współczynniki MFCC (n={n_mfcc})")
    fig.colorbar(image, ax=axes[1], format="%+2.0f dB")
    fig.tight_layout()
    output = output_dir / f"{safe_stem(path)}_signal_mfcc{n_mfcc}.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_mfcc_variants(path: Path, output_dir: Path) -> Path:
    audio = load_audio(path)
    fig, axes = plt.subplots(len(MFCC_OPTIONS), 1, figsize=(11, 9), sharex=True)
    for axis, n_mfcc in zip(axes, MFCC_OPTIONS):
        mfcc = audio_to_features(audio, n_mfcc=n_mfcc)
        image = librosa.display.specshow(
            mfcc, x_axis="time", sr=SR, hop_length=160, ax=axis, cmap="magma"
        )
        axis.set(title=f"MFCC: {n_mfcc} współczynników", ylabel="Numer MFCC")
        fig.colorbar(image, ax=axis, format="%+2.0f dB")
    fig.suptitle(f"Wpływ liczby MFCC: {path.name}")
    fig.tight_layout()
    output = output_dir / f"{safe_stem(path)}_mfcc_13_20_40.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_signal_comparison(paths: list[Path], output_dir: Path) -> Path:
    fig, axes = plt.subplots(len(paths), 1, figsize=(11, 2.6 * len(paths)), sharex=True)
    if len(paths) == 1:
        axes = [axes]
    for axis, path in zip(axes, paths):
        audio = load_audio(path)
        axis.plot(np.arange(len(audio)) / SR, audio, linewidth=0.8)
        axis.set(title=str(path), ylabel="Amplituda")
    axes[-1].set_xlabel("Czas [s]")
    fig.suptitle("Porównanie nagrań")
    fig.tight_layout()
    output = output_dir / "signals_comparison.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def create_visualizations(paths: list[Path], output_dir: Path) -> list[Path]:
    if not paths:
        raise ValueError("Podaj co najmniej jeden plik WAV.")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Nie znaleziono: {missing[0]}")
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [plot_signal_and_mfcc(path, output_dir) for path in paths]
    outputs.append(plot_mfcc_variants(paths[0], output_dir))
    if len(paths) >= 2:
        outputs.append(plot_signal_comparison(paths, output_dir))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", type=Path, nargs="+", help="Jeden lub więcej plików WAV")
    parser.add_argument("--output-dir", type=Path, default=Path("plots"))
    args = parser.parse_args()
    for output in create_visualizations(args.wav, args.output_dir):
        print("Zapisano:", output)


if __name__ == "__main__":
    main()
