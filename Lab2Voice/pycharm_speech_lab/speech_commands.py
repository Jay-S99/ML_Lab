"""Laboratorium: klasyfikacja pięciu komend głosowych (nie pełna transkrypcja)."""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime
import json
from pathlib import Path
import random
import time

import librosa
import numpy as np
import soundfile as sf


WORDS = ["start", "stop", "lewo", "prawo", "skok"]
SR = 16_000
DURATION = 1.5
N_SAMPLES = int(SR * DURATION)
MODEL_VARIANTS = ("base", "extended")
MFCC_OPTIONS = (13, 20, 40)


def load_audio(path: Path | str) -> np.ndarray:
    """Wczytaj mono 16 kHz, dopasuj długość i znormalizuj amplitudę."""
    audio, _ = librosa.load(path, sr=SR, mono=True)
    audio = librosa.util.fix_length(audio, size=N_SAMPLES).astype(np.float32)
    if not np.all(np.isfinite(audio)):
        raise ValueError(f"Nagranie zawiera wartości NaN lub inf: {path}")
    peak = float(np.max(np.abs(audio)))
    if peak > 1e-6:
        audio = audio / peak
    return audio.astype(np.float32)


def audio_to_features(audio: np.ndarray, n_mfcc: int = 20) -> np.ndarray:
    """Zamień sygnał o stałej długości na macierz współczynników MFCC."""
    if n_mfcc < 1:
        raise ValueError("Liczba MFCC musi być dodatnia.")
    audio = librosa.util.fix_length(np.asarray(audio, dtype=np.float32), size=N_SAMPLES)
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SR,
        n_mfcc=n_mfcc,
        n_fft=400,
        hop_length=160,
    )
    return mfcc.astype(np.float32)


def features(path: Path | str, n_mfcc: int = 20) -> np.ndarray:
    """Wczytaj WAV i wyznacz jego MFCC."""
    return audio_to_features(load_audio(path), n_mfcc=n_mfcc)


def add_noise(
    audio: np.ndarray,
    level: float = 0.01,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Dodaj szum Gaussa bez zmiany długości sygnału."""
    if level < 0:
        raise ValueError("Poziom szumu nie może być ujemny.")
    rng = rng or np.random.default_rng(42)
    noisy = np.asarray(audio, dtype=np.float32) + rng.normal(0, level, size=audio.shape)
    return np.clip(noisy, -1.0, 1.0).astype(np.float32)


def time_shift(
    audio: np.ndarray,
    max_fraction: float = 0.1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Przesuń sygnał w czasie i wyzeruj próbki zawinięte przez ``np.roll``."""
    if not 0 <= max_fraction <= 1:
        raise ValueError("max_fraction musi należeć do przedziału 0..1.")
    rng = rng or np.random.default_rng(42)
    audio = np.asarray(audio, dtype=np.float32)
    max_shift = int(len(audio) * max_fraction)
    if max_shift == 0:
        return audio.copy()
    shift = int(rng.integers(-max_shift, max_shift + 1))
    shifted = np.roll(audio, shift)
    if shift > 0:
        shifted[:shift] = 0
    elif shift < 0:
        shifted[shift:] = 0
    return shifted.astype(np.float32)


def augmented_audio(audio: np.ndarray, rng: np.random.Generator) -> list[tuple[str, np.ndarray]]:
    """Zwróć dwie augmentacje używane wyłącznie dla zbioru treningowego."""
    return [
        ("noise", add_noise(audio, level=0.01, rng=rng)),
        ("time_shift", time_shift(audio, max_fraction=0.1, rng=rng)),
    ]


def record(word: str, speaker: str, count: int, root: Path) -> None:
    """Nagraj serię przykładów jednej komendy."""
    import sounddevice as sd

    if word not in WORDS:
        raise ValueError(f"Wybierz: {WORDS}")
    dest = root / speaker / word
    dest.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        input(f'[{speaker}] Powiedz "{word}" po Enter (nagranie {i + 1}/{count})...')
        audio = sd.rec(N_SAMPLES, samplerate=SR, channels=1, dtype="float32")
        sd.wait()
        out = dest / f'{len(list(dest.glob("*.wav"))):03d}.wav'
        sf.write(out, audio[:, 0], SR)
        print("Zapisano:", out)


def scan_dataset(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zbierz ścieżki, etykiety i mówców bez wczytywania audio."""
    paths: list[Path] = []
    labels: list[int] = []
    groups: list[str] = []
    for label, word in enumerate(WORDS):
        for path in sorted(root.glob(f"*/{word}/*.wav")):
            paths.append(path)
            labels.append(label)
            groups.append(path.parent.parent.name)
    if not paths:
        raise ValueError("Brak nagrań: data/IMIE/KOMENDA/*.wav")
    return np.asarray(paths, dtype=object), np.asarray(labels), np.asarray(groups)


def split_by_speaker(
    labels: np.ndarray,
    groups: np.ndarray,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Podziel indeksy na train/validation/test bez wspólnych mówców."""
    from sklearn.model_selection import GroupShuffleSplit

    group_count = len(np.unique(groups))
    if group_count < 3:
        raise ValueError(
            "Potrzeba co najmniej 3 mówców do rozłącznego podziału "
            "train/validation/test."
        )
    test_group_count = min(max(1, round(group_count * 0.2)), group_count - 2)
    validation_group_count = min(
        max(1, round(group_count * 0.2)),
        group_count - test_group_count - 1,
    )
    dummy = np.zeros(len(labels))
    first = GroupShuffleSplit(
        n_splits=1,
        test_size=test_group_count,
        random_state=seed,
    )
    train_val_idx, test_idx = next(first.split(dummy, labels, groups))
    second = GroupShuffleSplit(
        n_splits=1,
        test_size=validation_group_count,
        random_state=seed + 1,
    )
    train_local, val_local = next(
        second.split(
            np.zeros(len(train_val_idx)),
            labels[train_val_idx],
            groups[train_val_idx],
        )
    )
    return train_val_idx[train_local], train_val_idx[val_local], test_idx


def _feature_dataset(
    paths: np.ndarray,
    labels: np.ndarray,
    n_mfcc: int,
    use_augmentation: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    feature_rows: list[np.ndarray] = []
    output_labels: list[int] = []
    sources: list[str] = []
    for path, label in zip(paths, labels):
        audio = load_audio(path)
        versions = [("original", audio)]
        if use_augmentation:
            versions.extend(augmented_audio(audio, rng))
        for version, version_audio in versions:
            feature_rows.append(audio_to_features(version_audio, n_mfcc=n_mfcc))
            output_labels.append(int(label))
            sources.append(str(path) if version == "original" else f"{path}#{version}")
    return np.stack(feature_rows), np.asarray(output_labels), sources


def load_dataset(root: Path, n_mfcc: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wczytaj cały zbiór bez augmentacji (prosty interfejs zgodności)."""
    paths, labels, groups = scan_dataset(root)
    X, y, _ = _feature_dataset(
        paths,
        labels,
        n_mfcc=n_mfcc,
        use_augmentation=False,
        rng=np.random.default_rng(42),
    )
    return X, y, groups


def build_model(variant: str = "base", n_classes: int = 5):
    """Zbuduj bazowy lub rozszerzony klasyfikator CNN."""
    from torch import nn

    if variant == "extended":
        return nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(64, n_classes),
        )
    if variant != "base":
        raise ValueError(f"Nieznany wariant {variant!r}. Wybierz: {MODEL_VARIANTS}")
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
        nn.Linear(32, n_classes),
    )


def count_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def model_parameter_counts() -> dict[str, int]:
    """Porównaj wielkość architektur bez potrzeby posiadania danych."""
    return {variant: count_parameters(build_model(variant, len(WORDS))) for variant in MODEL_VARIANTS}


def _evaluate(model, inputs, labels):
    import torch

    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(inputs), dim=1)
        predictions = probabilities.argmax(dim=1)
        accuracy = (predictions == labels).float().mean().item()
    return predictions, probabilities, accuracy


def _make_run_directory(results_root: Path, variant: str, n_mfcc: int) -> Path:
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = results_root / f"{run_id}_{variant}_mfcc{n_mfcc}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _save_training_artifacts(
    run_dir: Path,
    history: list[dict[str, float | int]],
    confusion: np.ndarray,
    errors: list[dict[str, object]],
    summary: dict[str, object],
) -> None:
    import matplotlib.pyplot as plt

    payload = dict(summary)
    payload["history"] = history
    payload["confusion_matrix"] = confusion.tolist()
    payload["misclassified"] = errors
    (run_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (run_dir / "misclassified.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["path", "true_label", "predicted_label", "confidence"],
        )
        writer.writeheader()
        writer.writerows(errors)

    epochs = [row["epoch"] for row in history]
    losses = [row["train_loss"] for row in history]
    accuracies = [row["val_accuracy"] for row in history]
    fig, loss_axis = plt.subplots(figsize=(8, 4.5))
    accuracy_axis = loss_axis.twinx()
    loss_axis.plot(epochs, losses, color="tab:blue", marker="o", label="train loss")
    accuracy_axis.plot(
        epochs, accuracies, color="tab:orange", marker="o", label="validation accuracy"
    )
    loss_axis.set(xlabel="Epoka", ylabel="Średnia strata treningowa")
    accuracy_axis.set_ylabel("Validation accuracy")
    accuracy_axis.set_ylim(0, 1)
    loss_axis.grid(alpha=0.25)
    lines = loss_axis.lines + accuracy_axis.lines
    loss_axis.legend(lines, [line.get_label() for line in lines], loc="best")
    fig.tight_layout()
    fig.savefig(run_dir / "training_history.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(confusion, cmap="Blues")
    axis.set(
        xticks=range(len(WORDS)), yticks=range(len(WORDS)),
        xticklabels=WORDS, yticklabels=WORDS,
        xlabel="Predykcja", ylabel="Klasa prawdziwa",
        title="Macierz pomyłek - zbiór testowy",
    )
    for row in range(confusion.shape[0]):
        for column in range(confusion.shape[1]):
            axis.text(column, row, int(confusion[row, column]), ha="center", va="center")
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(run_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def _append_experiment(results_root: Path, summary: dict[str, object]) -> None:
    csv_path = results_root / "experiments.csv"
    fields = [
        "run_id", "variant", "n_mfcc", "learning_rate", "batch_size",
        "augmentation", "epochs", "best_epoch", "parameter_count",
        "training_seconds", "test_accuracy", "train_speakers",
        "validation_speakers", "test_speakers",
    ]
    row = {field: summary[field] for field in fields}
    for key in ("train_speakers", "validation_speakers", "test_speakers"):
        row[key] = ";".join(row[key])
    with csv_path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerow(row)


def train(
    root: Path,
    model_path: Path,
    epochs: int,
    learning_rate: float = 0.001,
    batch_size: int = 16,
    variant: str = "base",
    n_mfcc: int = 20,
    augment: bool = False,
    seed: int = 42,
    results_root: Path | None = None,
) -> dict[str, object]:
    """Trenuj model, wybierz epokę na walidacji i raz oceń ją na teście."""
    import torch
    from torch import nn
    from sklearn.metrics import classification_report, confusion_matrix

    if epochs < 1:
        raise ValueError("Liczba epok musi być dodatnia.")
    if learning_rate <= 0:
        raise ValueError("Learning rate musi być dodatni.")
    if batch_size < 1:
        raise ValueError("Batch size musi być dodatni.")
    if variant not in MODEL_VARIANTS:
        raise ValueError(f"Wariant musi należeć do: {MODEL_VARIANTS}")
    if n_mfcc not in MFCC_OPTIONS:
        raise ValueError(f"Liczba MFCC musi należeć do: {MFCC_OPTIONS}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    paths, labels, groups = scan_dataset(root)
    train_idx, val_idx, test_idx = split_by_speaker(labels, groups, seed=seed)
    split_speakers = {
        "train_speakers": sorted(np.unique(groups[train_idx]).tolist()),
        "validation_speakers": sorted(np.unique(groups[val_idx]).tolist()),
        "test_speakers": sorted(np.unique(groups[test_idx]).tolist()),
    }
    print("Mówcy train:", ", ".join(split_speakers["train_speakers"]))
    print("Mówcy validation:", ", ".join(split_speakers["validation_speakers"]))
    print("Mówcy test:", ", ".join(split_speakers["test_speakers"]))

    X_train, y_train, _ = _feature_dataset(
        paths[train_idx], labels[train_idx], n_mfcc, augment, rng
    )
    X_val, y_val, _ = _feature_dataset(paths[val_idx], labels[val_idx], n_mfcc, False, rng)
    X_test, y_test, test_sources = _feature_dataset(
        paths[test_idx], labels[test_idx], n_mfcc, False, rng
    )
    mean = X_train.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    std = (X_train.std(axis=(0, 2), keepdims=True) + 1e-6).astype(np.float32)
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    x_train = torch.tensor(X_train[:, None], dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    x_val = torch.tensor(X_val[:, None], dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    x_test = torch.tensor(X_test[:, None], dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    model = build_model(variant, len(WORDS))
    parameter_count = count_parameters(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    history: list[dict[str, float | int]] = []
    best_val_accuracy = -1.0
    best_epoch = 0
    best_weights = copy.deepcopy(model.state_dict())
    training_started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(x_train))
        weighted_loss = 0.0
        for ids in order.split(batch_size):
            optimizer.zero_grad()
            loss = loss_fn(model(x_train[ids]), y_train_tensor[ids])
            loss.backward()
            optimizer.step()
            weighted_loss += loss.item() * len(ids)
        train_loss = weighted_loss / len(x_train)
        _, _, val_accuracy = _evaluate(model, x_val, y_val_tensor)
        history.append({
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "val_accuracy": float(val_accuracy),
        })
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1
            best_weights = copy.deepcopy(model.state_dict())
        print(
            f"Epoka {epoch + 1:02d}/{epochs} | mean loss={train_loss:.3f} | "
            f"val accuracy={val_accuracy:.1%}"
        )

    training_seconds = time.perf_counter() - training_started
    model.load_state_dict(best_weights)
    predictions, probabilities, test_accuracy = _evaluate(model, x_test, y_test_tensor)
    y_true = y_test_tensor.numpy()
    y_pred = predictions.numpy()
    print(f"Najlepsza epoka: {best_epoch} (val accuracy={best_val_accuracy:.1%})")
    print(f"Jednorazowa ocena testowa: accuracy={test_accuracy:.1%}")
    report_text = classification_report(
        y_true, y_pred, labels=list(range(len(WORDS))), target_names=WORDS, zero_division=0
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(WORDS))),
        target_names=WORDS,
        zero_division=0,
        output_dict=True,
    )
    print(report_text)
    confusion = confusion_matrix(y_true, y_pred, labels=list(range(len(WORDS))))
    print(confusion)

    errors: list[dict[str, object]] = []
    probabilities_np = probabilities.numpy()
    for index, (true_label, predicted_label) in enumerate(zip(y_true, y_pred)):
        if true_label != predicted_label:
            errors.append({
                "path": test_sources[index],
                "true_label": WORDS[int(true_label)],
                "predicted_label": WORDS[int(predicted_label)],
                "confidence": round(float(probabilities_np[index].max()), 6),
            })

    results_root = Path(results_root) if results_root else model_path.parent / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    run_dir = _make_run_directory(results_root, variant, n_mfcc)
    summary: dict[str, object] = {
        "run_id": run_dir.name,
        "variant": variant,
        "n_mfcc": n_mfcc,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "augmentation": augment,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_validation_accuracy": float(best_val_accuracy),
        "parameter_count": parameter_count,
        "training_seconds": round(training_seconds, 6),
        "test_accuracy": float(test_accuracy),
        "dataset_examples": len(paths),
        "train_examples_before_augmentation": len(train_idx),
        "train_examples_after_augmentation": len(X_train),
        "validation_examples": len(val_idx),
        "test_examples": len(test_idx),
        "classification_report": report_dict,
        **split_speakers,
    }
    checkpoint = {
        "weights": model.state_dict(),
        "mean": torch.tensor(mean),
        "std": torch.tensor(std),
        "variant": variant,
        "words": WORDS,
        "n_mfcc": n_mfcc,
        "sample_rate": SR,
        "duration": DURATION,
        "seed": seed,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, model_path)
    torch.save(checkpoint, run_dir / "model.pt")
    _save_training_artifacts(run_dir, history, confusion, errors, summary)
    _append_experiment(results_root, summary)
    print("Model zapisany:", model_path)
    print("Wyniki eksperymentu:", run_dir)
    return {**summary, "run_dir": str(run_dir), "errors": errors}


def predict(path: Path, model_path: Path) -> dict[str, object]:
    """Odtwórz konfigurację z checkpointu i sklasyfikuj plik WAV."""
    import torch

    state = torch.load(model_path, map_location="cpu", weights_only=True)
    words = list(state.get("words", WORDS))
    variant = str(state.get("variant", "base"))
    n_mfcc = int(state.get("n_mfcc", 20))
    model = build_model(variant, len(words))
    model.load_state_dict(state["weights"])
    model.eval()
    mean = state["mean"].cpu().numpy() if hasattr(state["mean"], "cpu") else state["mean"]
    std = state["std"].cpu().numpy() if hasattr(state["std"], "cpu") else state["std"]
    feat = (features(path, n_mfcc=n_mfcc) - mean[0]) / std[0]
    with torch.no_grad():
        probabilities = torch.softmax(
            model(torch.tensor(feat[None, None], dtype=torch.float32)), dim=1
        )[0].numpy()
    ranking = sorted(zip(words, probabilities), key=lambda item: -item[1])
    for word, probability in ranking:
        print(f"{word:>6}: {probability:.1%}")
    print("Uwaga: pewność softmax nie jest gwarancją poprawności.")
    return {
        "predicted": ranking[0][0],
        "confidence": float(ranking[0][1]),
        "probabilities": {word: float(probability) for word, probability in ranking},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    record_parser = sub.add_parser("record")
    record_parser.add_argument("--word", choices=WORDS, required=True)
    record_parser.add_argument("--speaker", required=True)
    record_parser.add_argument("--count", type=int, default=10)
    record_parser.add_argument("--data", type=Path, default=Path("data"))

    train_parser = sub.add_parser("train")
    train_parser.add_argument("--epochs", type=int, default=25)
    train_parser.add_argument("--learning-rate", type=float, default=0.001)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--variant", choices=MODEL_VARIANTS, default="base")
    train_parser.add_argument("--n-mfcc", choices=MFCC_OPTIONS, type=int, default=20)
    train_parser.add_argument("--augment", action="store_true")
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--data", type=Path, default=Path("data"))
    train_parser.add_argument("--model", type=Path, default=Path("speech_model.pt"))

    predict_parser = sub.add_parser("predict")
    predict_parser.add_argument("wav", type=Path)
    predict_parser.add_argument("--model", type=Path, default=Path("speech_model.pt"))
    sub.add_parser("model-info")
    args = parser.parse_args()
    if args.action == "record":
        record(args.word, args.speaker, args.count, args.data)
    elif args.action == "train":
        train(
            args.data, args.model, args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            variant=args.variant,
            n_mfcc=args.n_mfcc,
            augment=args.augment,
            seed=args.seed,
        )
    elif args.action == "predict":
        predict(args.wav, args.model)
    else:
        for name, count in model_parameter_counts().items():
            print(f"{name:>8}: {count:,} parametrów")


if __name__ == "__main__":
    main()
