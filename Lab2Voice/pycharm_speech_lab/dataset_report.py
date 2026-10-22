"""Raport jakości i równowagi zbioru nagrań WAV."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean, median

import numpy as np
import soundfile as sf

from speech_commands import SR, WORDS


def analyze_dataset(root: Path) -> dict[str, object]:
    """Sprawdź liczebność, parametry techniczne i typowe uszkodzenia nagrań."""
    files = sorted(root.glob("*/*/*.wav"))
    counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    class_counts: Counter[str] = Counter()
    durations: list[float] = []
    problems: list[dict[str, object]] = []

    for path in files:
        speaker = path.parent.parent.name
        word = path.parent.name
        counts[speaker][word] += 1
        class_counts[word] += 1
        try:
            info = sf.info(path)
            audio, _ = sf.read(path, dtype="float32", always_2d=False)
        except (RuntimeError, OSError, ValueError) as exc:
            problems.append({"path": str(path), "problem": "uszkodzony", "details": str(exc)})
            continue

        durations.append(float(info.duration))
        if info.samplerate != SR:
            problems.append({
                "path": str(path),
                "problem": "nieprawidłowa częstotliwość",
                "details": f"{info.samplerate} Hz zamiast {SR} Hz",
            })
        if info.frames == 0 or np.size(audio) == 0:
            problems.append({"path": str(path), "problem": "pusty", "details": "0 próbek"})
        elif float(np.max(np.abs(audio))) < 1e-4:
            problems.append({
                "path": str(path),
                "problem": "cisza lub niemal pusty sygnał",
                "details": f"peak={float(np.max(np.abs(audio))):.2e}",
            })
        if info.duration < 0.3:
            problems.append({
                "path": str(path),
                "problem": "bardzo krótkie nagranie",
                "details": f"{info.duration:.3f} s",
            })

    duration_stats = None
    if durations:
        duration_stats = {
            "minimum": min(durations),
            "mean": mean(durations),
            "median": median(durations),
            "maximum": max(durations),
        }
    speakers = sorted(counts)
    normalized_counts = {
        speaker: {word: counts[speaker][word] for word in WORDS} for speaker in speakers
    }
    normalized_class_counts = {word: class_counts[word] for word in WORDS}
    return {
        "root": str(root),
        "file_count": len(files),
        "speakers": speakers,
        "counts": normalized_counts,
        "class_counts": normalized_class_counts,
        "duration_seconds": duration_stats,
        "problems": problems,
    }


def print_report(root: Path) -> dict[str, object]:
    """Wyświetl czytelny raport i zwróć jego dane."""
    report = analyze_dataset(root)
    print("\n=== RAPORT ZBIORU DANYCH ===")
    print("Katalog:", root)
    print("Pliki WAV:", report["file_count"])
    if not report["file_count"]:
        print("Brak nagrań do analizy.")
        return report

    header = "mówca".ljust(18) + "".join(word.rjust(9) for word in WORDS)
    print("\nLiczba nagrań według mówcy i komendy:")
    print(header)
    for speaker, speaker_counts in report["counts"].items():
        row = speaker.ljust(18) + "".join(str(speaker_counts[word]).rjust(9) for word in WORDS)
        print(row)

    class_counts = report["class_counts"]
    print("\nŁącznie według klas:")
    print(", ".join(f"{word}: {class_counts[word]}" for word in WORDS))
    minimum = min(class_counts.values())
    maximum = max(class_counts.values())
    least = [word for word, count in class_counts.items() if count == minimum]
    most = [word for word, count in class_counts.items() if count == maximum]
    print(f"Najmniej reprezentowane ({minimum}): {', '.join(least)}")
    print(f"Najliczniej reprezentowane ({maximum}): {', '.join(most)}")
    if maximum and minimum / maximum < 0.8:
        print("Ocena równowagi: zbiór jest niezrównoważony; uzupełnij przede wszystkim klasy najmniej liczne.")
    else:
        print("Ocena równowagi: liczebności klas są zbliżone.")

    stats = report["duration_seconds"]
    if stats:
        print(
            "Czas [s]: "
            f"min={stats['minimum']:.3f}, średnia={stats['mean']:.3f}, "
            f"mediana={stats['median']:.3f}, max={stats['maximum']:.3f}"
        )
    problems = report["problems"]
    print("Problemy:", len(problems))
    for problem in problems:
        print(f"- {problem['problem']}: {problem['path']} ({problem['details']})")
    if problems:
        print("Nie usuwaj automatycznie trudnych, ale poprawnych nagrań; każdy przypadek sprawdź ręcznie.")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--json", type=Path, help="Opcjonalny plik JSON z raportem")
    args = parser.parse_args()
    report = print_report(args.data)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Zapisano raport:", args.json)


if __name__ == "__main__":
    main()
