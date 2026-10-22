"""Uruchom ten plik zielonym przyciskiem Run w PyCharm."""

from pathlib import Path

from speech_commands import (
    MFCC_OPTIONS,
    MODEL_VARIANTS,
    WORDS,
    model_parameter_counts,
    predict,
    record,
    train,
)


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODEL = ROOT / "speech_model.pt"


def ask_int(prompt: str, default: int, minimum: int, maximum: int) -> int:
    raw = input(f"{prompt} [{default}]: ").strip()
    value = default if not raw else int(raw)
    if not minimum <= value <= maximum:
        raise ValueError(f"Wartość musi należeć do przedziału {minimum}..{maximum}.")
    return value


def ask_float(prompt: str, default: float, minimum: float, maximum: float) -> float:
    raw = input(f"{prompt} [{default}]: ").strip().replace(",", ".")
    value = default if not raw else float(raw)
    if not minimum <= value <= maximum:
        raise ValueError(f"Wartość musi należeć do przedziału {minimum}..{maximum}.")
    return value


def ask_choice(prompt: str, choices: tuple, default):
    raw = input(f"{prompt} ({'/'.join(map(str, choices))}) [{default}]: ").strip().lower()
    value = default if not raw else raw
    if isinstance(default, int):
        value = int(value)
    if value not in choices:
        raise ValueError(f"Wybierz jedną z wartości: {choices}.")
    return value


def ask_yes_no(prompt: str, default: bool = False) -> bool:
    default_label = "t" if default else "n"
    raw = input(f"{prompt} (t/n) [{default_label}]: ").strip().lower()
    if not raw:
        return default
    if raw in {"t", "tak", "y", "yes"}:
        return True
    if raw in {"n", "nie", "no"}:
        return False
    raise ValueError("Wpisz t lub n.")


def show_recording_counts() -> None:
    if not DATA.exists():
        print("Brak katalogu data.")
        return
    for speaker in sorted(DATA.iterdir()):
        if speaker.is_dir():
            counts = ", ".join(
                f'{word}: {len(list((speaker / word).glob("*.wav")))}' for word in WORDS
            )
            print(speaker.name, "=>", counts)


def configure_training() -> None:
    print("Pozostaw puste pole, aby przyjąć wartość domyślną.")
    epochs = ask_int("Liczba epok", 25, 1, 500)
    learning_rate = ask_float("Learning rate", 0.001, 1e-7, 1.0)
    batch_size = ask_choice("Batch size", (8, 16, 32), 16)
    variant = ask_choice("Architektura", MODEL_VARIANTS, "base")
    n_mfcc = ask_choice("Liczba MFCC", MFCC_OPTIONS, 20)
    augment = ask_yes_no("Włączyć augmentację tylko dla train", False)
    seed = ask_int("Ziarno losowości", 42, 0, 2_147_483_647)
    train(
        DATA,
        MODEL,
        epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        variant=variant,
        n_mfcc=n_mfcc,
        augment=augment,
        seed=seed,
    )


def main() -> None:
    print("=== LABORATORIUM: ROZPOZNAWANIE KOMEND GŁOSOWYCH ===")
    print("Słowa:", ", ".join(WORDS))
    while True:
        print("\n1 - Nagraj przykłady")
        print("2 - Trenuj sieć neuronową")
        print("3 - Rozpoznaj nagranie WAV")
        print("4 - Sprawdź liczbę nagrań")
        print("5 - Raport jakości zbioru")
        print("6 - Porównaj rozmiary architektur CNN")
        print("0 - Zakończ")
        choice = input("Wybór: ").strip()
        try:
            if choice == "1":
                speaker = input("Identyfikator osoby (np. anna): ").strip()
                if not speaker or not speaker.replace("_", "").isalnum():
                    print("Użyj prostego identyfikatora bez spacji i ukośników.")
                    continue
                word = input("Komenda (start/stop/lewo/prawo/skok): ").strip().lower()
                if word not in WORDS:
                    print("Nieznana komenda.")
                    continue
                count = ask_int("Liczba nagrań", 15, 1, 200)
                record(word, speaker, count, DATA)
            elif choice == "2":
                configure_training()
            elif choice == "3":
                if not MODEL.exists():
                    print("Najpierw wytrenuj model (opcja 2).")
                    continue
                wav = Path(input("Ścieżka do pliku WAV: ").strip().strip('"'))
                if not wav.is_absolute():
                    wav = ROOT / wav
                if not wav.is_file():
                    print("Nie znaleziono pliku:", wav)
                    continue
                predict(wav, MODEL)
            elif choice == "4":
                show_recording_counts()
            elif choice == "5":
                from dataset_report import print_report

                print_report(DATA)
            elif choice == "6":
                for name, count in model_parameter_counts().items():
                    print(f"{name:>8}: {count:,} parametrów")
            elif choice == "0":
                break
            else:
                print("Nieznana opcja.")
        except (ValueError, OSError, RuntimeError) as exc:
            print("Błąd:", exc)


if __name__ == "__main__":
    main()
