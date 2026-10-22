# Rozpoznawanie komend głosowych — projekt do PyCharm

Projekt klasyfikuje pięć słów: `start`, `stop`, `lewo`, `prawo`, `skok`. Zawiera kompletną implementację zadań, które można przygotować bez nagrywania głosu. Rzeczywiste wyniki jakości wymagają później dodania nagrań.

## Szybki start (Windows)

1. Otwórz w PyCharm folder `pycharm_speech_lab`.
2. Ustaw interpreter projektu. Możesz wykorzystać istniejące środowisko `Lab2Voice/.venv` albo utworzyć Python 3.11/3.12 i wykonać `python -m pip install -r requirements.txt`.
3. Uruchom `main.py` albo użyj poleceń opisanych niżej.

Test niewymagający mikrofonu ani prawdziwej mowy:

```powershell
python -m unittest -v test_project.py
```

Test tworzy tymczasowy zbiór sztucznych tonów, sprawdza MFCC, augmentacje, podział mówców, wizualizacje, raport danych, trening, zapis modelu i predykcję. Sztuczne tony nie są wynikami eksperymentu rozpoznawania mowy.

## Zaimplementowane rozszerzenia

- `visualize_audio.py`: przebieg amplitudy, mapa MFCC, porównanie 13/20/40 MFCC oraz kilku plików.
- `dataset_report.py`: liczebności, czasy trwania, częstotliwość próbkowania, cisza, krótkie i uszkodzone pliki.
- `speech_commands.py`: augmentacja szumem i przesunięciem w czasie wykonywana wyłącznie po podziale i tylko dla train.
- Architektury CNN `base` i `extended`, pomiar liczby parametrów i czasu treningu.
- Rozłączne zbiory mówców train/validation/test, historia epok i wybór najlepszej epoki na walidacji.
- Jednorazowa ocena testowa: accuracy, precision, recall, F1, macierz pomyłek oraz lista błędnych predykcji.
- Konfiguracja learning rate, batch size, liczby MFCC, architektury, augmentacji i ziarna losowości.

Odpowiedzi opisowe znajdują się w `ODPOWIEDZI_TEORETYCZNE.md`.

## Dane

Docelowy układ:

```text
data/
  anna/
    start/000.wav
    stop/000.wav
  bartek/
    start/000.wav
```

Nagrania są przetwarzane jako mono, 16 kHz i 1,5 sekundy. Potrzeba co najmniej trzech mówców, ponieważ każda osoba trafia tylko do jednego ze zbiorów: train, validation albo test. Przy trzech osobach każdy zbiór zawiera głos jednej osoby; większa liczba uczestników daje bardziej miarodajne wyniki.

## Polecenia

Porównanie architektur bez danych:

```powershell
python speech_commands.py model-info
```

Raport danych, również przed rozpoczęciem nagrywania:

```powershell
python dataset_report.py --data data --json results/dataset_report.json
```

Wizualizacje jednego albo dwóch istniejących WAV-ów:

```powershell
python visualize_audio.py data/anna/start/000.wav --output-dir plots
python visualize_audio.py data/anna/start/000.wav data/bartek/start/000.wav --output-dir plots
```

Trening bazowy:

```powershell
python speech_commands.py train --epochs 25 --learning-rate 0.001 --batch-size 16 --variant base --n-mfcc 20
```

Trening rozszerzony z augmentacją:

```powershell
python speech_commands.py train --epochs 25 --learning-rate 0.001 --batch-size 16 --variant extended --n-mfcc 40 --augment
```

Predykcja wykorzystuje architekturę i liczbę MFCC zapisane w modelu:

```powershell
python speech_commands.py predict data/anna/start/000.wav
```

Nagrywanie pozostaje dostępne, gdy warunki na to pozwolą:

```powershell
python speech_commands.py record --speaker anna --word start --count 15
```

## Wyniki treningu

Najlepszy model jest zapisywany jako `speech_model.pt`. Każde uruchomienie tworzy osobny katalog `results/DATA_WARIANT_MFCC/`, zawierający:

- `model.pt` — checkpoint z wagami i konfiguracją;
- `summary.json` — konfigurację, historię i metryki;
- `training_history.png` — stratę treningową i validation accuracy;
- `confusion_matrix.png` — macierz pomyłek na teście;
- `misclassified.csv` — pliki błędnie sklasyfikowane wraz z wynikiem softmax.

Plik `results/experiments.csv` gromadzi wspólną tabelę konfiguracji, czasów i wyników kolejnych eksperymentów.

## Zasady metodologiczne

- Statystyki normalizacji są obliczane wyłącznie na train.
- Augmentacje powstają dopiero po podziale i nie dotyczą validation ani test.
- Najlepsza epoka jest wybierana na validation; test jest oceniany tylko raz po zakończeniu treningu.
- Ten sam mówca nie występuje w więcej niż jednym zbiorze.
- Softmax nie jest gwarantowanym prawdopodobieństwem poprawności.
- Nie należy usuwać trudnych, ale poprawnych nagrań tylko po to, aby podwyższyć wynik.

## Typowe problemy

- `No module named ...`: pakiety zainstalowano w innym interpreterze niż wybrany w PyCharm.
- `Brak nagrań`: sprawdź układ `data/osoba/slowo/*.wav` i katalog roboczy.
- `Potrzeba co najmniej 3 mówców`: nie kopiuj tych samych plików pod innymi nazwami; dodaj niezależne osoby.
- `load_state_dict`: model i predykcja muszą korzystać z architektury zapisanej w checkpointcie; obecny kod robi to automatycznie.
- Bardzo wysoka skuteczność: sprawdź przeciek danych i podobieństwo plików między zbiorami.
