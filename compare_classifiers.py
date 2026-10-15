

from __future__ import annotations

import inspect
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import all_estimators
from tabulate import tabulate

RANDOM_STATE = 42
TEST_SIZE = 0.2
TOP_N = 20
CV_SPLITS = 5

OUTPUT_DIR = Path(__file__).resolve().parent
RESULTS_CSV = OUTPUT_DIR / "classifiers_comparison.csv"
SKIPPED_LOG = OUTPUT_DIR / "skipped_classifiers.log"
RANKING_PNG = OUTPUT_DIR / "classifiers_ranking.png"
CONFUSION_PNG = OUTPUT_DIR / "best_model_confusion_matrix.png"

RESULT_COLUMNS = [
    "classifier",
    "f1_score",
    "accuracy",
    "balanced_accuracy",
    "precision_weighted",
    "recall_weighted",
    "roc_auc",
    "CV_Accuracy_Mean",
    "CV_Accuracy_Std",
    "fit_time_s",
    "predict_time_s",
]


def load_dataset(
    csv_path: str | Path | None = None,
    target_column: str | None = None,
) -> tuple[np.ndarray, np.ndarray, str, list[str]]:
    """Wczytaj cechy, etykiety i nazwy klas.

    csv_path=None wybiera sklearn.datasets.load_iris.
    Dla pliku CSV kolumną celu jest target_column, a gdy jej nie podano — ostatnia kolumna.
    Kolumny kategoryczne cech są kodowane one-hot.
    """
    if csv_path is None:
        bunch = load_iris()
        features = np.asarray(bunch.data, dtype=float)
        target = np.asarray(bunch.target)
        return features, target, "iris", [str(name) for name in bunch.target_names]

    path = Path(csv_path)
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Plik CSV jest pusty: {path}")

    target_name = target_column or str(frame.columns[-1])
    if target_name not in frame.columns:
        raise ValueError(f"Brak kolumny celu '{target_name}' w {path}")

    target_raw = frame[target_name]
    features_frame = pd.get_dummies(frame.drop(columns=[target_name]))
    features_frame = features_frame.apply(pd.to_numeric, errors="coerce")

    valid_rows = features_frame.notna().all(axis=1) & target_raw.notna()
    features_frame = features_frame.loc[valid_rows]
    target_raw = target_raw.loc[valid_rows]
    if features_frame.empty:
        raise ValueError(f"Po usunięciu braków nie zostały żadne próbki: {path}")

    if pd.api.types.is_numeric_dtype(target_raw):
        target = target_raw.to_numpy()
    else:
        target = LabelEncoder().fit_transform(target_raw.astype(str))

    features = features_frame.to_numpy(dtype=float)
    target = np.asarray(target)
    class_names = [str(label) for label in np.unique(target)]
    return features, target, path.name, class_names


def split_and_scale(
    features: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Podziel 80/20 ze stratyfikacją i zeskaluj cechy (scaler uczony tylko na train)."""
    features_train, features_test, target_train, target_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    scaler = StandardScaler()
    features_train = scaler.fit_transform(features_train)
    features_test = scaler.transform(features_test)
    return features_train, features_test, target_train, target_test


def build_classifier(estimator_cls: type):
    """Utwórz klasyfikator z domyślnymi argumentami.

    Meta-estymatory z obowiązkowymi argumentami (np. VotingClassifier) zgłaszają TypeError.
    Gdy konstruktor przyjmuje random_state, ustawiana jest stała wartość.
    """
    signature = inspect.signature(estimator_cls)
    missing = [
        name
        for name, param in signature.parameters.items()
        if param.default is inspect.Parameter.empty
        and param.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    ]
    if missing:
        joined = ", ".join(missing)
        raise TypeError(f"wymaga dodatkowych argumentów inicjalizacyjnych: {joined}")

    kwargs: dict[str, int] = {}
    random_state = signature.parameters.get("random_state")
    if random_state is not None and random_state.kind is not inspect.Parameter.VAR_KEYWORD:
        kwargs["random_state"] = RANDOM_STATE
    return estimator_cls(**kwargs)


def _multiclass_scores(model, features_test: np.ndarray) -> np.ndarray | None:
    """Macierz wyników dla każdej klasy: pełne predict_proba albo decision_function."""
    if hasattr(model, "predict_proba"):
        try:
            # Wszystkie kolumny — po jednej na klasę, bez obcinania do [:, 1].
            probabilities = np.asarray(model.predict_proba(features_test), dtype=float)
        except Exception:
            probabilities = None
        if probabilities is not None and probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            return probabilities

    if hasattr(model, "decision_function"):
        try:
            scores = np.asarray(model.decision_function(features_test), dtype=float)
        except Exception:
            return None
        if scores.ndim == 2 and scores.shape[1] >= 2:
            return scores
    return None


def safe_roc_auc(model, features_test: np.ndarray, target_test: np.ndarray) -> float:
    """Wieloklasowy ROC AUC (one-vs-rest, średnia ważona). Brak wyników daje NaN."""
    scores = _multiclass_scores(model, features_test)
    if scores is None:
        return float("nan")

    valid = np.isfinite(scores).all(axis=1)
    if not np.any(valid):
        return float("nan")
    scores = scores[valid]
    target_valid = np.asarray(target_test)[valid]

    try:
        return float(
            roc_auc_score(target_valid, scores, multi_class="ovr", average="weighted")
        )
    except Exception:
        return float("nan")


def cross_validate_accuracy(
    model,
    features: np.ndarray,
    target: np.ndarray,
    cv: StratifiedKFold,
) -> tuple[float, float]:
    """Średnia i odchylenie accuracy z przekazanych podziałów StratifiedKFold."""
    scores = cross_val_score(model, features, target, cv=cv, scoring="accuracy")
    return float(scores.mean()), float(scores.std())


def evaluate_classifier(
    model,
    features_train: np.ndarray,
    features_test: np.ndarray,
    target_train: np.ndarray,
    target_test: np.ndarray,
) -> dict[str, float]:
    """Wytrenuj model, zmierz czas fit/predict i policz metryki na zbiorze testowym."""
    start = time.perf_counter()
    model.fit(features_train, target_train)
    fit_time = time.perf_counter() - start

    start = time.perf_counter()
    predictions = model.predict(features_test)
    predict_time = time.perf_counter() - start

    return {
        "accuracy": float(accuracy_score(target_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(target_test, predictions)),
        "precision_weighted": float(
            precision_score(target_test, predictions, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(target_test, predictions, average="weighted", zero_division=0)
        ),
        "f1_score": float(f1_score(target_test, predictions, average="weighted", zero_division=0)),
        "roc_auc": safe_roc_auc(model, features_test, target_test),
        "fit_time_s": fit_time,
        "predict_time_s": predict_time,
    }


def compare_classifiers(
    features_train: np.ndarray,
    features_test: np.ndarray,
    target_train: np.ndarray,
    target_test: np.ndarray,
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Oceń każdy klasyfikator z all_estimators; błędy zapisz jako pominięcia."""
    warnings.filterwarnings("ignore")

    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    discovered = all_estimators(type_filter="classifier")
    rows: list[dict[str, float | str]] = []
    skipped: list[tuple[str, str]] = []
    total = len(discovered)

    for index, (name, estimator_cls) in enumerate(discovered, start=1):
        print(f"[{index}/{total}] {name}", flush=True)
        try:
            model = build_classifier(estimator_cls)
            try:
                cv_mean, cv_std = cross_validate_accuracy(model, features_train, target_train, cv)
            except Exception:
                cv_mean, cv_std = float("nan"), float("nan")
            metrics = evaluate_classifier(
                model,
                features_train,
                features_test,
                target_train,
                target_test,
            )
            metrics["CV_Accuracy_Mean"] = cv_mean
            metrics["CV_Accuracy_Std"] = cv_std
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}".replace("\n", " ").strip()
            if len(reason) > 240:
                reason = reason[:237] + "..."
            skipped.append((name, reason))
            print(f"  pominięto — {reason}", flush=True)
            continue

        row: dict[str, float | str] = {"classifier": name, **metrics}
        rows.append(row)
        print(
            f"  F1={metrics['f1_score']:.4f}  "
            f"accuracy={metrics['accuracy']:.4f}  "
            f"CV={metrics['CV_Accuracy_Mean']:.4f}±{metrics['CV_Accuracy_Std']:.4f}  "
            f"fit={metrics['fit_time_s']:.3f}s",
            flush=True,
        )

    if not rows:
        raise RuntimeError("Żaden klasyfikator nie zakończył oceny poprawnie.")

    results = pd.DataFrame(rows)
    results = results[RESULT_COLUMNS].sort_values(
        by=["f1_score", "accuracy"],
        ascending=False,
        ignore_index=True,
    )
    return results, skipped


def print_results(results: pd.DataFrame) -> None:
    """Wypisz tabelę wyników w konsoli."""
    display = results.copy()
    for column in RESULT_COLUMNS[1:]:
        display[column] = display[column].map(lambda value: f"{value:.4f}" if pd.notna(value) else "-")
    print()
    print(tabulate(display, headers="keys", tablefmt="github", showindex=False))
    print()


def save_outputs(results: pd.DataFrame, skipped: list[tuple[str, str]]) -> None:
    """Zapisz ranking CSV oraz log pominiętych klasyfikatorów."""
    results.to_csv(RESULTS_CSV, index=False, float_format="%.6f")
    lines = [f"{name}: {reason}" for name, reason in skipped]
    content = "\n".join(lines)
    if content:
        content += "\n"
    SKIPPED_LOG.write_text(content, encoding="utf-8")


def plot_ranking(results: pd.DataFrame, dataset_name: str) -> None:
    """Wykres TOP 20: F1-score i Accuracy oraz czas treningu."""
    top = results.head(TOP_N).copy()
    # seaborn układa pierwszą kategorię na górze osi Y.
    order = top["classifier"].tolist()

    melted = top.melt(
        id_vars="classifier",
        value_vars=["f1_score", "accuracy"],
        var_name="metric",
        value_name="score",
    )
    melted["metric"] = melted["metric"].map(
        {"f1_score": "F1-score", "accuracy": "Accuracy"}
    )

    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(1, 2, figsize=(18, 10), constrained_layout=True)

    sns.barplot(
        data=melted,
        y="classifier",
        x="score",
        hue="metric",
        order=order,
        hue_order=["F1-score", "Accuracy"],
        palette=["#2A9D8F", "#264653"],
        ax=axes[0],
    )
    axes[0].set_xlim(0, 1.22)
    axes[0].set_xlabel("Wynik na zbiorze testowym")
    axes[0].set_ylabel("")
    axes[0].set_title(f"TOP {len(top)} klasyfikatorów — F1-score i Accuracy")
    axes[0].legend(
        title="",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        frameon=False,
    )
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="%.3f", padding=3, fontsize=8)

    sns.barplot(
        data=top,
        y="classifier",
        x="fit_time_s",
        order=order,
        color="#E76F51",
        ax=axes[1],
    )
    axes[1].set_xlabel("Czas treningu (s)")
    axes[1].set_ylabel("")
    axes[1].set_title(f"Czas treningu — TOP {len(top)}")
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.3f", padding=3, fontsize=8)

    fig.suptitle(f"Porównanie klasyfikatorów scikit-learn ({dataset_name})", fontsize=14)
    fig.savefig(RANKING_PNG, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(
    model_name: str,
    features_train: np.ndarray,
    features_test: np.ndarray,
    target_train: np.ndarray,
    target_test: np.ndarray,
    class_names: list[str],
) -> None:
    """Macierz pomyłek najlepszego modelu na zbiorze testowym."""
    estimators = dict(all_estimators(type_filter="classifier"))
    model = build_classifier(estimators[model_name])
    model.fit(features_train, target_train)

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    ConfusionMatrixDisplay.from_estimator(
        model,
        features_test,
        target_test,
        display_labels=class_names,
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title(f"Macierz pomyłek — {model_name}")
    ax.set_xlabel("Predykcja")
    ax.set_ylabel("Prawdziwa klasa")
    fig.savefig(CONFUSION_PNG, dpi=150)
    plt.close(fig)


def main() -> None:
    # Podmień wbudowany zbiór, ustawiając ścieżkę do CSV, np. Path("dane.csv").
    # TARGET_COLUMN = None oznacza ostatnią kolumnę pliku.
    csv_path: str | Path | None = None
    target_column: str | None = None

    features, target, dataset_name, class_names = load_dataset(csv_path, target_column)
    print(
        f"Zbiór: {dataset_name} | próbki: {len(target)} | "
        f"cechy: {features.shape[1]} | klasy: {np.unique(target).size}"
    )

    features_train, features_test, target_train, target_test = split_and_scale(features, target)
    print(
        f"Podział: train={len(target_train)}, test={len(target_test)} "
        f"(test_size={TEST_SIZE}, random_state={RANDOM_STATE}, stratify=y)"
    )
    print("Skalowanie: StandardScaler (fit na train, transform na test)")
    print(
        f"Walidacja krzyżowa: StratifiedKFold(n_splits={CV_SPLITS}, "
        f"shuffle=True, random_state={RANDOM_STATE}) na zbiorze treningowym"
    )
    print()

    started = time.perf_counter()
    results, skipped = compare_classifiers(
        features_train,
        features_test,
        target_train,
        target_test,
    )
    elapsed = time.perf_counter() - started

    print_results(results)
    save_outputs(results, skipped)
    plot_ranking(results, dataset_name)

    best = results.iloc[0]
    plot_confusion_matrix(
        str(best["classifier"]),
        features_train,
        features_test,
        target_train,
        target_test,
        class_names,
    )
    print(f"Oceniono: {len(results)} | pominięto: {len(skipped)} | czas: {elapsed:.1f}s")
    print(
        f"Najlepszy: {best['classifier']} "
        f"(F1={best['f1_score']:.4f}, accuracy={best['accuracy']:.4f}, "
        f"CV={best['CV_Accuracy_Mean']:.4f}±{best['CV_Accuracy_Std']:.4f})"
    )
    print(
        f"Zapisano: {RESULTS_CSV.name}, {SKIPPED_LOG.name}, "
        f"{RANKING_PNG.name}, {CONFUSION_PNG.name}"
    )


if __name__ == "__main__":
    main()
