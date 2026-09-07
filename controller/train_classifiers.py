"""
train_classifiers.py
====================
Trains several candidate classifiers on the rule-teacher-labeled
dataset (see collect_dataset.py) to predict KEEP(0)/REQUEST_NEXT(1)
from the same 14 nn_features used elsewhere in ASTRID, compares them on
a held-out validation set (built from DIFFERENT scenarios than
training), and saves every candidate plus a pointer to the best one.

Candidates are plain scikit-learn classifiers, saved via joblib -- the
same persistence approach the existing HGB queue estimator already uses
(models/persistence.py's load_model loads a .joblib file).

Best model is selected by validation F1-macro (robust to the KEEP/
REQUEST_NEXT class imbalance you'll typically see), not raw accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def load_dataset(path: Path):
    data = np.load(path, allow_pickle=True)
    return data["X"], data["y"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare candidate ASTRID controller classifiers.")
    parser.add_argument("--train-dataset", type=str, required=True)
    parser.add_argument("--val-dataset", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="controller/forest_models")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    X_train, y_train = load_dataset(Path(args.train_dataset))
    X_val, y_val = load_dataset(Path(args.val_dataset))
    print(f"[data] train={X_train.shape[0]} rows, val={X_val.shape[0]} rows, features={X_train.shape[1]}")

    candidates = {
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=None, class_weight="balanced", random_state=args.seed, n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=args.seed),
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for name, clf in candidates.items():
        clf.fit(X_train, y_train)
        preds = clf.predict(X_val)
        acc = accuracy_score(y_val, preds)
        f1_macro = f1_score(y_val, preds, average="macro")
        results[name] = {"val_accuracy": float(acc), "val_f1_macro": float(f1_macro)}
        model_path = out_dir / f"{name}.joblib"
        joblib.dump(clf, model_path)
        print(f"[{name}] val_accuracy={acc:.4f} val_f1_macro={f1_macro:.4f} -> saved {model_path}")

    best_name = max(results, key=lambda n: results[n]["val_f1_macro"])
    best_path = out_dir / f"{best_name}.joblib"
    winner_path = out_dir / "best_model.joblib"
    joblib.dump(joblib.load(best_path), winner_path)

    summary = {"results": results, "best_model": best_name, "best_model_path": str(winner_path)}
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[best] {best_name} (val_f1_macro={results[best_name]['val_f1_macro']:.4f}) -> {winner_path}")


if __name__ == "__main__":
    main()