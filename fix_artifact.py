# fix_artifact.py  (run once from repo root: python fix_artifact.py)
import sys
from pathlib import Path

sys.path.insert(0, str(Path("models/results")))
from final_hist_gradient_boosting_evaluation import _EarlyStoppingHGB  # noqa: F401 -- import needed so pickle can resolve it

import joblib

path = Path("models/artifacts/layer2_p11/hist_gradient_boosting_layer2_p11_tuned/hist_gradient_boosting.joblib")
model = joblib.load(path)          # works now because _EarlyStoppingHGB is properly imported, not __main__
joblib.dump(model, path)           # re-save: pickle now records the real module path
print("Re-saved successfully:", type(model))