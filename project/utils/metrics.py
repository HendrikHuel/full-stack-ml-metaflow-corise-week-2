# auxiliary libary for experimental usage

from typing import Tuple
from sklearn.metrics import accuracy_score, roc_auc_score

def calc_scores(valdf: pd.DataFrame, model_col: str = "dummy_model") -> Tuple[float, float]:
    """Calculate and display ACC and ROC."""

    acc = accuracy_score(valdf['label'], valdf[model_col])
    roc = roc_auc_score(valdf['label'], valdf[model_col])
    print(f"ACC: {acc:.2f}")
    print(f"ROC: {roc:.2f}")

    return acc, roc
