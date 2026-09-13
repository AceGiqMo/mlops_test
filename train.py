import pickle
from datetime import datetime, timezone

import sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

X, y = load_iris(return_X_y=True)
model = LogisticRegression(max_iter=200)
model.fit(X, y)

artifact = {
    "model": model,
    "metadata": {
        "model_version": "1.0.0",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "feature_names": FEATURE_NAMES,
        "target_names": load_iris().target_names.tolist(),
        "train_accuracy": float(model.score(X, y)),
    },
}

with open("model.pkl", "wb") as f:
    pickle.dump(artifact, f)
print("Saved:", artifact["metadata"])