import pickle
from datetime import datetime, timezone
import os
from pathlib import Path
import pandas as pd

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import mlflow
import mlflow.sklearn

current_dir = Path(__file__).resolve().parent
MODEL_PATH = current_dir.parent.parent / "models" / "model.pkl"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# MLflow set-up
mlflow.set_tracking_uri(os.getenv("MLFLOW_URI", "http://localhost:5000"))
mlflow.set_experiment("wine-classification")

FEATURE_NAMES = ['Alcohol',
                 'Malic acid', 'Ash',
                 'Alcalinity of ash', 'Magnesium',
                 'Total phenols', 'Flavanoids',
                 'Nonflavanoid phenols',
                 'Proanthocyanins',
                 'Color intensity', 'Hue',
                 'OD280/OD315 of diluted wines',
                 'Proline']

# Data loading
data_dir = current_dir.parent.parent / "data" / "processed"
processed_files = list(data_dir.glob("*.csv"))

if not processed_files:
    print(f"[ERROR] There is no processed dataset in {data_dir}")
    exit(1)

# Take the latest file
latest_file = max(processed_files, key=lambda f: f.stat().st_mtime)
print(f"[INFO] Dataset loading: {latest_file.name}")

df = pd.read_csv(latest_file)
X = df[FEATURE_NAMES].values
y = df["Class label"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Начинаем эксперимент
with mlflow.start_run(run_name=f"logistic_regression_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
    # Логируем параметры
    mlflow.log_param("principal components", 2)
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_iter", 200)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("regularization", "L2")
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("dataset_file", latest_file.name)

    # Обучаем модель
    model = Pipeline([
        ('sc', StandardScaler()),
        ('pca', PCA(n_components=2)),
        ('clf', LogisticRegression(max_iter=200, l1_ratio=0))
    ])
    model.fit(X_train, y_train)

    # Вычисляем метрики
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    y_pred = model.predict(X_test)

    # Логируем метрики
    mlflow.log_metric("train_accuracy", train_accuracy)
    mlflow.log_metric("test_accuracy", test_accuracy)

    # Логируем модель как артефакт
    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name="wine-model"  # автоматически регистрирует в Model Registry
    )

    # Сохраняем метаданные в старый формат (для совместимости с main.py)
    artifact = {
        "model": model,
        "metadata": {
            "model_version": mlflow.active_run().info.run_id,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "sklearn_version": sklearn.__version__,
            "feature_names": FEATURE_NAMES,
            "target_names": [1, 2, 3],
            "train_accuracy": float(train_accuracy),
            "test_accuracy": float(test_accuracy),
            "dataset_source": latest_file.name
        },
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)

    # Логируем pickle как дополнительный артефакт
    mlflow.log_artifact(str(MODEL_PATH))

    print(f"Experiment logged:")
    print(f"  Train accuracy: {train_accuracy:.3f}")
    print(f"  Test accuracy: {test_accuracy:.3f}")
    print(f"  Run ID: {mlflow.active_run().info.run_id}")