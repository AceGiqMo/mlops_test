import pickle
import mlflow
import mlflow.sklearn
from datetime import datetime, timezone
import os

import sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Настройка MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_URI", "http://localhost:5000"))
mlflow.set_experiment("iris-classification")

FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# Загружаем данные
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Начинаем эксперимент
with mlflow.start_run(run_name=f"logistic_regression_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
    # Логируем параметры
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_iter", 200)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("test_size", 0.3)

    # Обучаем модель
    model = LogisticRegression(max_iter=200)
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
        registered_model_name="iris-model"  # автоматически регистрирует в Model Registry
    )

    # Сохраняем метаданные в старый формат (для совместимости с main.py)
    artifact = {
        "model": model,
        "metadata": {
            "model_version": mlflow.active_run().info.run_id,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "sklearn_version": sklearn.__version__,
            "feature_names": FEATURE_NAMES,
            "target_names": load_iris().target_names.tolist(),
            "train_accuracy": float(train_accuracy),
            "test_accuracy": float(test_accuracy),
        },
    }

    with open("model.pkl", "wb") as f:
        pickle.dump(artifact, f)

    # Логируем pickle как дополнительный артефакт
    mlflow.log_artifact("model.pkl")

    print(f"Experiment logged:")
    print(f"  Train accuracy: {train_accuracy:.3f}")
    print(f"  Test accuracy: {test_accuracy:.3f}")
    print(f"  Run ID: {mlflow.active_run().info.run_id}")