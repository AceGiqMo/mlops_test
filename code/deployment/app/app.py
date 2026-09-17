import os

import pandas as pd
import requests
import streamlit as st

# --- Конфигурация через переменные окружения (12-factor) ---
API_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "")

FEATURE_NAMES = ['Alcohol',
                 'Malic acid', 'Ash',
                 'Alcalinity of ash', 'Magnesium',
                 'Total phenols', 'Flavanoids',
                 'Nonflavanoid phenols',
                 'Proanthocyanins',
                 'Color intensity', 'Hue',
                 'OD280/OD315 of diluted wines',
                 'Proline']

TARGET_NAMES = [1, 2, 3]
DEFAULTS = [1., 14.23, 1.71, 2.43, 15.6, 127., 2.8,
            3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065.]

st.set_page_config(page_title="Wine Classifier", layout="centered")


def get_headers() -> dict:
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    return headers


@st.cache_data(ttl=60)
def get_model_info():
    """Информация о модели из API (кешируем на 60 сек)."""
    try:
        r = requests.get(f"{API_URL}/info", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# --- Шапка и статус модели ---
st.title("Wine Classifier")
st.caption("Streamlit → FastAPI → MLflow Model Registry")

info = get_model_info()
if info and "model_version" in info:
    acc = info.get("metrics", {}).get("test_accuracy")
    acc_str = f"{acc:.3f}" if acc is not None else "n/a"
    st.success(f"Model: **{info['model_name']}** v{info['model_version']} | test_accuracy: {acc_str}")
elif info and info.get("status") == "no_model_loaded":
    st.warning("API работает, но модель ещё не загружена. Запустите Airflow DAG или /reload.")
else:
    st.error(f"API недоступен по адресу {API_URL}")

# --- Ввод признаков ---
st.header("Input features")
cols = st.columns(13)
values = []
for col, name, default in zip(cols, FEATURE_NAMES, DEFAULTS):
    with col:
        v = st.number_input(
            name,
            min_value=0.0,
            max_value=10000.0,
            value=float(default),
            step=0.1,
            help=f"Признак {name}",
        )
        values.append(v)

# --- Кнопка Predict и вывод результата ---
if st.button("Predict", type="primary"):
    with st.spinner("Calling API..."):
        try:
            resp = requests.post(
                f"{API_URL}/predict",
                json={"features": values},
                headers=get_headers(),
                timeout=10,
            )

            if resp.status_code == 401:
                st.error("Invalid API key: проверьте переменную API_KEY.")
            elif resp.status_code == 503:
                st.error("Модель не загружена в API. Обучите модель (Airflow DAG) и сделайте /reload.")
            else:
                resp.raise_for_status()
                data = resp.json()
                pred = data["prediction"]
                probs = data["probabilities"]

                # Подписи классов: берём из модели, фоллбэк на wine-имена, затем.generic
                model_classes = (info or {}).get("target_names") or []
                if len(model_classes) == len(probs):
                    labels = model_classes
                elif len(probs) == len(TARGET_NAMES):
                    labels = TARGET_NAMES
                else:
                    labels = [f"class {i}" for i in range(len(probs))]

                st.header("Result")
                st.metric(
                    label="Predicted class",
                    value=labels[pred],
                    delta=f"class index: {pred} | model v{data['model_version']}",
                )

                df = pd.DataFrame({"class": labels, "probability": probs})
                st.bar_chart(df, x="class", y="probability")

                with st.expander("Raw API response"):
                    st.json(data)

        except requests.exceptions.ConnectionError:
            st.error(f"Не удалось подключиться к API: {API_URL}")
        except Exception as e:
            st.error(f"Ошибка: {e}")