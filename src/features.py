"""
Новые признаки и препроцессор.

Модель ничего не знает про телеком, она видит только те колонки, которые
я ей дал. Поэтому признаки, которые кажутся важными, приходится делать руками.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import TARGET

# все дополнительные услуги оператора
SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]


def add_features(df):
    """
    Добавляю 7 признаков. Идеи взял из EDA:

    num_services      - сколько услуг подключено. Если одна, уйти легко.
    avg_monthly_spend - средний чек за всё время (TotalCharges / tenure).
    price_trend       - текущий чек делить на средний. Больше 1 - счёт вырос.
    price_per_service - сколько платит за одну услугу.
    is_new            - первые 6 месяцев, в них отток самый большой.
    no_protection     - нет ни техподдержки, ни онлайн-защиты.
    auto_payment      - платит автоматически, такие уходят реже.
    """
    df = df.copy()

    # считаем только реальные "Yes", потому что есть ещё "No internet service"
    df["num_services"] = sum(
        (df[c] == "Yes").astype(int) for c in SERVICE_COLS if c in df.columns
    )

    tenure_safe = df["tenure"].clip(lower=1)  # чтобы не делить на ноль
    df["avg_monthly_spend"] = df["TotalCharges"] / tenure_safe
    df["price_trend"] = df["MonthlyCharges"] / df["avg_monthly_spend"].replace(0, np.nan)
    df["price_trend"] = df["price_trend"].fillna(1.0).clip(upper=5)
    df["price_per_service"] = df["MonthlyCharges"] / df["num_services"].clip(lower=1)

    df["is_new"] = (df["tenure"] <= 6).astype(int)
    df["no_protection"] = (
        (df["OnlineSecurity"] != "Yes") & (df["TechSupport"] != "Yes")
    ).astype(int)
    df["auto_payment"] = df["PaymentMethod"].str.contains("automatic", case=False).astype(int)

    return df


def split_columns(df):
    """Делим колонки на числа и категории. Таргет не берём."""
    X = df.drop(columns=[TARGET], errors="ignore")
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    return numeric, categorical


def build_preprocessor(numeric, categorical):
    """
    Числа масштабируем, категории кодируем через one-hot.

    Сначала я хотел сделать это через pd.get_dummies прямо в ноутбуке,
    но так нельзя: на одном клиенте получится другой набор колонок и
    модель упадёт. ColumnTransformer внутри Pipeline запоминает все
    категории на обучении и повторяет их потом.

    handle_unknown="ignore" - если придёт категория, которой не было
    в обучении, приложение не упадёт.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        remainder="drop",
    )


def build_pipeline(model, numeric, categorical):
    """Препроцессор и модель вместе, чтобы сохранить всё одним файлом."""
    return Pipeline([
        ("prep", build_preprocessor(numeric, categorical)),
        ("model", model),
    ])


if __name__ == "__main__":
    from data import load_clean

    df = add_features(load_clean())
    num, cat = split_columns(df)
    print("числовых признаков:      ", len(num))
    print("категориальных признаков:", len(cat))
    print(df[["num_services", "avg_monthly_spend", "price_trend",
              "price_per_service", "is_new", "no_protection",
              "auto_payment"]].describe().round(2).to_string())
