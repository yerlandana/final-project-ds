"""
Прогноз для одного клиента и объяснение, почему такой риск.

Здесь только ML, никакого LLM. Все числа считает модель, а языковая модель
потом просто пересказывает их словами (это в ai_explain.py).
"""
import joblib
import pandas as pd

from config import MODEL_PATH
from data import load_clean
from features import add_features

# Что компания может предложить клиенту. Для каждого варианта смотрим,
# как изменится вероятность, если поменять эти поля.
RETENTION_LEVERS = {
    "Перевести на годовой контракт": {"Contract": "One year"},
    "Перевести на двухлетний контракт": {"Contract": "Two year"},
    "Подключить техподдержку бесплатно": {"TechSupport": "Yes"},
    "Подключить онлайн-защиту": {"OnlineSecurity": "Yes"},
    "Перевести на автоплатёж с карты": {"PaymentMethod": "Credit card (automatic)"},
    "Дать скидку 15% на тариф": {"__discount__": 0.15},
}

# Названия колонок по-русски, чтобы показывать их пользователю
FEATURE_RU = {
    "Contract": "тип контракта", "tenure": "срок обслуживания",
    "MonthlyCharges": "ежемесячный платёж", "TotalCharges": "сумма всех платежей",
    "InternetService": "тип интернета", "PaymentMethod": "способ оплаты",
    "TechSupport": "техподдержка", "OnlineSecurity": "онлайн-защита",
    "OnlineBackup": "облачный бэкап", "DeviceProtection": "защита устройства",
    "StreamingTV": "ТВ-стриминг", "StreamingMovies": "онлайн-кинотеатр",
    "PaperlessBilling": "электронный счёт", "SeniorCitizen": "пенсионер",
    "Partner": "есть партнёр", "Dependents": "есть иждивенцы",
    "PhoneService": "телефония", "MultipleLines": "несколько линий",
    "gender": "пол", "num_services": "количество услуг",
    "avg_monthly_spend": "средний чек", "price_trend": "динамика счёта",
    "price_per_service": "цена за услугу", "is_new": "новый клиент",
    "no_protection": "нет защиты и поддержки", "auto_payment": "автоплатёж",
}

# грузим модель и считаем "среднего клиента" один раз, а не на каждый запрос
_model = None
_baseline = None


def load_model():
    """Загружаем model.pkl. Второй раз уже берём из памяти."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Модель не найдена: " + str(MODEL_PATH) +
                ". Сначала запустите: python src/train.py"
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def population_baseline():
    """
    Типичный клиент: медиана для чисел, самое частое значение для категорий.

    Нужен, чтобы понять, что именно у конкретного клиента повышает риск:
    подменяем его поле на типичное и смотрим, насколько упадёт вероятность.
    """
    global _baseline
    if _baseline is None:
        df = load_clean().drop(columns=["Churn"])
        base = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                base[col] = float(df[col].median())
            else:
                base[col] = df[col].mode().iloc[0]
        _baseline = base
    return _baseline


def _to_frame(customer):
    """Словарь превращаем в таблицу из одной строки и добавляем признаки."""
    return add_features(pd.DataFrame([customer]))


def predict_proba(customer):
    """Вероятность того, что клиент уйдёт."""
    model = load_model()
    return float(model["pipeline"].predict_proba(_to_frame(customer))[:, 1][0])


def risk_level(proba, threshold):
    """Переводим вероятность в три понятные категории."""
    if proba >= threshold:
        return "высокий"
    if proba >= threshold * 0.6:
        return "средний"
    return "низкий"


def risk_drivers(customer, top_n=5):
    """
    Почему у этого клиента такой риск.

    Берём каждое поле, подменяем на типичное по базе и заново спрашиваем
    модель. Насколько упала вероятность - настолько это поле и виновато.
    Получается что-то вроде SHAP, только проще.
    """
    base_proba = predict_proba(customer)
    baseline = population_baseline()

    effects = []
    for col, typical in baseline.items():
        if col not in customer or customer[col] == typical:
            continue
        modified = dict(customer)
        modified[col] = typical
        delta = base_proba - predict_proba(modified)
        effects.append({
            "feature": col,
            "feature_ru": FEATURE_RU.get(col, col),
            "value": customer[col],
            "typical": typical,
            "effect": round(float(delta), 4),   # больше нуля - повышает риск
        })

    effects.sort(key=lambda e: -e["effect"])
    return effects[:top_n]


def retention_actions(customer, top_n=3):
    """
    Что предложить клиенту. Считаем "а что если" для каждого варианта.

    Это самое полезное для бизнеса: не просто "уйдёт с вероятностью 0.89",
    а "предложите двухлетний контракт и риск станет 0.64".
    """
    base_proba = predict_proba(customer)
    results = []
    for name, changes in RETENTION_LEVERS.items():
        modified = dict(customer)
        if "__discount__" in changes:
            d = changes["__discount__"]
            modified["MonthlyCharges"] = float(customer["MonthlyCharges"]) * (1 - d)
        else:
            # если у клиента уже так и есть, предлагать нечего
            if all(str(customer.get(k)) == str(v) for k, v in changes.items()):
                continue
            modified.update(changes)
        new_proba = predict_proba(modified)
        results.append({
            "action": name,
            "new_proba": round(new_proba, 4),
            "delta": round(base_proba - new_proba, 4),   # больше нуля - риск упал
        })

    results.sort(key=lambda r: -r["delta"])
    return [r for r in results if r["delta"] > 0][:top_n]


def analyze(customer):
    """Полный разбор клиента, его и отдаёт приложение."""
    model = load_model()
    threshold = model["threshold"]
    proba = predict_proba(customer)
    return {
        "proba": round(proba, 4),
        "percent": round(proba * 100, 1),
        "threshold": threshold,
        "will_churn": bool(proba >= threshold),
        "risk_level": risk_level(proba, threshold),
        "model_name": model.get("model_name", "модель"),
        "drivers": risk_drivers(customer),
        "actions": retention_actions(customer),
    }


if __name__ == "__main__":
    # рискованный клиент: новичок, помесячный контракт, дорогой интернет,
    # оплата чеком и никакой защиты
    risky = {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
        "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
        "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 95.0, "TotalCharges": 190.0,
    }
    res = analyze(risky)
    print("Вероятность ухода:", res["percent"], "%  порог:", res["threshold"],
          " риск:", res["risk_level"])
    print("\nЧто повышает риск:")
    for d in res["drivers"]:
        print("  ", d["feature_ru"], "=", d["value"], " вклад", d["effect"])
    print("\nЧто предложить:")
    for a in res["actions"]:
        print("  ", a["action"], "-> риск", a["new_proba"], "(", a["delta"], ")")
