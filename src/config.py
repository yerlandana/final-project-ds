"""
Все настройки проекта в одном файле, чтобы не искать их по коду.
"""
from pathlib import Path

# папка проекта (на уровень выше, чем src)
ROOT = Path(__file__).resolve().parent.parent

RAW_DATA = ROOT / "data" / "raw" / "telco_churn.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"

MODEL_PATH = MODELS_DIR / "model.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"

TARGET = "Churn"
POSITIVE_LABEL = "Yes"   # значит, что клиент ушёл

RANDOM_STATE = 42        # чтобы результат каждый раз был одинаковый
TEST_SIZE = 0.20
CV_FOLDS = 5

# Деньги. Нужны, чтобы посчитать порог (см. train.py).
# Цифры я взял условные, в реальности их дал бы финансовый отдел.
COST_RETENTION_OFFER = 20.0    # сколько стоит позвонить и дать скидку
VALUE_SAVED_CUSTOMER = 200.0   # сколько приносит удержанный клиент
RETENTION_SUCCESS_RATE = 0.25  # как часто скидка реально срабатывает

# Отсюда получается минимальная precision, при которой звонить ещё выгодно:
# 20 / (0.25 * 200) = 0.4
BREAK_EVEN_PRECISION = COST_RETENTION_OFFER / (RETENTION_SUCCESS_RATE * VALUE_SAVED_CUSTOMER)

# сколько процентов базы отдел удержания успевает обзвонить
CAPACITY_SHARES = [0.05, 0.10, 0.20]

# создаём папки, если их ещё нет
for d in (PROCESSED_DIR, MODELS_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)
