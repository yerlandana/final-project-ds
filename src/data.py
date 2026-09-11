"""
Загрузка и очистка данных.

Очистку вынес в функцию, а не оставил в ноутбуке, потому что те же самые
шаги нужны в веб-приложении для нового клиента. Если делать это в двух
местах руками, рано или поздно сделаешь по-разному.
"""
import pandas as pd

from config import RAW_DATA, TARGET, POSITIVE_LABEL


def load_raw(path=RAW_DATA):
    """Читаем csv как есть, ничего не меняем."""
    return pd.read_csv(path)


def clean(df, drop_duplicates=True):
    """
    Приводим данные в порядок. Что делаем и почему:

    1) Убираем кавычки из значений.
       Датасет скачан с OpenML в формате ARFF, поэтому значения с пробелом
       приехали как "'Fiber optic'". Если не убрать, получится две разные
       категории вместо одной.

    2) TotalCharges делаем числом.
       Колонка пришла текстом, потому что у 11 клиентов там пробел ' '.
       У всех этих 11 tenure = 0, то есть они только подключились и ещё
       ни разу не платили. Значит правильное значение - 0, а не медиана.

    3) Churn делаем 0/1, потому что sklearn работает с числами.
       1 - это "ушёл", то есть то, что мы ищем.

    4) Удаляем полные дубликаты.
       customerID в датасете нет, поэтому 22 одинаковые строки не отличить
       друг от друга. Если одна и та же строка попадёт в train и в test,
       метрика будет завышенной (утечка данных). Теряем 0.3% строк, но
       зато оценка честная.
    """
    df = df.copy()

    # 1) кавычки
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip().str.strip("'").str.strip()

    # 2) TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    new_clients = df["tenure"] == 0
    df.loc[new_clients & df["TotalCharges"].isna(), "TotalCharges"] = 0.0

    # 3) таргет
    df[TARGET] = (df[TARGET] == POSITIVE_LABEL).astype(int)

    # 4) дубликаты
    if drop_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)

    return df


def load_clean(path=RAW_DATA):
    """Сразу чистые данные, чтобы не писать две строчки каждый раз."""
    return clean(load_raw(path))


if __name__ == "__main__":
    raw = load_raw()
    df = load_clean()
    print("сырые данные:  ", raw.shape)
    print("после очистки: ", df.shape)
    print("пропусков:     ", df.isna().sum().sum())
    print("доля ушедших:  ", round(df[TARGET].mean(), 3))
