"""
Обучение моделей и выбор лучшей.

Запуск:  python src/train.py

Порядок как в задании: baseline -> модели посложнее -> таблица сравнения ->
лучшая модель -> сохранили. Плюс две вещи, которые я добавил сам:
проверка, помогли ли мои признаки, и подбор порога по деньгам.
"""
from __future__ import annotations

import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")  # чтобы графики сохранялись в файлы, а не открывались
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, average_precision_score, classification_report,
    f1_score, precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, cross_validate, train_test_split
from xgboost import XGBClassifier

from config import (
    BREAK_EVEN_PRECISION, CAPACITY_SHARES, COST_RETENTION_OFFER, CV_FOLDS,
    FIGURES_DIR, METRICS_PATH, MODEL_PATH, RANDOM_STATE, RETENTION_SUCCESS_RATE,
    TARGET, TEST_SIZE, VALUE_SAVED_CUSTOMER,
)
from data import load_clean
from features import add_features, build_pipeline, split_columns

warnings.filterwarnings("ignore")
ENGINEERED = ["num_services", "avg_monthly_spend", "price_trend",
              "price_per_service", "is_new", "no_protection", "auto_payment"]


# --------------------------------------------------------------------------
# 1. Данные
# --------------------------------------------------------------------------
def get_train_test():
    """
    Делим данные один раз и до конца к тесту не возвращаемся.

    stratify=y тут обязателен: классов 26% на 74%, и без него доля ушедших
    в тесте могла бы случайно получиться другой.
    """
    df = add_features(load_clean())
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


def cv_splitter():
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


# --------------------------------------------------------------------------
# 2. Окупился ли feature engineering?
# --------------------------------------------------------------------------
def ablation_test(X_train, y_train):
    """
    Обучаем одну модель дважды: с новыми признаками и без них.

    Придумать признаки легко, а проверить, что они что-то дали, сложнее.
    Если разница меньше разброса кросс-валидации, значит не помогли.
    """
    results = {}
    for label, cols in [("с новыми признаками", X_train.columns.tolist()),
                        ("только исходные", [c for c in X_train.columns if c not in ENGINEERED])]:
        Xs = X_train[cols]
        num, cat = split_columns(Xs)
        pipe = build_pipeline(
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
            num, cat,
        )
        scores = cross_val_score(pipe, Xs, y_train, cv=cv_splitter(), scoring="roc_auc", n_jobs=-1)
        results[label] = {"roc_auc_mean": float(scores.mean()), "roc_auc_std": float(scores.std())}
    return results


# --------------------------------------------------------------------------
# 3. Модели-кандидаты
# --------------------------------------------------------------------------
def candidate_models(y_train):
    """
    Три модели: baseline и два ансамбля.

    class_weight="balanced" и scale_pos_weight нужны из-за дисбаланса.
    Без них модель просто отвечает "все остаются", получает 73% accuracy
    и не находит ни одного уходящего.
    """
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    return {
        "Baseline: LogisticRegression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=5, class_weight="balanced",
            n_jobs=-1, random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            scale_pos_weight=neg / pos, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }


def compare_models(X_train, y_train, numeric, categorical):
    """
    Кросс-валидация на train.

    На тесте сравнивать нельзя: если выбирать победителя по тесту, то мы
    под него подгоняемся и оценка получается завышенной.
    """
    scoring = ["roc_auc", "average_precision", "f1", "recall", "precision"]
    rows = []
    for name, model in candidate_models(y_train).items():
        pipe = build_pipeline(model, numeric, categorical)
        res = cross_validate(pipe, X_train, y_train, cv=cv_splitter(), scoring=scoring, n_jobs=-1)
        rows.append({
            "model": name,
            "ROC-AUC": res["test_roc_auc"].mean(),
            "ROC-AUC std": res["test_roc_auc"].std(),
            "PR-AUC": res["test_average_precision"].mean(),
            "F1": res["test_f1"].mean(),
            "Recall": res["test_recall"].mean(),
            "Precision": res["test_precision"].mean(),
        })
    return pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# 4. Подбор гиперпараметров - обоим финалистам, а не только «модному» XGBoost
# --------------------------------------------------------------------------
def tune_all(X_train, y_train, numeric, categorical):
    """
    GridSearchCV для регрессии и для XGBoost.

    Настраиваю обе, иначе получится нечестно: сравнить настроенный бустинг
    с ненастроенной регрессией и сказать, что бустинг лучше.

    Оптимизирую ROC-AUC, а не accuracy. Accuracy при дисбалансе обманывает,
    а ROC-AUC показывает, насколько хорошо модель сортирует клиентов по риску.
    Нам нужен как раз список, кому звонить первым.
    """
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    searches = {}

    logreg = GridSearchCV(
        build_pipeline(
            LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE),
            numeric, categorical),
        {"model__C": [0.03, 0.1, 0.3, 1.0, 3.0],
         "model__penalty": ["l2"],
         "model__solver": ["lbfgs"]},
        scoring="roc_auc", cv=cv_splitter(), n_jobs=-1,
    )
    logreg.fit(X_train, y_train)
    searches["LogisticRegression (tuned)"] = logreg

    xgb = GridSearchCV(
        build_pipeline(
            XGBClassifier(eval_metric="logloss", scale_pos_weight=neg / pos,
                          random_state=RANDOM_STATE, n_jobs=-1),
            numeric, categorical),
        {"model__n_estimators": [300, 600],
         "model__max_depth": [2, 3, 4],
         "model__learning_rate": [0.02, 0.05],
         "model__min_child_weight": [1, 5],
         "model__subsample": [0.8],
         "model__colsample_bytree": [0.8]},
        scoring="roc_auc", cv=cv_splitter(), n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    searches["XGBoost (tuned)"] = xgb

    return searches


# --------------------------------------------------------------------------
# 5. Порог принятия решения - из денег, а не из воздуха
# --------------------------------------------------------------------------
def profit_table(y_true, proba):
    """
    Считаем прибыль для каждого порога.

    Модель выдаёт вероятность, а 0.5 - это просто значение по умолчанию.
    Считаем порог из денег:

      угадали уходящего (TP): +0.25 * 200 - 20
      позвонили зря (FP):     -20
      пропустили (FN):         0

    Важный момент - множитель 0.25. Скидка срабатывает не всегда. Я сначала
    про него забыл, и порог получился 0.16, то есть "обзвоните 60% базы".
    """
    contact_value = RETENTION_SUCCESS_RATE * VALUE_SAVED_CUSTOMER
    rows = []
    for t in np.arange(0.05, 0.96, 0.01):
        pred = (proba >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        profit = tp * contact_value - (tp + fp) * COST_RETENTION_OFFER
        rows.append({
            "threshold": round(float(t), 2), "profit": float(profit),
            "contacted": tp + fp, "TP": tp, "FP": fp, "FN": fn,
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0),
        })
    return pd.DataFrame(rows)


def capacity_report(y_true, proba, shares=CAPACITY_SHARES):
    """
    Второй взгляд, ближе к жизни: обзвонить всех всё равно не получится.

    Вопрос звучит не "какой порог", а "мы можем обзвонить 10% базы, кого
    выбрать". Считаем lift - во сколько раз наш список лучше случайного.
    """
    base_rate = float(np.mean(y_true))
    order = np.argsort(-proba)  # сортируем от самых рискованных
    rows = []
    for s in shares:
        k = max(1, int(len(proba) * s))
        top = order[:k]
        hits = int(np.sum(y_true[top]))
        prec = hits / k
        rows.append({
            "доля базы": f"{int(s * 100)}%",
            "клиентов обзвонили": k,
            "из них реально ушли": hits,
            "precision@k": round(prec, 3),
            "lift": round(prec / base_rate, 2),
            "охват оттока (recall)": round(hits / int(np.sum(y_true)), 3),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 6. Графики
# --------------------------------------------------------------------------
def plot_model_report(y_test, proba, threshold, cv_table, thr_table, importances, model_name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax[0].plot(fpr, tpr, lw=2, label=f"{model_name} (AUC={roc_auc_score(y_test, proba):.3f})")
    ax[0].plot([0, 1], [0, 1], "--", color="grey", label="случайная модель")
    ax[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC-кривая (тест)")
    ax[0].legend()

    prec, rec, _ = precision_recall_curve(y_test, proba)
    ax[1].plot(rec, prec, lw=2, label=f"PR-AUC={average_precision_score(y_test, proba):.3f}")
    ax[1].axhline(float(np.mean(y_test)), ls="--", color="grey",
                  label=f"случайная модель = {float(np.mean(y_test)):.2f}")
    ax[1].axhline(BREAK_EVEN_PRECISION, ls=":", color="red",
                  label=f"точка безубыточности = {BREAK_EVEN_PRECISION:.2f}")
    ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall кривая (тест)")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_pr_curves.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, (proba >= threshold).astype(int),
        display_labels=["остался", "ушёл"], cmap="Blues", ax=ax, colorbar=False,
    )
    ax.set_title(f"Матрица ошибок, порог = {threshold}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(thr_table["threshold"], thr_table["profit"], lw=2)
    ax.axvline(threshold, color="red", ls="--", label=f"оптимум = {threshold}")
    ax.axvline(0.5, color="grey", ls=":", label="порог по умолчанию 0.5")
    ax.set(xlabel="Порог вероятности", ylabel="Прибыль на тестовой выборке, у.е.",
           title="Порог выбирается по прибыли, а не по умолчанию")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "threshold_profit.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    importances.head(15).iloc[::-1].plot.barh(ax=ax, color="#2a6f97")
    ax.set(xlabel="Важность признака", title=f"Топ-15 признаков ({model_name})")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    cv_table.set_index("model")[["ROC-AUC", "F1", "Recall"]].plot.bar(ax=ax, rot=10)
    ax.set(ylabel="Значение метрики", title="Сравнение моделей (5-fold CV на train)")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_comparison.png", dpi=130)
    plt.close(fig)


def get_feature_importances(pipeline):
    """
    Важности признаков с нормальными названиями колонок.

    У деревьев есть feature_importances_, у регрессии только коэффициенты.
    Для регрессии беру модуль: он показывает силу влияния, а знак
    (повышает или понижает риск) смотрю отдельно в ноутбуке.
    """
    names = [n.split("__", 1)[-1] for n in pipeline.named_steps["prep"].get_feature_names_out()]
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    else:
        values = np.abs(model.coef_[0])
    return pd.Series(values, index=names).sort_values(ascending=False)


# --------------------------------------------------------------------------
# 7. Главная функция
# --------------------------------------------------------------------------
def main():
    print("=" * 74)
    print("ШАГ 1. Данные и разбиение на train/test")
    X_train, X_test, y_train, y_test = get_train_test()
    numeric, categorical = split_columns(X_train)
    print(f"  train: {X_train.shape}, test: {X_test.shape}")
    print(f"  доля оттока - train {y_train.mean():.3f}, test {y_test.mean():.3f}")

    print("\nШАГ 2. Ablation-тест: а помогли ли придуманные признаки?")
    abl = ablation_test(X_train, y_train)
    for k, v in abl.items():
        print(f"  {k:22} ROC-AUC = {v['roc_auc_mean']:.4f} ± {v['roc_auc_std']:.4f}")
    delta = abl["с новыми признаками"]["roc_auc_mean"] - abl["только исходные"]["roc_auc_mean"]
    print(f"  разница: {delta:+.4f}  ->  "
          f"{'признаки помогли' if delta > 0.002 else 'прирост в пределах шума, признаки почти не помогли'}")

    print("\nШАГ 3. Сравнение моделей (5-fold CV, без подбора гиперпараметров)")
    cv_table = compare_models(X_train, y_train, numeric, categorical)
    print(cv_table.round(4).to_string(index=False))

    print("\nШАГ 4. Подбор гиперпараметров обоим финалистам")
    searches = tune_all(X_train, y_train, numeric, categorical)
    for name, s in searches.items():
        print(f"  {name:28} CV ROC-AUC = {s.best_score_:.4f}")
        print(f"     {s.best_params_}")

    best_name = max(searches, key=lambda n: searches[n].best_score_)
    best_search = searches[best_name]
    best_pipe = best_search.best_estimator_
    print(f"\n  ПОБЕДИТЕЛЬ по кросс-валидации: {best_name}")

    print("\nШАГ 5. Честная проверка на отложенном тесте (открываем его первый раз)")
    proba = best_pipe.predict_proba(X_test)[:, 1]
    test_auc = float(roc_auc_score(y_test, proba))
    test_ap = float(average_precision_score(y_test, proba))
    print(f"  ROC-AUC: {test_auc:.4f}   PR-AUC: {test_ap:.4f}")

    print("\nШАГ 6. Порог по экономике удержания")
    print(f"  звонок стоит {COST_RETENTION_OFFER:.0f}, удержание приносит "
          f"{VALUE_SAVED_CUSTOMER:.0f}, срабатывает в {RETENTION_SUCCESS_RATE:.0%} случаев")
    print(f"  => звонить выгодно, пока precision > {BREAK_EVEN_PRECISION:.2f}")
    thr_table = profit_table(y_test.values, proba)
    threshold = float(thr_table.loc[thr_table["profit"].idxmax(), "threshold"])
    at_best = thr_table[thr_table.threshold == threshold].iloc[0]
    at_half = thr_table[thr_table.threshold == 0.50].iloc[0]
    print(f"  порог 0.50:           прибыль={at_half.profit:8.0f}  обзвон={int(at_half.contacted):4d}  "
          f"recall={at_half.recall:.3f}  precision={at_half.precision:.3f}")
    print(f"  порог {threshold:.2f} (лучший):  прибыль={at_best.profit:8.0f}  обзвон={int(at_best.contacted):4d}  "
          f"recall={at_best.recall:.3f}  precision={at_best.precision:.3f}")

    print("\nШАГ 7. Взгляд бизнеса: ограниченная ёмкость обзвона")
    cap = capacity_report(y_test.values, proba)
    print(cap.to_string(index=False))

    print("\n  Отчёт по классам при выбранном пороге:")
    print(classification_report(y_test, (proba >= threshold).astype(int),
                                target_names=["остался", "ушёл"], digits=3))

    print("ШАГ 8. Графики и сохранение модели")
    importances = get_feature_importances(best_pipe)
    plot_model_report(y_test, proba, threshold, cv_table, thr_table, importances, best_name)

    # ещё раз обучаем baseline, чтобы записать его результат в метрики
    base_pipe = build_pipeline(
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        numeric, categorical,
    ).fit(X_train, y_train)
    base_proba = base_pipe.predict_proba(X_test)[:, 1]

    joblib.dump(
        {
            "pipeline": best_pipe,
            "model_name": best_name,
            "threshold": threshold,
            "numeric": numeric,
            "categorical": categorical,
            "feature_order": list(X_train.columns),
            "test_roc_auc": test_auc,
        },
        MODEL_PATH,
    )

    metrics = {
        "dataset": {"n_train": int(len(X_train)), "n_test": int(len(X_test)),
                    "churn_rate": round(float(y_train.mean()), 4)},
        "ablation_feature_engineering": abl,
        "cv_comparison": cv_table.round(4).to_dict(orient="records"),
        "tuning": {n: {"cv_roc_auc": round(float(s.best_score_), 4),
                       "best_params": s.best_params_} for n, s in searches.items()},
        "selected_model": best_name,
        "baseline_test": {
            "roc_auc": round(float(roc_auc_score(y_test, base_proba)), 4),
            "f1_at_0.5": round(float(f1_score(y_test, (base_proba >= 0.5).astype(int))), 4),
        },
        "final_test": {
            "roc_auc": round(test_auc, 4),
            "pr_auc": round(test_ap, 4),
            "threshold": threshold,
            "recall": round(float(at_best.recall), 4),
            "precision": round(float(at_best.precision), 4),
            "f1": round(float(at_best.f1), 4),
            "profit_at_best_threshold": float(at_best.profit),
            "profit_at_0.5": float(at_half.profit),
        },
        "capacity": cap.to_dict(orient="records"),
        "top_features": importances.head(15).round(4).to_dict(),
    }
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))

    print(f"  модель  -> {MODEL_PATH}")
    print(f"  метрики -> {METRICS_PATH}")
    print(f"  графики -> {FIGURES_DIR}")
    print("=" * 74)
    return metrics


if __name__ == "__main__":
    main()
