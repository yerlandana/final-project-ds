"""
AI-часть проекта.

Языковая модель получает готовые числа от ML-модели и пересказывает их
словами для менеджера. Сама она ничего не считает и ничего не предсказывает.

Почему не спросить у неё сразу "уйдёт ли клиент":
  - она не видела наши 7000 клиентов и не знает статистику оттока,
    поэтому просто выдумает правдоподобное число;
  - такой ответ нельзя измерить метрикой и нельзя повторить.

Получается разделение: модель считает "сколько", а LLM объясняет по-человечески.

Если ключа нет, работает обычный шаблон. Сделал так специально, чтобы
демо на защите не сломалось из-за интернета.
"""
import os

SYSTEM_PROMPT = """Ты помощник менеджера по удержанию клиентов телеком-оператора.
Тебе дают готовый результат ML-модели. Объясни его менеджеру на русском языке.

Правила:
1. Используй ТОЛЬКО те числа, которые тебе передали. Новые не придумывай.
2. Не давай медицинских, юридических и финансовых гарантий.
3. Ответ - ровно три абзаца без заголовков и списков:
   первый: оценка риска простыми словами;
   второй: почему модель так решила;
   третий: что предложить клиенту.
4. Максимум 130 слов. Без воды."""


def build_prompt(customer, analysis):
    """Собираем факты для LLM. Всё это посчитала ML-модель."""
    drivers = "\n".join(
        "  - " + d["feature_ru"] + ": " + str(d["value"]) +
        " (вклад в риск " + format(d["effect"], "+.2f") + ")"
        for d in analysis["drivers"]
    ) or "  - выраженных факторов риска не найдено"

    actions = "\n".join(
        "  - " + a["action"] + ": риск снизится до " +
        format(a["new_proba"] * 100, ".0f") + "% (на " +
        format(a["delta"] * 100, ".0f") + " п.п.)"
        for a in analysis["actions"]
    ) or "  - действенных предложений не найдено"

    # собираю построчно, а не одной f-строкой: многострочные вставки
    # ломают отступы и промпт приезжает лесенкой
    lines = [
        "КЛИЕНТ:",
        "  срок обслуживания: " + str(customer["tenure"]) + " мес.",
        "  контракт: " + str(customer["Contract"]),
        "  интернет: " + str(customer["InternetService"]),
        "  ежемесячный платёж: " + str(customer["MonthlyCharges"]) + " у.е.",
        "  способ оплаты: " + str(customer["PaymentMethod"]),
        "",
        "РЕЗУЛЬТАТ МОДЕЛИ (" + str(analysis["model_name"]) + "):",
        "  вероятность ухода: " + str(analysis["percent"]) + "%",
        "  порог тревоги: " + format(analysis["threshold"] * 100, ".0f") + "%",
        "  уровень риска: " + analysis["risk_level"],
        "",
        "ГЛАВНЫЕ ПРИЧИНЫ РИСКА:",
        drivers,
        "",
        "РАССЧИТАННЫЕ СЦЕНАРИИ УДЕРЖАНИЯ:",
        actions,
    ]
    return "\n".join(lines)


def _fallback_text(customer, analysis):
    """
    Версия без LLM: тот же текст, собранный по шаблону.

    Он заметно суше, и это как раз видно на защите: понятно, что именно
    добавляет языковая модель.
    """
    p = analysis["percent"]
    lvl = analysis["risk_level"]
    thr = format(analysis["threshold"] * 100, ".0f")

    if lvl == "высокий":
        head = ("Риск ухода высокий - " + str(p) + "% при пороге " + thr +
                "%. Клиента стоит взять в работу сейчас.")
    elif lvl == "средний":
        head = ("Риск умеренный - " + str(p) +
                "%. Клиент пока не в красной зоне, но требует внимания.")
    else:
        head = "Риск низкий - " + str(p) + "%. Специальных действий не требуется."

    if analysis["drivers"]:
        causes = "Основные причины: " + "; ".join(
            d["feature_ru"] + " - " + str(d["value"]) for d in analysis["drivers"][:3]
        ) + "."
    else:
        causes = "Профиль клиента близок к типичному, ярких факторов риска нет."

    if analysis["actions"]:
        best = analysis["actions"][0]
        recs = ("Рекомендация: " + best["action"].lower() +
                " - по расчёту модели риск снизится до " +
                format(best["new_proba"] * 100, ".0f") + "% (минус " +
                format(best["delta"] * 100, ".0f") + " п.п.). ")
        if len(analysis["actions"]) > 1:
            recs += "Запасные варианты: " + ", ".join(
                a["action"].lower() for a in analysis["actions"][1:]) + "."
    else:
        recs = "Действенных рычагов удержания модель не нашла."

    return head + "\n\n" + causes + "\n\n" + recs


def generate_explanation(customer, analysis, timeout=20):
    """
    Главная функция AI-части.

    Пробуем сходить в LLM. Если ключа нет, интернета нет или сервис ответил
    ошибкой - отдаём шаблон. В ответе честно пишем, кто написал текст.

    Ключ берём из переменной окружения, а не из кода: выложить его в GitHub -
    очень дорогая ошибка, боты находят такие ключи за минуты.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"text": _fallback_text(customer, analysis), "source": "offline-шаблон"}

    try:
        import requests

        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        resp = requests.post(
            base_url + "/chat/completions",
            headers={"Authorization": "Bearer " + api_key},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_prompt(customer, analysis)},
                ],
                "temperature": 0.3,   # низкая температура = меньше фантазий
                "max_tokens": 400,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        return {"text": text, "source": "LLM (" + model + ")"}

    except Exception as e:
        # причина не важна, приложение должно работать в любом случае
        return {
            "text": _fallback_text(customer, analysis),
            "source": "offline-шаблон (LLM недоступна: " + type(e).__name__ + ")",
        }


if __name__ == "__main__":
    from predict import analyze

    risky = {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
        "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
        "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 95.0, "TotalCharges": 190.0,
    }
    a = analyze(risky)
    out = generate_explanation(risky, a)
    print("--- источник текста:", out["source"], "---\n")
    print(out["text"])
