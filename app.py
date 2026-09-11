"""
Веб-интерфейс на FastAPI.

Запуск:
    uvicorn app:app --reload
    открыть http://127.0.0.1:8000

Ноутбук показывает, что модель работает у меня на компьютере. А приложение
нужно, чтобы моделью мог пользоваться человек, который не знает Python.

Эндпоинты:
    GET  /            - форма с данными клиента
    POST /predict     - результат страницей, для человека
    POST /api/predict - то же самое в JSON, для программ
"""
import sys
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

# чтобы можно было импортировать из папки src
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ai_explain import generate_explanation  # noqa: E402
from predict import analyze, load_model      # noqa: E402

app = FastAPI(
    title="Churn Guard",
    description="Предсказание оттока клиентов телеком-оператора + AI-объяснение",
    version="1.0.0",
)
# путь абсолютный, иначе приложение не запустится из другой папки
templates = Jinja2Templates(directory=str(Path(__file__).parent / "app" / "templates"))

# Варианты для выпадающих списков. Взял ровно те, что были в данных.
CHOICES = {
    "gender": ["Female", "Male"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["Fiber optic", "DSL", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": ["Electronic check", "Mailed check",
                      "Bank transfer (automatic)", "Credit card (automatic)"],
}

# По умолчанию подставляю рискованного клиента, чтобы на защите сразу
# было видно что-то интересное, а не пустая форма.
DEFAULTS = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 95.0, "TotalCharges": 190.0,
}


class Customer(BaseModel):
    """
    Схема клиента для JSON-API.

    Pydantic проверит типы до того, как данные дойдут до модели. Если
    прислать tenure="два месяца", будет понятная ошибка, а не падение
    где-то внутри sklearn.
    """
    gender: str = "Female"
    SeniorCitizen: int = Field(0, ge=0, le=1)
    Partner: str = "No"
    Dependents: str = "No"
    tenure: int = Field(2, ge=0, le=100)
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    InternetService: str = "Fiber optic"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "Yes"
    StreamingMovies: str = "Yes"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = Field(95.0, ge=0)
    TotalCharges: float = Field(190.0, ge=0)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    bundle = load_model()
    # в новых версиях Starlette request идёт первым аргументом
    return templates.TemplateResponse(request, "index.html", {
        "choices": CHOICES, "values": DEFAULTS,
        "model_name": bundle.get("model_name"), "threshold": bundle["threshold"],
        "roc_auc": round(bundle.get("test_roc_auc", 0), 3),
    })


@app.post("/predict", response_class=HTMLResponse)
def predict_form(
    request: Request,
    gender: str = Form(...), SeniorCitizen: int = Form(...),
    Partner: str = Form(...), Dependents: str = Form(...),
    tenure: int = Form(...), PhoneService: str = Form(...),
    MultipleLines: str = Form(...), InternetService: str = Form(...),
    OnlineSecurity: str = Form(...), OnlineBackup: str = Form(...),
    DeviceProtection: str = Form(...), TechSupport: str = Form(...),
    StreamingTV: str = Form(...), StreamingMovies: str = Form(...),
    Contract: str = Form(...), PaperlessBilling: str = Form(...),
    PaymentMethod: str = Form(...), MonthlyCharges: float = Form(...),
    TotalCharges: float = Form(...),
):
    customer = {
        "gender": gender, "SeniorCitizen": SeniorCitizen, "Partner": Partner,
        "Dependents": Dependents, "tenure": tenure, "PhoneService": PhoneService,
        "MultipleLines": MultipleLines, "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity, "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection, "TechSupport": TechSupport,
        "StreamingTV": StreamingTV, "StreamingMovies": StreamingMovies,
        "Contract": Contract, "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod, "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges,
    }
    result = analyze(customer)
    explanation = generate_explanation(customer, result)
    return templates.TemplateResponse(request, "result.html", {
        "choices": CHOICES, "values": customer,
        "r": result, "explanation": explanation,
    })


@app.post("/api/predict")
def predict_api(customer: Customer):
    """То же самое, но JSON. Пример запроса есть в README."""
    data = customer.model_dump()
    result = analyze(data)
    result["explanation"] = generate_explanation(data, result)
    return JSONResponse(result)


@app.get("/health")
def health():
    """Проверка, что сервис работает и модель загрузилась."""
    bundle = load_model()
    return {"status": "ok", "model": bundle.get("model_name"),
            "threshold": bundle["threshold"]}
