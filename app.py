from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="Credit Risk Multi Model API")

# -----------------------------
# Load Models
# -----------------------------
models = {
    "logistic": joblib.load("Models/logistic_pipeline.pkl"),
    "random_forest": joblib.load("Models/rf_pipeline.pkl"),
    "xgboost": joblib.load("Models/xgb_pipeline.pkl"),
    "ann": joblib.load("Models/best_model.pkl")
}

# -----------------------------
# Input Schema (RAW FEATURES)
# -----------------------------
class CreditInput(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float
    age: float
    NumberOfTime30_59DaysPastDueNotWorse: float
    DebtRatio: float
    MonthlyIncome: float
    NumberOfOpenCreditLinesAndLoans: float
    NumberOfTimes90DaysLate: float
    NumberRealEstateLoansOrLines: float
    NumberOfTime60_89DaysPastDueNotWorse: float
    NumberOfDependents: float


@app.get("/")
def home():
    return {"message": "Credit Risk API is running 🚀"}


@app.post("/predict/{model_name}")
def predict(model_name: str, data: CreditInput):

    if model_name not in models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model name. Choose from {list(models.keys())}"
        )

    model = models[model_name]

    # Convert to dictionary
    input_dict = data.dict()

    # Rename safe names to original dataset names
    input_dict["NumberOfTime30-59DaysPastDueNotWorse"] = input_dict.pop("NumberOfTime30_59DaysPastDueNotWorse")
    input_dict["NumberOfTime60-89DaysPastDueNotWorse"] = input_dict.pop("NumberOfTime60_89DaysPastDueNotWorse")

    df = pd.DataFrame([input_dict])

    # -----------------------------
    # FEATURE ENGINEERING (MATCH NOTEBOOK)
    # -----------------------------
    df["TotalLatePayments"] = (
        df["NumberOfTimes90DaysLate"] +
        df["NumberOfTime30-59DaysPastDueNotWorse"] +
        df["NumberOfTime60-89DaysPastDueNotWorse"]
    )

    df["IncomePerDependent"] = df["MonthlyIncome"] / (df["NumberOfDependents"] + 1)

    df["LogIncome"] = np.log1p(df["MonthlyIncome"])
    df["LogDebtRatio"] = np.log1p(df["DebtRatio"])

    # Drop columns that were removed in notebook
    df.drop(columns=[
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse"
    ], inplace=True)

    # Ensure correct column order
    df = df.reindex(columns=model.feature_names_in_)

    prediction = model.predict(df)

    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(df)[0][1])
    else:
        probability = None

    return {
        "model_used": model_name,
        "prediction": int(prediction[0]),
        "probability": probability
    }