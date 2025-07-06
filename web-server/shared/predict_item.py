from pydantic import BaseModel


class PredictItem(BaseModel):
    age: int
    dataset: str
    sex: str
    cp: str
    trestbps: float
    chol: float
    fbs: bool
    restecg: str
    thalch: float
    exang: bool
    oldpeak: float
    slope: str
