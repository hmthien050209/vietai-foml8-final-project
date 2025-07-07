from urllib.parse import ResultBase

from fastapi import FastAPI
from scalar_fastapi import get_scalar_api_reference

from shared.model import Model
from shared.predict_item import PredictItem

app = FastAPI()
model = Model('../data/heart_disease_uci.csv')


@app.get("/")
def read_root():
    return "Hello, World!"


@app.post("/train")
def train():
    try:
        model.train()
        return {"status": "success", "f1": model.f1, "precision": model.precision, "recall": model.recall,
            "last_updated": model.last_updated}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/predict")
def predict(predict_item: PredictItem):
    if not model.trained:
        return {"status": "error", "result": "Model not trained"}
    else:
        try:
            return {"status": "success", "result": model.predict(predict_item)}
        except Exception as e:
            return {"status": "error", "error": str(e)}


@app.get("/model_info")
def last_updated():
    if not model.trained:
        return {"status": "Not Trained"}
    else:
        return {"status": "Trained", "last_updated": model.last_updated, "f1": model.f1, "precision": model.precision, "recall": model.recall}


@app.get("/scalar", include_in_schema=False)
async def scalar_html():
    return get_scalar_api_reference(
        openapi_url=app.openapi_url,
        title=app.title,
    )
