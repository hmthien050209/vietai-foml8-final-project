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


@app.get("/train")
def train():
    model.train()
    return {"status": "success", "f1": model.f1, "precision": model.precision, "recall": model.recall,
            "last_updated": model.last_updated}


@app.post("/predict")
def predict(predict_item: PredictItem):
    if not model.trained:
        return {"status": "error", "result": "Model not trained"}
    else:
        result = model.predict(predict_item)
        print(result)
        return {"status": "success", "result": result}


@app.get("/last_updated")
def last_updated():
    if model.trained:
        return {"last_updated": model.last_updated}
    else:
        return {"last_updated": "Not Trained"}


@app.get("/scalar", include_in_schema=False)
async def scalar_html():
    return get_scalar_api_reference(
        openapi_url=app.openapi_url,
        title=app.title,
    )
