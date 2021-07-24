from http import HTTPStatus

import numpy as np
from albumentations.pytorch.transforms import ToTensorV2
from fastapi import FastAPI, File, UploadFile
from PIL import Image

from image_to_latex.lit_models import LitResNetTransformer


app = FastAPI(
    title="Image to Latex Convert",
    desription="Convert an image of math equation into LaTex code.",
)


@app.on_event("startup")
async def load_model():
    global lit_model
    global transform
    lit_model = LitResNetTransformer.load_from_checkpoint("artifacts/model.pt")
    lit_model.freeze()
    transform = ToTensorV2()


@app.get("/", tags=["General"])
def read_root():
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {},
    }
    return response


@app.post("/predict/", tags=["Prediction"])
def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("L")
    image_tensor = transform(image=np.array(image))["image"]  # type: ignore
    pred = lit_model.model.predict(image_tensor.unsqueeze(0).float())[0]  # type: ignore
    decoded = lit_model.tokenizer.decode(pred.tolist())  # type: ignore
    decoded_str = " ".join(decoded)
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"pred": decoded_str},
    }
    return response
