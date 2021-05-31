from http import HTTPStatus

from fastapi import FastAPI, File, UploadFile
from PIL import Image

from image_to_latex.image_to_latex_converter import ImageToLatexConverter


app = FastAPI(
    title="Image to Latex Convert",
    desription="Convert an image of math equation into LaTex code.",
)


@app.on_event("startup")
async def load_model():
    global model
    model = ImageToLatexConverter()


@app.get("/", tags=["General"])
def read_root():
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {},
    }
    return response


@app.post("/predict/{beam_width}", tags=["Prediction"])
def get_image(beam_width: int, file: UploadFile = File(...)):
    image = Image.open(file.file).convert(mode="L")
    pred = model.predict(image, beam_width=beam_width)  # type: ignore
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"pred": pred},
    }
    return response
