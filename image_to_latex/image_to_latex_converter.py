import json
from pathlib import Path
from typing import Union

import torch
from PIL import Image
from torchvision import transforms

from image_to_latex.models import ResnetTransformer
from image_to_latex.utils.data import Tokenizer


ARTIFACTS_DIRNAME = Path(__file__).resolve().parents[1] / "artifacts"


class ImageToLatexConverter:
    def __init__(self):
        with open(ARTIFACTS_DIRNAME / "token_to_index.json") as f:
            token_to_index = json.load(f)
        self.tokenizer = Tokenizer(token_to_index)
        with open(ARTIFACTS_DIRNAME / "config.json") as f:
            args = json.load(f)
        checkpoint = torch.load(
            ARTIFACTS_DIRNAME / "model.pth",
            map_location=torch.device("cpu"),
        )
        self.model = ResnetTransformer(self.tokenizer, args)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        self.transform = transforms.ToTensor()

    def predict(self, image: Union[str, Path, Image.Image]) -> str:
        if not isinstance(image, Image.Image):
            image_pil = Image.open(image).convert(mode="L")
        else:
            image_pil = image
        image_tensor = self.transform(image_pil).unsqueeze(0)
        y_pred = self.model.predict(
            image_tensor, beam_width=5, max_output_len=150
        )
        tokens = self.tokenizer.unindex(y_pred, inference=True)[0]
        formula = " ".join(tokens)
        return formula
