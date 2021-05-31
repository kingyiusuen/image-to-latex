import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from torchvision import transforms

from image_to_latex.data.base_dataset import BaseDataset
from image_to_latex.utils.data import Tokenizer
from image_to_latex.utils.misc import find_max_length


FILE_DIRNAME = Path(__file__).resolve().parent


@pytest.fixture
def images():
    with open(FILE_DIRNAME / "test_data.json") as f:
        test_data = json.load(f)
    images = []
    for entry in test_data:
        filename = entry["filename"]
        image = Image.open(FILE_DIRNAME / filename).convert("L")
        images.append(image)
    return images


@pytest.fixture
def formulas():
    with open(FILE_DIRNAME / "test_data.json") as f:
        test_data = json.load(f)
    formulas = []
    for entry in test_data:
        formula = entry["formula"].split()
        formulas.append(formula)
    return formulas


@pytest.fixture
def tokenizer(formulas):
    tokenizer = Tokenizer()
    tokenizer.build(formulas)
    return tokenizer


@pytest.fixture
def train_dataloader(images, formulas, tokenizer):
    max_seq_len = find_max_length(formulas) + 2
    targets = tokenizer.index(
        formulas, add_sos=True, add_eos=True, pad_to=max_seq_len
    )
    train_dataset = BaseDataset(
        images, torch.LongTensor(targets), transform=transforms.ToTensor()
    )
    return torch.utils.data.DataLoader(train_dataset, batch_size=5)
