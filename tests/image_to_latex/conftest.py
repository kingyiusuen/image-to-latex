import pytest

from image_to_latex.data import SampleData


@pytest.fixture()
def sample_data():
    sample_data = SampleData(batch_size=3)
    sample_data.create_datasets()
    return sample_data


@pytest.fixture
def train_dataloader(sample_data):
    return sample_data.get_dataloader("train")
