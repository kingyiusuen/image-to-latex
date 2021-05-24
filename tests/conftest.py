import numpy as np
import pytest


@pytest.fixture
def random():
    np.random.seed(0)
