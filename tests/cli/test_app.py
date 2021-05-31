import pytest
from typer.testing import CliRunner

from cli.app import app


runner = CliRunner()


# This test will most likley get a warning that says 'The hypothesis contains
# 0 counts of 4-gram overlaps.' from the nlkt package, which is expected
# because the data are random.
@pytest.mark.training
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_train():
    command = (
        "train CRNN --dataset-name FakeData --no-use-wandb "
        "--no-save-best-model --no-use-scheduler --num-samples 256 "
        "--image-height 32 --image-width 128 --num-classes 10 --seq-len 10 "
        "--conv-dim 16 --max-epochs 2 --max-output-len 30"
    ).split()
    result = runner.invoke(app, command)
    assert result.exit_code == 0
