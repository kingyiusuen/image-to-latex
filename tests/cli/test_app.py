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
        "train --dataset-name SampleData --max-epochs 2 "
        "--no-use-wandb --no-save-best-model --no-use-scheduler "
        "--resnet-layers 3 --tf-dim 128 --tf-fc-dim 256 --tf-layers 3"
    ).split()
    result = runner.invoke(app, command)
    assert result.exit_code == 0
