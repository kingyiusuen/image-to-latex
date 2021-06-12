import pytest

from image_to_latex.models import ResnetTransformer
from image_to_latex.trainers import Trainer


@pytest.mark.training
def overfit_one_batch(sample_data, train_dataloader):
    config = {
        "resnet_layers": 2,
        "tf_dim": 128,
        "tf_fc_dim": 256,
        "tf_nhead": 4,
        "tf_dropout": 0.2,
        "tf_layers": 4,
        "max_output_len": 250,
    }
    model = ResnetTransformer(sample_data.tokenizer, config)
    trainer = Trainer(
        model,
        max_epochs=100,
        patience=30,
        monitor="train_loss",
        use_scheduler=False,
        save_best_model=False,
    )
    performance = trainer.fit(train_dataloader)
    assert performance["best_monitor_val"] == pytest.approx(0.0, abs=0.05)
