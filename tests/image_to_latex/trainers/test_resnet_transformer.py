import pytest

from image_to_latex.models.resnet_transformer import ResnetTransformer
from image_to_latex.trainers.resnet_transformer import ResnetTransformerTrainer


@pytest.mark.training
def test_overfit_one_batch_resnet_transformer(tokenizer, train_dataloader):
    config = {
        "resnet_layers": 2,
        "tf_dim": 128,
        "tf_fc_dim": 256,
        "tf_nhead": 4,
        "tf_dropout": 0.4,
        "tf_layers": 4,
        "max_output_len": 250,
    }
    model = ResnetTransformer(tokenizer, config)
    trainer = ResnetTransformerTrainer(
        model,
        max_epochs=300,
        patience=30,
        monitor="train_loss",
        use_scheduler=False,
        save_best_model=False,
    )
    performance = trainer.fit(train_dataloader)
    assert performance["best_monitor_val"] == pytest.approx(0.0, abs=0.05)
