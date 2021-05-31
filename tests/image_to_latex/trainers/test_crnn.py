import pytest

from image_to_latex.models import CRNN
from image_to_latex.trainers import CRNNTrainer


@pytest.mark.training
def test_overfit_one_batch_crnn(sample_data, train_dataloader):
    config = {
        "conv_dim": 32,
        "rnn_type": "RNN",
        "rnn_dim": 512,
        "rnn_layers": 2,
        "rnn_dropout": 0.3,
    }
    model = CRNN(sample_data.tokenizer, config)
    trainer = CRNNTrainer(
        model,
        max_epochs=100,
        patience=30,
        monitor="train_loss",
        use_scheduler=False,
        save_best_model=False,
    )
    performance = trainer.fit(train_dataloader)
    assert performance["best_monitor_val"] == pytest.approx(0.0, abs=0.05)
