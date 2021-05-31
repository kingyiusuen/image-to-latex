import pytest

from image_to_latex.models.crnn import CRNN
from image_to_latex.trainers.crnn import CRNNTrainer


@pytest.mark.training
def test_overfit_one_batch_crnn(tokenizer, train_dataloader):
    config = {
        "conv_dim": 32,
        "rnn_type": "RNN",
        "rnn_dim": 512,
        "rnn_layers": 2,
        "rnn_dropout": 0.3,
    }
    model = CRNN(tokenizer, config)
    trainer = CRNNTrainer(
        model,
        max_epochs=300,
        patience=30,
        monitor="train_loss",
        use_scheduler=False,
        save_best_model=False,
    )
    performance = trainer.fit(train_dataloader)
    assert performance["best_monitor_val"] == pytest.approx(0.0, abs=0.05)
