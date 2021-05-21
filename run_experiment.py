import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from torchinfo import summary

from image_to_latex.trainers import BaseTrainer
from image_to_latex.utils.misc import import_class


DATA_CLASS = "Im2Latex"
MODEL_CLASS = "ResnetTransformer"
ARTIFACTS_DIR = Path(__file__).resolve() / "artifacts"

np.random.seed(2021)
torch.manual_seed(2021)


def _setup_parser() -> ArgumentParser:
    """Add arguments for parser."""
    parser = ArgumentParser(add_help=False)

    parser.add_argument("--data_class", type=str, default=DATA_CLASS)
    parser.add_argument("--model_class", type=str, default=MODEL_CLASS)
    parser.add_argument("--load_config", type=str, default=None)
    parser.add_argument("--load_checkpoint", type=str, default=None)

    temp_args, _ = parser.parse_known_args()

    data_class = import_class(f"image_to_latex.data.{temp_args.data_class}")
    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)  # type: ignore

    model_class = import_class(f"image_to_latex.models.{temp_args.model_class}")
    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)  # type: ignore

    trainer_group = parser.add_argument_group("Trainer Args")
    BaseTrainer.add_to_argparse(trainer_group)

    parser.add_argument("--help", "-h", action="help")
    return parser


def main() -> None:
    """Model training."""
    parser = _setup_parser()
    args = parser.parse_args()

    # Convert the parsed arguments into a dictionary
    config = vars(args) if args is not None else {}

    checkpoint_filename = config["load_checkpoint"]

    if config["load_config"] is not None:
        with open(config["load_config"], "r") as f:
            config = json.load(f)

    # Set up dataloaders
    data_class = import_class(f"image_to_latex.data.{config['data_class']}")
    data_module = data_class(config)
    data_module.prepare_data()
    data_module.build_vocab()
    data_module.setup()
    train_dataloader = data_module.get_dataloader("train")
    val_dataloader = data_module.get_dataloader("val")
    test_dataloader = data_module.get_dataloader("test")

    # Set up the model
    model_class = import_class(f"image_to_latex.models.{config['model_class']}")
    model = model_class(data_module.id2token, config)
    print("\nModel summary:")
    summary(model)

    trainer_class = import_class(f"image_to_latex.trainers.{config['model_class']}Trainer")
    trainer = trainer_class(model, config)
    trainer.fit(train_dataloader, val_dataloader, checkpoint_filename)
    trainer.test(test_dataloader)


if __name__ == "__main__":
    main()
