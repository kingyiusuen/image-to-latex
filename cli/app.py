import json
import shutil
import tempfile
from ast import literal_eval
from pathlib import Path
from typing import Any, Dict, Optional

import typer
from torchinfo import summary

import wandb
from image_to_latex.models import ResnetTransformer
from image_to_latex.trainers import Trainer
from image_to_latex.utils.misc import import_class


PROJECT_DIRNAME = Path(__file__).resolve().parents[1]
ARTIFACTS_DIRNAME = PROJECT_DIRNAME / "artifacts"
TRAINING_LOGS_DIRNAME = PROJECT_DIRNAME / "logs"


app = typer.Typer()


def _parse_args(ctx: typer.Context) -> Dict[str, Any]:
    unknown_args = ctx.args
    if (len(unknown_args) % 2) != 0:
        raise RuntimeError(
            "A name and a value must be provided for every additional "
            "argument."
        )
    args: Dict[str, Any] = {}
    for i in range(0, len(unknown_args), 2):
        name = unknown_args[i].split("--")[1].replace("-", "_")
        value = unknown_args[i + 1]
        args[name] = literal_eval(value)
    return args


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def train(
    dataset_name: str = typer.Option(
        "Im2Latex",
        help=(
            "{'Im2Latex', 'SampleData'}. Dataset to use. SampleData only has "
            "six samples and should only be used for debugging."
        ),
    ),
    batch_size: int = typer.Option(
        32, help="The number of samples per batch."
    ),
    num_workers: int = typer.Option(
        0, help="The number of subprocesses to use for data loading."
    ),
    max_epochs: int = typer.Option(
        100, help="Maximum number of epochs to run."
    ),
    patience: int = typer.Option(
        10,
        help=(
            "Number of epochs with no improvement before stopping the "
            "training. Use -1 to disable early stopping."
        ),
    ),
    monitor: str = typer.Option(
        "val_loss",
        help=(
            "{'train_loss', 'val_loss'}. Quantity to be monitored for early "
            "stopping."
        ),
    ),
    lr: float = typer.Option(0.001, help="Learning rate."),
    max_lr: float = typer.Option(
        -1,
        help=(
            "Maximum learning rate to use in one-cycle learning rate "
            "scheduler. Use -1 to to run learning rate range test. Ignored if "
            "`--no-use-scheduler` is used."
        ),
    ),
    use_scheduler: bool = typer.Option(
        False, help="Specifies whether to use a learning rate scheduler."
    ),
    save_best_model: bool = typer.Option(
        True,
        help=(
            "Specifies whether to save the model that has the best validation "
            "loss."
        ),
    ),
    use_wandb: bool = typer.Option(
        False,
        help=(
            "Specifies whether to use Weights & Biases for experiment "
            "tracking. Registration required."
        ),
    ),
    ctx: typer.Context = typer.Option(
        None,
        help=(
            "Additional arguments passed to configure the data module, "
            "model and trainer."
        ),
    ),
) -> None:
    """Train a model using the specified parameters.

    Usage:
        image-to-latex train --batch-size 64
    """
    assert dataset_name in ["Im2Latex", "SampleData"]

    args = _parse_args(ctx)

    # Set up dataloaders
    data_class = import_class(f"image_to_latex.data.{dataset_name}")
    data_module = data_class(batch_size, num_workers, args)
    data_module.prepare_data()
    data_module.create_datasets()
    train_dataloader = data_module.get_dataloader("train")
    val_dataloader = data_module.get_dataloader("val")
    test_dataloader = data_module.get_dataloader("test")

    # Set up the model
    model = ResnetTransformer(data_module.tokenizer, args)
    print("\nModel summary:")
    summary(model)

    # Set up Weights & Biases logger
    if use_wandb:
        wandb_run = wandb.init(project="image-to-latex")
    else:
        wandb_run = None

    # Start training
    trainer = Trainer(
        model=model,
        max_epochs=max_epochs,
        patience=patience,
        monitor=monitor,
        lr=lr,
        max_lr=max_lr,
        use_scheduler=use_scheduler,
        save_best_model=save_best_model,
        wandb_run=wandb_run,
    )
    trainer.fit(train_dataloader, val_dataloader)
    trainer.test(test_dataloader)

    # Upload artifacts
    if wandb_run:
        all_config = {}
        all_config.update(data_module.config())
        all_config.update(model.config())
        all_config.update(trainer.config())
        wandb.config.update(all_config)
        if save_best_model:
            wandb.save(str(Path(TRAINING_LOGS_DIRNAME, "best_model.pth")))
        with tempfile.TemporaryDirectory() as td:
            trainer.tokenizer.save(Path(td, "token_to_index.json"))
            wandb.save(str(Path(td, "token_to_index.json")))


@app.command()
def download_artifacts(
    run_path: str = typer.Argument(
        "", help="Run path in the format of '<entity>/<project>/<run_id>'."
    )
):
    """Download artifacts (configurations, run command and model checkpoint).

    To find the run path for a run, go to the Overview tab in wandb dashboard.
    """
    api = wandb.Api()
    wandb_run = api.run(f"{run_path}")
    # Download config
    ARTIFACTS_DIRNAME.mkdir(parents=True, exist_ok=True)
    config_filename = ARTIFACTS_DIRNAME / "config.json"
    with open(config_filename, "w") as file:
        json.dump(wandb_run.config, file, indent=4)
    print(f"Configuration file downloaded to {str(config_filename)}.")
    # Download model checkpoint
    _download_model_checkpoint(wandb_run, ARTIFACTS_DIRNAME)


def _download_model_checkpoint(
    wandb_run: wandb.apis.public.Run,
    output_dirname: Path,
) -> Optional[Path]:
    """Download model checkpoint to output_dirname.

    Reference:
    https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2021-labs/blob/main/lab9/training/save_best_model.py
    """
    checkpoint_wandb_files = [
        file for file in wandb_run.files() if file.name.endswith(".pth")
    ]
    if not checkpoint_wandb_files:
        print("Model checkpoint not found.")
        return None

    wandb_file = checkpoint_wandb_files[0]
    output_filename = output_dirname / "model.pth"
    with tempfile.TemporaryDirectory() as tmp_dirname:
        wandb_file.download(root=tmp_dirname, replace=True)
        checkpoint_filename = f"{tmp_dirname}/{wandb_file.name}"
        shutil.copyfile(
            src=checkpoint_filename,
            dst=output_filename,
        )
        print(f"Model checkpoint downloaded to {str(output_filename)}.")
    return output_filename


if __name__ == "__main__":
    app()
