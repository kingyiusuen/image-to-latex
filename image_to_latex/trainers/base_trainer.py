import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from image_to_latex.models.base_model import BaseModel
from image_to_latex.utils.data import convert_labels_to_strings
from image_to_latex.utils.lr_finder import LRFinder_
from image_to_latex.utils.metrics import bleu_score, edit_distance
from image_to_latex.utils.misc import compute_time_elapsed


MAX_EPOCHS = 100
PATIENCE = 10
LR = 1e-4
MAX_LR = None
SAVE_BEST_MODEL = False

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"
BEST_MODEL_FILENAME = ARTIFACTS_DIR / "best.pth"
LAST_MODEL_FILENAME = ARTIFACTS_DIR / "last.pth"


class BaseTrainer:
    """Specify every aspect of training.

    Attributes:
        max_epochs: Maximum number of epochs to run.
        patience: Number of epochs with no improvement before stopping the
            training. Use -1 to disable early stopping.
        lr: Learning rate.
        max_lr: Maximum learning rate to use in OneCycleLR scheduler.
        checkpoint_dir: Directory to save the checkpoint file.
        save_best_model: Save a checkpoint when the current model has the best
            validation loss so far.
        save_last: Save a checkpoint named "last.pt" after every epoch.
        start_epoch: The first epoch number.
        best_val_loss: Best validation loss encountered so far.
        no_improve_count: Number of epochs since the last improvement in
            validation loss.
        criterion: Loss function.
        device: Which device to put the model and data in.
        scheduler: Learning rate scheduler.
    """

    def __init__(self, model: BaseModel, config: Dict[str, Any] = None) -> None:
        self.model = model
        self.config = config if config is not None else {}

        self.max_epochs = self.config.get("max_epochs", MAX_EPOCHS)
        self.patience = self.config.get("patience", PATIENCE)
        self.lr = self.config.get("lr", LR)
        self.max_lr = self.config.get("max_lr", MAX_LR)
        self.save_best_model = self.config.get("save_best_model", SAVE_BEST_MODEL)

        self.start_epoch = 1
        self.best_val_loss = float("inf")
        self.no_improve_count = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

        self.criterion: Union[nn.CrossEntropyLoss, nn.CTCLoss]
        self.optimizer: optim.Optimizer
        self.scheduler: optim.lr_scheduler._LRScheduler

    @staticmethod
    def add_to_argparse(parser):
        """Add arguments to a parser."""
        parser.add_argument("--max_epochs", type=int, default=MAX_EPOCHS)
        parser.add_argument("--patience", type=int, default=PATIENCE)
        parser.add_argument("--lr", type=float, default=LR)
        parser.add_argument("--max_lr", type=float, default=MAX_LR)
        parser.add_argument("--save_best_model", action="store_true")
        return parser

    def fit(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        checkpoint_filename: Optional[str] = None,
    ) -> None:
        """Specify what happens during training."""
        # Configure optimizier
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-6)

        # Load checkpoint
        if checkpoint_filename is not None:
            self._resume_from_checkpoint(checkpoint_filename)

        # Find learning rate
        if self.config["max_lr"] is None:
            print("Running learning rate range test...")
            self.config["max_lr"] = self._find_optimal_lr(train_dataloader)
            self.max_lr = self.config["max_lr"]

        # Save command line arguments
        if self.config["load_config"] is None:
            self._save_config()

        # Configure scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(  # type: ignore
            self.optimizer,
            max_lr=self.max_lr,
            epochs=self.max_epochs,
            steps_per_epoch=len(train_dataloader),
            pct_start=0.5,
            div_factor=10,
            final_div_factor=1e4,
        )

        # For display purpose
        width = len(str(self.max_epochs))

        data_loaders = {"train": train_dataloader, "val": val_dataloader}
        avg_loss = {"train": 0.0, "val": 0.0}
        self.model.to(self.device)

        for epoch in range(self.start_epoch, self.max_epochs + 1):
            start_time = time.time()
            for phase in ["train", "val"]:
                total_loss = 0.0
                if phase == "train":
                    self.model.train()
                else:
                    self.model.eval()
                pbar = tqdm(data_loaders[phase], desc=phase, leave=False)
                for batch in pbar:
                    batch = self._move_to_device(batch)
                    if phase == "train":
                        loss = self.training_step(batch)
                        loss.backward()
                        self.optimizer.step()
                        self.scheduler.step()
                    else:
                        loss = self.validation_step(batch)
                    total_loss += loss.item()
                    pbar.set_postfix({f"{phase}_loss": loss.item()})
                avg_loss[phase] = total_loss / len(data_loaders[phase])
            end_time = time.time()

            # Print model performance
            epoch_mins, epoch_secs = compute_time_elapsed(start_time, end_time)
            print(
                f"Epoch {epoch:{width}d}/{self.max_epochs} | "
                f"Train loss: {avg_loss['train']:.3f} | "
                f"Val loss: {avg_loss['val']:.3f} | "
                f"Time: {epoch_mins}m {epoch_secs}s"
            )

            # Early stopping and save checkpoint
            if self._early_stopping(epoch, avg_loss["val"]):
                print(
                    f"Training is terminated because validation loss has "
                    f"stopped decreasing for {self.patience} epochs."
                )
                return

        print("Training completed.")

    def training_step(self, batch: Sequence):
        """Training step."""
        imgs, targets = batch
        logits = self.model(imgs, targets)
        loss = self.criterion(logits, targets)
        return loss

    @torch.no_grad()
    def validation_step(self, batch: Sequence):
        """Validation step."""
        imgs, targets = batch
        logits = self.model(imgs, targets)
        loss = self.criterion(logits, targets)
        return loss

    @torch.no_grad()
    def test(self, test_dataloader: DataLoader) -> None:
        """Specify what happens during testing."""
        references = []
        hypothesis = []

        self.model.to(self.device)
        self.model.eval()

        pbar = tqdm(test_dataloader, desc="Testing: ", leave=False)
        for batch in pbar:
            batch = self._move_to_device(batch)
            imgs, targets = batch
            preds = self.model.predict(imgs)
            references += convert_labels_to_strings(targets, self.model.id2token)
            hypothesis += convert_labels_to_strings(preds, self.model.id2token)

        bleu = bleu_score(references, hypothesis) * 100
        ed = edit_distance(references, hypothesis) * 100
        print(
            "--------------------\n"
            "Evaluation Results\n"
            "--------------------\n"
            f"BLEU: {bleu:.3f}\n"
            f"Edit Distance: {ed:.3f}\n"
        )

    def _move_to_device(self, batch: Sequence) -> List[Any]:
        """Move tensors to device."""
        return [x.to(self.device) for x in batch if isinstance(x, torch.Tensor)]

    def _early_stopping(self, epoch: int, current_val_loss: float) -> bool:
        """Returns whether the training should stop."""
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.no_improve_count = 0
            self._save_checkpoint(epoch, is_best=True)
        else:
            self.no_improve_count += 1
            if self.no_improve_count == self.patience:
                return True
            self._save_checkpoint(epoch, is_best=False)
        return False

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save a checkpoint to be used for inference or resume training."""
        if not self.save_best_model:
            return
        checkpoint = {
            "start_epoch": epoch + 1,
            "best_val_loss": self.best_val_loss,
            "no_improve_count": self.no_improve_count,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }
        if is_best and self.save_best_model:
            torch.save(checkpoint, BEST_MODEL_FILENAME)

    def _resume_from_checkpoint(
        self, checkpoint_filename: str, load_model_only: bool = False
    ) -> None:
        """Restore states from a checkpoint."""
        checkpoint = torch.load(checkpoint_filename)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if load_model_only:
            return
        self.start_epoch = checkpoint["start_epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.no_improve_count = checkpoint["no_improve_count"]
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print("Resume training from checkpoint...")

    def _save_config(self) -> None:
        """Store the inputs from command line to a json file."""
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(ARTIFACTS_DIR / "config.json", "w") as f:
            json.dump(self.config, f)

    def _find_optimal_lr(self, train_dataloader: DataLoader) -> float:
        """Returns suggested learning rate."""
        lr_finder = LRFinder_(self.model, self.optimizer, self.criterion)
        lr_finder.range_test(train_dataloader, end_lr=100, num_iter=100)
        max_lr = lr_finder.suggest_lr()
        return max_lr
