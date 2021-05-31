import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from image_to_latex.models.base_model import BaseModel
from image_to_latex.trainers.lr_finder import LRFinder_
from image_to_latex.utils.metrics import bleu_score, edit_distance
from image_to_latex.utils.misc import compute_time_elapsed


TRAINING_LOGS_DIRNAME = Path(__file__).resolve().parents[2] / "logs"


class BaseTrainer:
    """Specify every aspect of training.

    Args:
        model: The model to be fitted.
        max_epochs: Maximum number of epochs to run.
        patience: Number of epochs with no improvement before stopping the
            training. Use -1 to disable early stopping.
        monitor: Quantity to be monitored for early stopping.
        lr: Learning rate.
        max_lr: Maximum learning rate to use in one-cycle learning rate
            scheduler. Use -1 to to run learning rate range test. Ignored if
            `use_scheduler` is False.
        use_scheduler: Specifies whether to use learning rate scheduler or not.
        save_best_model: Specifies whether to save the model that has the best
            validation loss.
        wandb_run: An instance of a Weights & Biases run.
        config: Configurations passed from command line.

    Attributes:
        start_epoch: The first epoch number.
        best_val_loss: Best validation loss encountered so far.
        no_improve_count: Number of epochs since the last improvement in
            validation loss.
        device: Which device to put the model and data in.
        criterion: Loss function.
        optimizer: Optimization algorithm to use.
        scheduler: Learning rate scheduler.
    """

    def __init__(
        self,
        model: BaseModel,
        max_epochs: int = 100,
        patience: int = 10,
        monitor: str = "val_loss",
        lr: float = 0.001,
        max_lr: float = -1,
        use_scheduler: bool = True,
        save_best_model: bool = True,
        wandb_run: Optional[wandb.sdk.wandb_run.Run] = None,
    ) -> None:
        self.model = model
        self.max_epochs = max_epochs
        self.patience = patience
        self.monitor = monitor
        self.lr = lr
        self.max_lr = max_lr
        self.use_scheduler = use_scheduler
        self.save_best_model = save_best_model
        self.wandb_run = wandb_run

        self.tokenizer = self.model.tokenizer
        self.start_epoch = 1
        self.best_monitor_val = float("inf")
        self.no_improve_count = 0
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        TRAINING_LOGS_DIRNAME.mkdir(parents=True, exist_ok=True)

        self.criterion: Union[nn.CrossEntropyLoss, nn.CTCLoss]
        self.optimizer: optim.Optimizer
        self.scheduler: optim.lr_scheduler._LRScheduler

    def config(self) -> Dict[str, Any]:
        """Returns important configuration for reproducibility."""
        return {
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "lr": self.lr,
            "max_lr": self.max_lr,
            "use_scheduler": self.use_scheduler,
            "save_best_model": self.save_best_model,
        }

    def fit(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
    ) -> Dict:
        """Specify what happens during training."""
        self._configure_optimizer(train_dataloader)
        self.model.to(self.device)

        loss = {"train_loss": float("inf"), "val_loss": float("inf")}

        for epoch in range(self.start_epoch, self.max_epochs + 1):
            start_time = time.time()
            loss["train_loss"] = self.training_epoch(train_dataloader, epoch)
            loss["val_loss"] = self.validation_epoch(val_dataloader, epoch)
            end_time = time.time()
            mins, secs = compute_time_elapsed(start_time, end_time)

            # Print training progress
            width = len(str(self.max_epochs))
            print(f"Epoch {epoch:{width}d}/{self.max_epochs} |", end=" ")
            print(f"Train loss: {loss['train_loss']:.3f} |", end=" ")
            if val_dataloader is not None:
                print(f"Val loss: {loss['val_loss']:.3f} |", end=" ")
            print(f"Time: {mins}m {secs}s")

            # Check if the model stops improving
            # Save checkpoint if necessary
            if self._early_stopping(loss[self.monitor]):
                print(
                    f"Training is terminated because validation loss has "
                    f"stopped decreasing for {self.patience} epochs.\n"
                )
                break

        if self.wandb_run:
            wandb.run.summary["epoch"] = epoch  # type: ignore

        return {
            "epoch": epoch,
            "best_monitor_val": self.best_monitor_val,
        }

    def training_epoch(
        self, train_dataloader: DataLoader, epoch: int
    ) -> float:
        total_loss = 0.0
        self.model.train()
        pbar = tqdm(train_dataloader, desc="Training", leave=False)
        for batch in pbar:
            batch = self._move_to_device(batch)
            self.optimizer.zero_grad()
            loss = self.training_step(batch)
            loss.backward()
            self.optimizer.step()
            if self.use_scheduler:
                self.scheduler.step()
            total_loss += loss.item()
            pbar.set_postfix({"train_loss": loss.item()})
        avg_train_loss = total_loss / len(train_dataloader)
        if self.wandb_run:
            wandb.log({"train_loss": avg_train_loss, "epoch": epoch})
        return avg_train_loss

    @torch.no_grad()
    def validation_epoch(
        self, val_dataloader: Optional[DataLoader], epoch: int
    ) -> Union[float, None]:
        if not val_dataloader:
            return None
        total_loss = 0.0
        self.model.eval()
        pbar = tqdm(val_dataloader, desc="Validating", leave=False)
        for batch in pbar:
            batch = self._move_to_device(batch)
            loss = self.validation_step(batch)
            total_loss += loss.item()
            pbar.set_postfix({"val_loss": loss.item()})
        avg_val_loss = total_loss / len(val_dataloader)
        if self.wandb_run:
            wandb.log({"val_loss": avg_val_loss, "epoch": epoch})
        return avg_val_loss

    def training_step(self, batch: Sequence):
        """Training step."""
        imgs, targets = batch
        logits = self.model(imgs, targets)
        loss = self.criterion(logits, targets)
        return loss

    def validation_step(self, batch: Sequence):
        """Validation step."""
        imgs, targets = batch
        logits = self.model(imgs, targets)
        loss = self.criterion(logits, targets)
        return loss

    @torch.no_grad()
    def test(self, test_dataloader: DataLoader) -> None:
        """Specify what happens during testing."""
        if self.save_best_model:
            checkpoint = torch.load(TRAINING_LOGS_DIRNAME / "model.pth")
            self.model.load_state_dict(checkpoint)  # type: ignore

        references: List[List[str]] = []
        hypothesis: List[List[str]] = []

        self.model.to(self.device)
        self.model.eval()

        start_time = time.time()
        pbar = tqdm(test_dataloader, desc="Testing: ", leave=False)
        for batch in pbar:
            batch = self._move_to_device(batch)
            imgs, targets = batch
            preds = self.model.predict(imgs)
            references += self.tokenizer.unindex(
                targets.tolist(), inference=True
            )
            hypothesis += self.tokenizer.unindex(
                preds.tolist(), inference=True
            )
        bleu = bleu_score(references, hypothesis) * 100
        ed = edit_distance(references, hypothesis) * 100
        end_time = time.time()
        mins, secs = compute_time_elapsed(start_time, end_time)
        print(
            "Evaluation Results:\n"
            "====================\n"
            f"BLEU: {bleu:.3f}\n"
            f"Edit Distance: {ed:.3f}\n"
            "====================\n"
            f"Time: {mins}m {secs}s"
        )

        if self.wandb_run:
            wandb.run.summary["bleu"] = bleu  # type: ignore
            wandb.run.summary["edit_distance"] = ed  # type: ignore

    def _configure_optimizer(self, train_dataloader: DataLoader) -> None:
        """Configure optimizier and scheduler."""
        # Configure optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        if not self.use_scheduler:
            return

        # Run learning rate range test to find maximum learning rate
        if self.max_lr < 0:
            print("Running learning rate range test...")
            self.max_lr = self._find_optimal_lr(train_dataloader)

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

    def _move_to_device(self, batch: Sequence) -> List[Any]:
        """Move tensors to device."""
        return [
            x.to(self.device) if isinstance(x, torch.Tensor) else x
            for x in batch
        ]

    def _early_stopping(self, curr_monitor_val: float) -> bool:
        """Returns whether the training should stop."""
        if curr_monitor_val < self.best_monitor_val:
            self.best_monitor_val = curr_monitor_val
            self.no_improve_count = 0
            self._save_checkpoint()
        else:
            self.no_improve_count += 1
            if self.no_improve_count == self.patience:
                return True
        return False

    def _save_checkpoint(self) -> None:
        """Save a checkpoint to be used for inference."""
        if not self.save_best_model:
            return
        checkpoint = self.model.state_dict()
        torch.save(checkpoint, TRAINING_LOGS_DIRNAME / "model.pth")

    def _find_optimal_lr(self, dataloader: DataLoader) -> Optional[float]:
        """Returns suggested learning rate."""
        lr_finder = LRFinder_(
            self.model, self.optimizer, self.criterion, self.device
        )
        lr_finder.range_test(dataloader, end_lr=100, num_iter=100)
        max_lr = lr_finder.suggest_lr()
        lr_finder.reset()
        return max_lr
