import argparse
import shutil
import tempfile
from pathlib import Path

import wandb


def download_checkpoint(run_path: str) -> None:
    """Download model checkpoint from Weights & Biases.

    Args:
        run_path: The run path for a run, in the format of
            '<entity>/<project>/<run_id>'. To find the run path for a run, go
            to the Overview tab in wandb dashboard.
    """
    artifacts_dirname = Path(__file__).parents[1].resolve() / "artifacts"
    artifacts_dirname.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    wandb_run = api.run(f"{run_path}")

    checkpoint_wandb_files = [file for file in wandb_run.files() if file.name.endswith("ckpt")]

    if not checkpoint_wandb_files:
        print("Model checkpoint not found.")
        return

    wandb_file = checkpoint_wandb_files[0]
    with tempfile.TemporaryDirectory() as tmp_dirname:
        print("Downloading model checkpoint...")
        wandb_file.download(root=tmp_dirname, replace=True)
        checkpoint_filename = f"{tmp_dirname}/{wandb_file.name}"
        shutil.copyfile(src=checkpoint_filename, dst=artifacts_dirname / "model.pt")
    print(f"Model checkpoint downloaded to {str(artifacts_dirname / 'model.pt')}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_path", type=str)
    args = parser.parse_args()
    download_checkpoint(args.run_path)


if __name__ == "__main__":
    main()
