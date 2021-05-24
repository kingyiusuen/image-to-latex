import hashlib
import importlib
import tarfile
from pathlib import Path
from typing import Any, List, Tuple, Union
from urllib.request import urlretrieve

import torch
from tqdm import tqdm


def import_class(module_and_class_name: str) -> type:
    """Import class from a module."""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


class TqdmUpTo(tqdm):
    """From https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py."""

    def update_to(self, blocks=1, bsize=1, tsize=None) -> None:
        """Inform the progress bar how many data have been downloaded.

        Args:
            blocks: Number of blocks transferred so far.
            bsize: Size of each block (in tqdm units).
            tsize: Total size (in tqdm units).
        """
        if tsize is not None:
            self.total = tsize
        self.update(blocks * bsize - self.n)


def download_url(url: str, filename: Union[Path, str]) -> None:
    """Download a file from url to filename, with a progress bar."""
    with TqdmUpTo(
        unit="B", unit_scale=True, unit_divisor=1024, miniters=1
    ) as t:
        if isinstance(filename, Path):
            t.set_description(filename.name)
        else:
            t.set_description(filename)
        urlretrieve(url, filename, reporthook=t.update_to, data=None)


def extract_tar_file(filename: Union[Path, str]) -> None:
    """Extract a .tar or .tar.gz file."""
    if isinstance(filename, Path):
        print(f"Extracting {filename.name}...")
    else:
        print(f"Extracting {filename}...")
    with tarfile.open(filename, "r") as f:
        f.extractall()


def verify_sha256(filename: Union[Path, str], expected_sha256: str):
    """Verify the SHA-256 of a downloaded file."""
    with open(filename, "rb") as f:
        actual_sha256 = hashlib.sha256(f.read()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "Downloaded data file SHA-256 does not match that "
            "listed in metadata document."
        )


def compute_time_elapsed(
    start_time: Union[float, int],
    end_time: Union[float, int],
) -> Tuple[int, int]:
    """Compute time elapsed."""
    assert end_time >= start_time
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_time -= elapsed_mins * 60
    elapsed_secs = int(elapsed_time)
    return elapsed_mins, elapsed_secs


def find_max_length(lst: List[List[Any]]) -> int:
    """Find the length of the longest list in a list of lists."""
    return max(len(x) for x in lst)


def find_first(
    x: torch.Tensor,
    element: Union[int, float],
    dim: int = 1,
) -> torch.Tensor:
    """Return indices of first occurence of element in x.

    If not found, return length of x along dim.

    Reference:
        https://discuss.pytorch.org/t/first-nonzero-index/24769/10

    Usage:
        >>> find_first(torch.tensor([[1, 2, 3], [2, 3, 3], [1, 1, 1]]), 3)
        tensor([2, 1, 3])
    """
    nonz = x == element
    ind = ((nonz.cumsum(dim) == 1) & nonz).max(dim).indices
    ind[ind == 0] = x.shape[dim]
    return ind
