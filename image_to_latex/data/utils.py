import json
import tarfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
from urllib.request import urlretrieve

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


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


def download_url(url: str, filename: str) -> None:
    """Download a file from url to filename, with a progress bar."""
    with TqdmUpTo(unit="B", unit_scale=True, unit_divisor=1024, miniters=1) as t:
        t.set_description(filename)
        urlretrieve(url, filename, reporthook=t.update_to, data=None)


def extract_tar_file(filename: str) -> None:
    """Extract a .tar or .tar.gz file."""
    print(f"Extracting {filename}...")
    with tarfile.open(filename, "r") as f:
        
        import os
        
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(f)


class BaseDataset(Dataset):
    """A base Dataset class.

    Args:
        image_filenames: (N, *) feature vector.
        targets: (N, *) target vector relative to data.
        transform: Feature transformation.
        target_transform: Target transformation.
    """

    def __init__(
        self,
        root_dir: Path,
        image_filenames: List[str],
        formulas: List[List[str]],
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        assert len(image_filenames) == len(formulas)
        self.root_dir = root_dir
        self.image_filenames = image_filenames
        self.formulas = formulas
        self.transform = transform

    def __len__(self) -> int:
        """Returns the number of samples."""
        return len(self.formulas)

    def __getitem__(self, idx: int):
        """Returns a sample from the dataset at the given index."""
        image_filename, formula = self.image_filenames[idx], self.formulas[idx]
        image_filepath = self.root_dir / image_filename
        if image_filepath.is_file():
            image = pil_loader(image_filepath, mode="L")
        else:
            # Returns a blank image if cannot find the image
            image = Image.fromarray(np.full((64, 128), 255, dtype=np.uint8))
            formula = []
        if self.transform is not None:
            image = self.transform(image=np.array(image))["image"]
        return image, formula


def pil_loader(fp: Path, mode: str) -> Image.Image:
    with open(fp, "rb") as f:
        img = Image.open(f)
        return img.convert(mode)


class Tokenizer:
    def __init__(self, token_to_index: Optional[Dict[str, int]] = None) -> None:
        self.pad_token = "<PAD>"
        self.sos_token = "<SOS>"
        self.eos_token = "<EOS>"
        self.unk_token = "<UNK>"

        self.token_to_index: Dict[str, int]
        self.index_to_token: Dict[int, str]

        if token_to_index:
            self.token_to_index = token_to_index
            self.index_to_token = {index: token for token, index in self.token_to_index.items()}
            self.pad_index = self.token_to_index[self.pad_token]
            self.sos_index = self.token_to_index[self.sos_token]
            self.eos_index = self.token_to_index[self.eos_token]
            self.unk_index = self.token_to_index[self.unk_token]
        else:
            self.token_to_index = {}
            self.index_to_token = {}
            self.pad_index = self._add_token(self.pad_token)
            self.sos_index = self._add_token(self.sos_token)
            self.eos_index = self._add_token(self.eos_token)
            self.unk_index = self._add_token(self.unk_token)

        self.ignore_indices = {self.pad_index, self.sos_index, self.eos_index, self.unk_index}

    def _add_token(self, token: str) -> int:
        """Add one token to the vocabulary.

        Args:
            token: The token to be added.

        Returns:
            The index of the input token.
        """
        if token in self.token_to_index:
            return self.token_to_index[token]
        index = len(self)
        self.token_to_index[token] = index
        self.index_to_token[index] = token
        return index

    def __len__(self):
        return len(self.token_to_index)

    def train(self, formulas: List[List[str]], min_count: int = 2) -> None:
        """Create a mapping from tokens to indices and vice versa.

        Args:
            formulas: Lists of tokens.
            min_count: Tokens that appear fewer than `min_count` will not be
                included in the mapping.
        """
        # Count the frequency of each token
        counter: Dict[str, int] = {}
        for formula in formulas:
            for token in formula:
                counter[token] = counter.get(token, 0) + 1

        for token, count in counter.items():
            # Remove tokens that show up fewer than `min_count` times
            if count < min_count:
                continue
            index = len(self)
            self.index_to_token[index] = token
            self.token_to_index[token] = index

    def encode(self, formula: List[str]) -> List[int]:
        indices = [self.sos_index]
        for token in formula:
            index = self.token_to_index.get(token, self.unk_index)
            indices.append(index)
        indices.append(self.eos_index)
        return indices

    def decode(self, indices: List[int], inference: bool = True) -> List[str]:
        tokens = []
        for index in indices:
            if index not in self.index_to_token:
                raise RuntimeError(f"Found an unknown index {index}")
            if index == self.eos_index:
                break
            if inference and index in self.ignore_indices:
                continue
            token = self.index_to_token[index]
            tokens.append(token)
        return tokens

    def save(self, filename: Union[Path, str]):
        """Save token-to-index mapping to a json file."""
        with open(filename, "w") as f:
            json.dump(self.token_to_index, f)

    @classmethod
    def load(cls, filename: Union[Path, str]) -> "Tokenizer":
        """Create a `Tokenizer` from a mapping file outputted by `save`.

        Args:
            filename: Path to the file to read from.

        Returns:
            A `Tokenizer` object.
        """
        with open(filename) as f:
            token_to_index = json.load(f)
        return cls(token_to_index)


def get_all_formulas(filename: Path) -> List[List[str]]:
    """Returns all the formulas in the formula file."""
    with open(filename) as f:
        all_formulas = [formula.strip("\n").split() for formula in f.readlines()]
    return all_formulas


def get_split(
    all_formulas: List[List[str]],
    filename: Path,
) -> Tuple[List[str], List[List[str]]]:
    image_names = []
    formulas = []
    with open(filename) as f:
        for line in f:
            img_name, formula_idx = line.strip("\n").split()
            image_names.append(img_name)
            formulas.append(all_formulas[int(formula_idx)])
    return image_names, formulas


def first_and_last_nonzeros(arr):
    for i in range(len(arr)):
        if arr[i] != 0:
            break
    left = i
    for i in reversed(range(len(arr))):
        if arr[i] != 0:
            break
    right = i
    return left, right


def crop(filename: Path, padding: int = 8) -> Optional[Image.Image]:
    image = pil_loader(filename, mode="RGBA")

    # Replace the transparency layer with a white background
    new_image = Image.new("RGBA", image.size, "WHITE")
    new_image.paste(image, (0, 0), image)
    new_image = new_image.convert("L")

    # Invert the color to have a black background and white text
    arr = 255 - np.array(new_image)

    # Area that has text should have nonzero pixel values
    row_sums = np.sum(arr, axis=1)
    col_sums = np.sum(arr, axis=0)
    y_start, y_end = first_and_last_nonzeros(row_sums)
    x_start, x_end = first_and_last_nonzeros(col_sums)

    # Some images have no text
    if y_start >= y_end or x_start >= x_end:
        print(f"{filename.name} is ignored because it does not contain any text")
        return None

    # Cropping
    cropped = arr[y_start : y_end + 1, x_start : x_end + 1]
    H, W = cropped.shape

    # Add paddings
    new_arr = np.zeros((H + padding * 2, W + padding * 2))
    new_arr[padding : H + padding, padding : W + padding] = cropped

    # Invert the color back to have a white background and black text
    new_arr = 255 - new_arr
    return Image.fromarray(new_arr).convert("L")
