import os
import tempfile
import zipfile
from typing import List, Tuple

import numpy as np

from .constants import LEXICON


class LangVec:
    def __init__(self, lexicon: List[str] = LEXICON, chunk_size: int = 3):
        self.lexicon = lexicon
        self.lexicon_size = len(lexicon)
        self._lexicon_distribution = self._initialize_lexicon_distribution()
        self.chunk_size = chunk_size
        self.percentiles = None

    def _initialize_lexicon_distribution(self) -> Tuple[float, ...]:
        if self.lexicon_size <= 1:
            return (50,)  # Default to median if only one word
        return tuple(np.linspace(0, 100, self.lexicon_size + 1)[1:-1])

    def _calculate_percentiles(self, data: np.ndarray) -> None:
        self.percentiles = np.percentile(data, self._lexicon_distribution)

    def fit(self, X: List[np.ndarray]) -> None:
        if not isinstance(X, list):
            raise TypeError("❌ Input data must be provided as a list of numpy arrays.")

        if not all(isinstance(x, np.ndarray) for x in X):
            raise TypeError("❌ Each element of the input list must be a numpy array.")

        if any(x.ndim != 1 for x in X):
            raise ValueError(
                "❌ Each numpy array in the input list must be 1-dimensional."
            )

        try:
            elements = np.concatenate(X)
            self._calculate_percentiles(elements)
        except ValueError as e:
            raise ValueError(f"❌ Error in fitting data: {e}")

    def predict(
            self, input_vector: np.ndarray, chunk_size: int = 3, summarized: bool = False
    ) -> List[str]:
        if not isinstance(self.percentiles, np.ndarray) or len(self.percentiles) == 0:
            raise ValueError(
                "❌ Model not fitted. Call 'fit' with appropriate data before prediction."
            )

        if not isinstance(input_vector, np.ndarray) or input_vector.ndim != 1:
            raise ValueError("❌ Input vector must be a 1-dimensional numpy array.")

        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("❌ Chunk size must be a positive integer.")

        words_for_vector = [
            self.lexicon[
                sum(np.mean(input_vector[i: i + chunk_size]) > self.percentiles)
            ]
            for i in range(0, len(input_vector), chunk_size)
        ]

        if summarized:
            return words_for_vector[:3] + ["....."] + words_for_vector[-3:]
        return words_for_vector

    def save(self, filepath: str) -> None:
        if self.percentiles is None:
            raise ValueError("❌ Model not fitted. Call 'fit' before saving.")

        # Create a temporary directory to store the model artifacts
        with tempfile.TemporaryDirectory() as tmp_dir:
            percentiles_filepath = os.path.join(tmp_dir, "percentiles.npy")
            np.save(percentiles_filepath, self.percentiles)

            # Create a zip file and add the model artifacts
            with zipfile.ZipFile(filepath, "w") as zipf:
                zipf.write(percentiles_filepath, arcname="percentiles.npy")

    def load(self, filepath: str) -> None:
        with zipfile.ZipFile(filepath, "r") as zipf:
            # Extract the model artifacts to a temporary directory
            with tempfile.TemporaryDirectory() as tmp_dir:
                zipf.extractall(tmp_dir)
                percentiles_filepath = os.path.join(tmp_dir, "percentiles.npy")

                # Load the percentiles from the extracted file
                self.percentiles = np.load(percentiles_filepath)

    def info(self) -> None:
        print("📦LangVec model info:")
        print(f"✔ Lexicon size: {self.lexicon_size}")
        print(f"✔ Chunk size: {self.chunk_size}")
        percentiles_count = len(self.percentiles) if self.percentiles is not None else 0
        model_fitted = "yes" if self.percentiles is not None else "no"
        print(f"✔ Percentiles count: {percentiles_count}")
        print(f"✔ Model fitted: {model_fitted}")
