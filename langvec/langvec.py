import os
import tempfile
import zipfile
from typing import List, Tuple

import numpy as np

from .constants import LEXICON, MAX_SAMPLES


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

    def _validate_input(self, X: List[np.ndarray]):
        if not isinstance(X, list):
            raise TypeError("❌ Input data must be provided as a list of numpy arrays.")

        if not all(isinstance(x, np.ndarray) for x in X):
            raise TypeError("❌ Each element of the input list must be a numpy array.")

        if any(x.ndim != 1 for x in X):
            raise ValueError(
                "❌ Each numpy array in the input list must be 1-dimensional."
            )

        return X

    def fit(self, X: List[np.ndarray], max_samples: int = MAX_SAMPLES) -> None:
        self._validate_input(X)

        try:
            elements = np.concatenate(X)
            if elements.shape[0] > max_samples:
                # Perform random sampling if the number of elements exceeds max_samples
                sample_indices = np.random.choice(
                    elements.shape[0], size=max_samples, replace=False
                )
                elements = elements[sample_indices]
            self._calculate_percentiles(elements)
        except ValueError as e:
            raise ValueError(f"❌ Error in fitting data: {e}")

    def predict(
            self, input_vector: np.ndarray, chunk_size: int = 3, summarized: bool = False, padding: bool = True
    ) -> List[str]:
        if not isinstance(self.percentiles, np.ndarray) or len(self.percentiles) == 0:
            raise ValueError(
                "❌ Model not fitted or loaded. Call 'fit' or 'load' with appropriate data before prediction."
            )

        if not isinstance(input_vector, np.ndarray) or input_vector.ndim != 1:
            raise ValueError("❌ Input vector must be a 1-dimensional numpy array.")

        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("❌ Chunk size must be a positive integer.")

        if self.lexicon_size - 1 != len(self.percentiles):
            raise ValueError(
                "❌ Lexicon size does not match learned distribution. Maybe you used different lexicon on training?")

        words_for_vector = []
        for i in range(0, len(input_vector), chunk_size):
            chunk = input_vector[i: i + chunk_size]
            if len(chunk) < chunk_size:
                if padding:
                    # Pad the last chunk with zeros if it has fewer elements than chunk_size
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
                else:
                    # Trim the last chunk if padding is not desired
                    continue
            word_index = sum(np.mean(chunk) > self.percentiles)
            words_for_vector.append(self.lexicon[word_index])

        if summarized and len(words_for_vector) > 3:
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

    def update(self, X: List[np.ndarray], max_samples: int = MAX_SAMPLES) -> None:
        self._validate_input(X)

        try:
            elements = np.concatenate(X)
            if elements.shape[0] > max_samples:
                # Perform random sampling if the number of elements exceeds max_samples
                sample_indices = np.random.choice(
                    elements.shape[0], size=max_samples, replace=False
                )
                elements = elements[sample_indices]

            # Calculate percentiles for the new data
            new_percentiles = np.percentile(elements, self._lexicon_distribution)

            # Combine the new percentiles with the existing percentiles
            if self.percentiles is not None:
                combined_percentiles = np.concatenate(
                    (self.percentiles, new_percentiles)
                )
            else:
                combined_percentiles = new_percentiles

            # Recalculate the percentiles using the combined percentiles
            self.percentiles = np.percentile(
                combined_percentiles, self._lexicon_distribution
            )

        except ValueError as e:
            raise ValueError(f"❌ Error in updating data: {e}")

    def info(self) -> None:
        print("📦LangVec model info:")
        print(f"✔ Lexicon size: {self.lexicon_size}")
        print(f"✔ Chunk size: {self.chunk_size}")
        percentiles_count = len(self.percentiles) if self.percentiles is not None else 0
        model_fitted = "yes" if self.percentiles is not None else "no"
        print(f"✔ Percentiles count: {percentiles_count}")
        print(f"✔ Model fitted: {model_fitted}")
