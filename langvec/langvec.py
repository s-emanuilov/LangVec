import numpy as np
from typing import List, Tuple

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
        try:
            elements = np.concatenate(X)
            self._calculate_percentiles(elements)
        except ValueError as e:
            raise ValueError(f"Error in fitting data: {e}")

    def predict(
        self, input_vector: np.ndarray, chunk_size: int = 3, summarized: bool = False
    ) -> List[str]:
        if not isinstance(self.percentiles, np.ndarray) or len(self.percentiles) == 0:
            raise ValueError(
                "Model not fitted. Call 'fit' with appropriate data before prediction."
            )

        words_for_vector = [
            self.lexicon[
                sum(np.mean(input_vector[i: i + chunk_size]) > self.percentiles)
            ]
            for i in range(0, len(input_vector), chunk_size)
        ]

        if summarized:
            return words_for_vector[:3] + ["....."] + words_for_vector[-3:]
        return words_for_vector
