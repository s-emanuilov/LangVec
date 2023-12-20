import numpy as np

from constants import LEXICON


class LoV:
    def __init__(self, lexicon=LEXICON, chunk_size=3):
        self.lexicon = lexicon
        self.lexicon_size = len(lexicon)
        self.lexicon_distribution = None
        self.chunk_size = chunk_size
        self.percentiles = None
        self.get_lexicon_distribution()

    def get_lexicon_distribution(self):
        if self.lexicon_size <= 1:
            self.lexicon_distribution = (50,)  # Default to median if only two words
        self.lexicon_distribution = tuple(
            np.linspace(0, 100, self.lexicon_size + 1)[1:-1]
        )

    def calculate_percentiles(self, data):
        self.percentiles = np.percentile(data, self.lexicon_distribution)

    def fit(self, X):
        elements = np.concatenate(X)
        self.calculate_percentiles(elements)

    def predict(self, input_vector, chunk_size=3, summarized=False):
        words_for_vector = []

        for i in range(0, len(input_vector), chunk_size):
            chunk = input_vector[i : i + chunk_size]
            avg_value = np.mean(chunk)
            index = sum(avg_value > self.percentiles)
            words_for_vector.append(self.lexicon[index])
        if summarized:
            return (
                words_for_vector[:3]
                + ["....."]
                + words_for_vector[len(words_for_vector) - 3 :]
            )
        return words_for_vector
