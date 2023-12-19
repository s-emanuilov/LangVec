import numpy as np

from constants import ALPHABET


class LoV:
    def __init__(self, alphabet=ALPHABET):
        self.alphabet = alphabet
        self.alphabet_size = len(alphabet)
        self.alphabet_distribution = None
        self.chunk_size = 3
        self.dimension = 10
        self.percentiles = None
        self.get_alphabet_distribution()

    def get_alphabet_distribution(self):
        if self.alphabet_size <= 1:
            self.alphabet_distribution = (50,)  # Default to median if only two words
        self.alphabet_distribution = tuple(
            np.linspace(0, 100, self.alphabet_size + 1)[1:-1]
        )

    def calculate_percentiles(self, data):
        self.percentiles = np.percentile(data, self.alphabet_distribution)

    def fit(self, X):
        elements = np.concatenate(X)
        self.calculate_percentiles(elements)

    def predict(self, input_vector, chunk_size=3, summarized=False):
        words_for_vector = []

        for i in range(0, len(input_vector), chunk_size):
            chunk = input_vector[i: i + chunk_size]
            avg_value = np.mean(chunk)
            index = sum(avg_value > self.percentiles)
            words_for_vector.append(self.alphabet[index])
        if summarized:
            return (
                    words_for_vector[:3]
                    + ["....."]
                    + words_for_vector[len(words_for_vector) - 3:]
            )
        return words_for_vector
