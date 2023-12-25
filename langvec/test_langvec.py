import unittest

import numpy as np

from .langvec import LangVec


class TestLangVec(unittest.TestCase):
    def setUp(self):
        self.lang_vec = LangVec()
        self.vectors = [np.random.uniform(0, 1, 50) for _ in range(4)]
        self.short_vector = np.random.uniform(0, 1, 10)

    def test_initialization(self):
        """Test initialization of LangVec class."""
        self.assertIsInstance(self.lang_vec, LangVec)

    def test_fit(self):
        """Test fitting the model with vectors."""
        try:
            self.lang_vec.fit(self.vectors)
        except Exception as e:
            self.fail(f"Fit method failed with exception {e}")

    def test_predict_after_fit(self):
        """Test prediction after fitting the model."""
        self.lang_vec.fit(self.vectors)
        try:
            prediction = self.lang_vec.predict(self.vectors[0])
            self.assertTrue(len(prediction) > 0)
        except Exception as e:
            self.fail(f"Predict method failed with exception {e}")

    def test_predict_before_fit(self):
        """Test prediction before fitting the model."""
        with self.assertRaises(ValueError):
            self.lang_vec.predict(self.vectors[0])

    def test_predict_summarized(self):
        """Test summarized prediction."""
        self.lang_vec.fit(self.vectors)
        prediction = self.lang_vec.predict(self.vectors[0], summarized=True)
        self.assertTrue(isinstance(prediction, list) and len(prediction) == 7)

    def test_predict_with_short_vector(self):
        """Test prediction with a shorter vector."""
        self.lang_vec.fit(self.vectors + [self.short_vector])
        try:
            prediction = self.lang_vec.predict(self.short_vector)
            self.assertTrue(len(prediction) > 0)
        except Exception as e:
            self.fail(f"Predict method failed with a short vector, exception: {e}")


if __name__ == "__main__":
    unittest.main()
