import os
import time
from typing import List

import numpy as np

from langvec import LangVec

os.environ["OMP_NUM_THREADS"] = "12"


def generate_dummy_data(num_vectors: int, dimensions: int) -> List[np.ndarray]:
    """
    Generate dummy data as a list of NumPy arrays.

    Args:
        num_vectors (int): The number of vectors to generate.
        dimensions (int): The number of dimensions for each vector.

    Returns:
        List[np.ndarray]: A list of NumPy arrays representing the dummy data.
    """
    return [np.random.rand(dimensions) for _ in range(num_vectors)]


def benchmark_fitting(data: List[np.ndarray]) -> float:
    """
    Benchmark the fitting process of the LangVec library.

    Args:
        data (List[np.ndarray]): The input data as a list of NumPy arrays.

    Returns:
        float: The elapsed time (in seconds) for the fitting process.
    """
    lv = LangVec()
    start_time = time.time()
    lv.fit(data)
    end_time = time.time()
    return end_time - start_time


def benchmark_prediction(
    lv: LangVec, num_predictions: int, vector_dimensions: int
) -> float:
    """
    Benchmark the prediction process of the LangVec library.

    Args:
        lv (LangVec): The LangVec instance.
        num_predictions (int): The number of predictions to perform.
        vector_dimensions (int): The number of dimensions for each vector.

    Returns:
        float: The average time (in seconds) for the prediction process.
    """
    dummy_vectors = generate_dummy_data(num_predictions, vector_dimensions)
    prediction_times = []

    for vector in dummy_vectors:
        start_time = time.time()
        lv.predict(vector)
        end_time = time.time()
        prediction_times.append(end_time - start_time)

    return sum(prediction_times) / len(prediction_times)


if __name__ == "__main__":
    dimensions = 256
    vector_counts = [10**i for i in range(3, 7)]

    num_predictions = 10000

    fitting_times = []
    prediction_times = []

    for num_vectors in vector_counts:
        print(f"Benchmarking {num_vectors} x {dimensions} dimensional vectors...")
        dummy_data = generate_dummy_data(num_vectors, dimensions)

        # Benchmark fitting
        elapsed_time = benchmark_fitting(dummy_data)
        fitting_times.append(elapsed_time)
        print(f"Fitting time: {elapsed_time:.4f} seconds")

        # Initialize and fit the LangVec instance
        lv = LangVec()
        lv.fit(dummy_data)

        # Benchmark prediction
        average_prediction_time = benchmark_prediction(lv, num_predictions, dimensions)
        prediction_times.append(average_prediction_time)
        print(f"Average prediction time: {average_prediction_time:.6f} seconds")
        print("-" * 30)

    # Print the results in a tabular format
    print(
        "{:<20} {:<20} {:<20}".format(
            "Num Vectors", "Fitting Time (s)", "Prediction Time (s)"
        )
    )
    print("-" * 60)
    for i, num_vectors in enumerate(vector_counts):
        print(
            "{:<20} {:<20.4f} {:<20.6f}".format(
                num_vectors, fitting_times[i], prediction_times[i]
            )
        )
