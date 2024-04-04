import numpy as np

from langvec import LangVec

# Set a seed for reproducibility
np.random.seed(42)

# Define the dimensionality of the vectors
NUM_VECTORS = 1000
VECTORS_DIMENSIONS = 768

# Generate random vectors using list comprehension
vectors = [np.random.uniform(0, 1, VECTORS_DIMENSIONS) for _ in range(NUM_VECTORS)]

input_vector = np.random.uniform(0, 1, VECTORS_DIMENSIONS)

# Initiate object
lv = LangVec()

# Fit to get with the distribution
lv.fit(vectors)
lv.save("model.lv")
lv.info()
# lv.load("model.lv")

# Predict with summary on some of the vectors (getting the lexicon representation)
print(lv.predict(input_vector, summarized=True))

# Change slightly some of the elements of given vector
input_vector_copy = input_vector
input_vector_copy[0] += 0.1
print(lv.predict(input_vector_copy, summarized=True))

# Predict without summary on some of the vectors (getting the lexicon representation)
print(lv.predict(input_vector, summarized=False))
