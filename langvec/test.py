import numpy as np

from langvec import LangVec

# Define the dimensionality of the vectors
VECTORS_DIMENSIONS = 50

# Generate some random vectors
v_1 = np.random.uniform(0, 1, VECTORS_DIMENSIONS)
v_2 = np.random.uniform(0, 1, VECTORS_DIMENSIONS)
v_3 = np.random.uniform(0, 1, VECTORS_DIMENSIONS)
v_4 = np.random.uniform(0, 1, VECTORS_DIMENSIONS)
v_5 = np.random.uniform(0, 1, 10)

# Make them as list
vectors = [v_1, v_2, v_3, v_4, v_5]

# Initiate object
lv = LangVec()

# Fit to get with the distribution
lv.fit(vectors)

# Predict with summary on some of the vectors (getting the lexicon representation)
print(lv.predict(v_5, summarized=True))

# Change slighly some of the elements of given vector
v_6 = v_5
v_6[0] += 0.1
print(lv.predict(v_6, summarized=True))

# Predict without summary on some of the vectors (getting the lexicon representation)
print(lv.predict(v_1, summarized=False))
