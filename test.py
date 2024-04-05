import string
import numpy as np
from langvec import LangVec

np.random.seed(42)

# Define a new lexicon with lowercase and uppercase letters
LEXICON = list(string.ascii_letters)

# Initialize LangVec with the new lexicon
lv = LangVec(lexicon=LEXICON)

NUM_VECTORS = 10000
DIMENSIONS = 256

# Generate some random data
vectors = [np.random.uniform(0, 1, DIMENSIONS) for _ in range(NUM_VECTORS)]

# Fit to this data (getting to know the distribution)
lv.fit(vectors)

# Example vector for prediction
input_vector = np.random.uniform(0, 1, DIMENSIONS)

# Make prediction on the unseen vector embedding
predicted_string = ''.join(lv.predict(input_vector))
print(predicted_string)
if len(predicted_string) > 6:
    summarized_string = ''.join(predicted_string[:3]) + '...' + ''.join(predicted_string[-3:])
else:
    summarized_string = ''.join(predicted_string)

print(summarized_string)