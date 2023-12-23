<p align="center">
  <img src="assets/logo.png" alt="LangVec Logo" width="110">
</p>

<p align="center">
  <i>Language of Vectors (LangVec) is a simple Python library designed for transforming numerical vector data into a language-like structure using a predefined set of words (lexicon).</i>
</p>

## Approach

`LangVec` package leverages the concept of percentile-based mapping to assign words from a lexicon to numerical values, facilitating intuitive and human-readable representations of numerical data.

## Where to use LangVec
The `LangVec` library finds application in semantic search and similarity based systems, where understanding the proximity between vectors is crucial.  
By transforming complex numerical vectors into a lexicon-based representation, `LangVec` facilitates an intuitive understanding of these similarities for humans.  
This transformation is particularly advantageous in scenarios where interpreting raw numerical data (like floating points or integers) can be challenging or less informative. 
In fields like machine learning and natural language processing, `LangVec` can assist in tasks such as clustering or categorizing data, where a human-readable format is preferable for quick insights and decision-making.  

## Installation
```bash
pip install langvec
```

## Usage
```python
import numpy as np
import langvec

# Initialize LangVec
lv = LangVec()

# Generate some random data
data = np.random.rand(100)

# Fit to this data (getting know to distribution)
lv.fit([data])

# Example vector for prediction
input_vector = np.random.rand(15)

# Make prediction on unseen vector embedding
print(lv.predict(input_vector))
```