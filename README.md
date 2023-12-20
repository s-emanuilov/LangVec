# LoV

Language of Vectors (LoV) is a simple Python library designed for transforming numerical vector data into a language-like structure using a predefined set of words (lexicon). 
This approach leverages the concept of percentile-based mapping to assign words from a lexicon to numerical values, facilitating intuitive and human-readable representations of numerical data.

## Where to use LoV
The LoV library finds its prime application is in semantic search and similarity based systems, where understanding the proximity between vectors is crucial.  
By transforming complex numerical vectors into a lexicon-based representation, LoV facilitates an intuitive understanding of these similarities for humans.  
This transformation is particularly advantageous in scenarios where interpreting raw numerical data (like floating points or integers) can be challenging or less informative. 
In fields like machine learning and natural language processing, LoV can assist in tasks such as clustering or categorizing data, where a human-readable format is preferable for quick insights and decision-making.  

## Installation
TODO

## Usage
```python
import numpy as np
from LoV import LoV

# Initialize LoV
lov = LoV()

# Sample data for fitting
data = np.random.rand(100)
lov.fit([data])

# Example vector for prediction
input_vector = np.random.rand(15)
print(lov.predict(input_vector))
```