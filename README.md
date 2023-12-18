# LoV

Language of Vectors (LoV) is a simple library that helps to visually understand and compare high-dimensional vectors.

## Installation
TODO

## Usage
### Create a new LoV object
```python
from lov import LoV
lov = LoV()
```

### Fit to vectors distribution
```python
lov.fit(vectors)
```

### Show representation for a new vector
```python
representation = lov.predict(vector, summarized=True)
print(representation)
```