import numpy as np
from lov import LoV

VECTORS_DIMENSIONS = 10

v_1 = np.random.uniform(0, 1, VECTORS_DIMENSIONS)
v_2 = np.random.uniform(0, 1, VECTORS_DIMENSIONS)
v_3 = np.random.uniform(0, 1, VECTORS_DIMENSIONS)
v_4 = np.random.uniform(0, 1, VECTORS_DIMENSIONS)
v_5 = np.random.uniform(0, 1, VECTORS_DIMENSIONS)

d = (v_1, v_2, v_3, v_4, v_5)

lov = LoV()
lov.fit(d)

d = lov.predict(v_5, summarized=True)
print(d)
d = lov.predict(v_1, summarized=True)
print(d)