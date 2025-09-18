import numpy as np
import pandas as pd

df = pd.DataFrame([[4, 9]] * 3, columns=['A', 'B'])
df.apply(np.sum, axis=1)