import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_boston

boston_data = load_boston()
df = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
X=df
y=boston_data.target
print df.head()