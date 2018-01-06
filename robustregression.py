import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import RANSACRegressor

df = pd.read_csv('housing.data', delim_whitespace=True, header=None)

col_name = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX','RM','AGE','DIS','RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.columns = col_name

X=df['RM'].values.reshape(-1,1)

y=df['MEDV'].values

ransac = RANSACRegressor()

ransac.fit(X,y)

inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

line_X = np.arange(3,10,1)
line_y_ransac = ransac.predict(line_X.reshape(-1,1))

sns.set(style="darkgrid", context ='notebook')
plt.figure(figsize=(12,10))
plt.scatter(X[inlier_mask], y[inlier_mask], c='blue', marker='o', label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask], c='brown', marker='s', label='Outliers')
plt.plot(line_X, line_y_ransac, color='red')
plt.legend(loc='upper left')
plt.show()
print ransac.estimator_.coef_