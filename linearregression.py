import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

df = pd.read_csv('housing.data', delim_whitespace=True, header=None)

col_name = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX','RM','AGE','DIS','RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.columns = col_name

# 1.Choose a class of model
# 2. Choose Model hyperparameters
# 3. Arrange data into features matrix and target feature
# 4. Fit the Model
# 5. Apply the model to new data

#Arrange data into feature matrix
X=df['LSTAT'].values.reshape(-1,1)
#Arrange data into target feature
y=df['MEDV'].values
#Choose a class of model
model = LinearRegression()
#Fit the model
model.fit(X,y)

sns.jointplot(x="LSTAT",y="MEDV", data=df, kind='reg',size=10)
plt.show()