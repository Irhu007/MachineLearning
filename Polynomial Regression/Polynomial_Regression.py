#Polynomial Regression
#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Datasets
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#Fitting Linear Regression to the Dataset
from sklearn.linear_model import LinearRegression
lin_Reg = LinearRegression()
lin_Reg.fit(X,y)

#Fitting Polynomial Regression to the Dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 6)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)

#Visualising the linear Regression result 
plt.scatter(X,y,color = 'red')
plt.plot(X,lin_Reg.predict(X),color='blue')
plt.title('Truth or Bluff(Linear)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

#Visualising the Polynomial Regression result
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid)),1)
plt.scatter(X,y,color = 'red')
plt.plot(X_grid,lin_reg_2.predict(poly_reg.fit_transform(X_grid)),color = 'blue')
plt.title('Truth or Bluff(Poly)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()