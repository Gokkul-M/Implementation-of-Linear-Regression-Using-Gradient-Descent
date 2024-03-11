# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Intialize weights randomly. 
2. Compute predicted.
3. Compute gradient of loss function. 
4. Update weights using gradient descent.


## Program:
```
Program to implement the linear regression using gradient descent.
Developed by: Gokkul M
RegisterNumber:212223240039
**import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        errors=(predictions-y).reshape(-1,1)
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data =pd.read_csv('50_Startups.csv',header=None)
print(data.head())
X=(data.iloc[1:, :-2].values)
print(X)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_Scaled =scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)
theta=linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value:{pre}")**
```

## Output:
![image](https://github.com/Gokkul-M/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144870543/42c1fe32-4cda-4cc3-9ff3-a575050fc8bd)
![image](https://github.com/Gokkul-M/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144870543/fdf13a10-f3b1-4750-9a6c-093f7a49b742)
![image](https://github.com/Gokkul-M/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144870543/fd030c46-6a2f-4305-8d39-b4edc3f066e8)
![image](https://github.com/Gokkul-M/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144870543/ec407468-8b10-43a2-a34a-92acb80bc067)
![image](https://github.com/Gokkul-M/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144870543/032b49a4-7666-448a-ac97-4f6c85083bfd)
![image](https://github.com/Gokkul-M/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144870543/85eb9052-7a04-4113-83eb-5d0299aea768)
![image](https://github.com/Gokkul-M/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144870543/ece933db-27dc-4329-bdaa-b96c303ea8bd)
![image](https://github.com/Gokkul-M/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144870543/c9678e45-c7da-431a-ade5-6a0b5b6bfc3e)
![image](https://github.com/Gokkul-M/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144870543/05e2a581-da82-42ae-9b97-cd5eb94e238b)
![image](https://github.com/Gokkul-M/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144870543/3749a682-abd7-4210-832d-c4caedb483d0)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
