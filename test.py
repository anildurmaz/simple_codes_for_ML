
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

plt.style.use("seaborn")
plt.figure(figsize = (13,10))

########################################################################################
# reading data from with pandas
data = pd.read_csv("SomervilleHappinessSurvey2015.csv", encoding='utf-16')
corr = data.corr()
data.describe()
data.info()

plt.subplot(3,3,1)
plt.scatter(data["X1"], data["D"])
plt.xlabel("X1")
plt.ylabel("D")

x = data[["X1"]]
y = data[["D"]]
print(x)

train_set, test_set = train_test_split(data, test_size = 0.2 , random_state = 42)
train_x = train_set[["X1"]]
train_y = train_set[["D"]]
reg = LinearRegression()
reg.fit(train_x, train_y)

# coefficient
w = reg.coef_[0][0]
b = reg.intercept_[0]
print("slope : ", w)
print("intercept : ",b)

plt.title("linear Regrasyon")
plt.subplot(3,3,2)
plt.scatter(train_set.X1,train_set.D, color = "b")
plt.plot(train_x, w*train_x + b, '-r')
plt.xlabel("X1")
plt.ylabel("D")

# TEST DATA
test_x = test_set[["X1"]]
test_y = test_set[["D"]]
test_yhat = reg.predict(test_x)
print("mean absolute error : %.3f" % np.mean(np.absolute(test_yhat - test_y)))
print("residual sum of squares (MSE) : %.3f" % np.mean((test_yhat - test_y) ** 2))
print("R2-score : %.2f" % r2_score(test_yhat , test_y))

def D(x):
    D = x*w+b
    return D

print(D(3))

plt.subplot(3,3,3)
plt.hist(data["D"])
