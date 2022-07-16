import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv('SomervilleHappinessSurvey2015.csv',encoding=("utf-16"))
corr = data.corr()
#heatmap = sns.heatmap(corr)

distance = [t for t in range(0,len(data["X1"]),1)]
print(distance)
plt.subplot(2,3,1)
plt.hist(data["D"])
plt.subplot(2,3,2)
plt.hist(data["X1"])
plt.subplot(2,3,3)
plt.hist(data["X2"])
plt.subplot(2,3,4)
plt.hist(data["X3"])
plt.subplot(2,3,5)
plt.hist(data["X4"])



#plt.bar(distance,data["x1"],
#label="x1",width=.5)
#plt.bar(distance,data["x2"],
#label="audi", color='r',width=.5)
#plt.legend()

plt.show()

