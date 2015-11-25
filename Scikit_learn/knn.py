import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
#---------------------csv location------------------------------------------------
dataLocation=r'/Users/yuliang/Documents/MachineLearning-master/Example_Data/KNN_Example_1.csv'
data=pd.read_csv(dataLocation)
all_data = np.loadtxt(open(dataLocation),
    delimiter=",",
    skiprows=1,
    #dtype=np.float64
    )
#---------------------split data into Training data and Target Value------------------------------------------------
dataTraining = all_data[:,:2]
dataTarget = all_data[:,2]
#---------------------cross validation------------------------------------------------
kf = KFold(data['X'].count(), n_folds=2, shuffle = True)
print(kf)
#---------------------knn part------------------------------------------------
neigh = KNeighborsClassifier(n_neighbors=3)
for train, test in kf:
    neigh.fit(dataTraining[train],dataTarget[train])
#predict
unknownpoint = [[3.3,6.2]]
result = neigh.predict(unknownpoint)
if result == 0:
    print ("Internet Service Provider Alpha")
elif result == 1:
    print ("Internet Service Provider Beta")
else:
    print ("Unexpected prediction")
#probability of the unknownpoint belonging to the two classes
print(neigh.predict_proba([[3.3,6.2]]))
#mean accuracy of each fold on the given test data and labels.
for train, test in kf:
    print (neigh.score(dataTraining[test],dataTarget[test]))
#---------------------data of the two classses------------------------------------------------
dataAlpha = data[data['Label']==0]
dataBeta = data[data['Label']==1]

x_dataAlpha = dataAlpha['X'].reshape(dataAlpha['X'].count(),1)
y_dataAlpha = dataAlpha['Y'].reshape(dataAlpha['Y'].count(),1)
x_dataBeta = dataBeta['X'].reshape(dataBeta['X'].count(),1)
y_dataBeta = dataBeta['Y'].reshape(dataBeta['Y'].count(),1)
#---------------------plot two classes------------------------------------------------
plt.title ('Alpha and Beta class')
plt.scatter(x_dataAlpha, y_dataAlpha, color='red') 
plt.scatter(x_dataBeta, y_dataBeta, color='blue')

plt.show()