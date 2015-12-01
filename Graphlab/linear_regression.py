# coding: utf-8
# ###Linear Regression with Graphlab 

import graphlab

#load data
data = graphlab.SFrame.read_csv("Downloads/MachineLearning-master/Example Data/OLS_Regression_Example_3.csv")

data.head()
data.tail()

#Set visualize target to ipython notebook
graphlab.canvas.set_target("ipynb")

#Replace Us metrics with cm and kilo's
height_transform = lambda x:x * 2.54
weight_transform = lambda x:x * 0.45359237


data['Height'] = height_transform(data['Height'])
data['Weight'] = weight_transform(data['Weight'])

#view data
data.show(view="Scatter Plot", x = "Height", y = "Weight")

#train a linear model
lm_model = graphlab.linear_regression.create(data, target="Weight", features=['Gender', 'Height'])

#we can get that rmse = 4.54
#predict weight based on male, height 170, female 170
#firstly construct a SFrame
predict_data = graphlab.SFrame({'Gender':['Male','Female'], "Height":[170,170]})


lm_model.predict(predict_data)



