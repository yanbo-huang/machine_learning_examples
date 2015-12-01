##Machine Learning with Graphlab
###Linear Regression

import graphlab and load data

```python
import graphlab
data = graphlab.SFrame.read_csv("Downloads/MachineLearning-master/Example Data/OLS_Regression_Example_3.csv")
```

Set visualize target to ipython notebook

```python
graphlab.canvas.set_target("ipynb")
```

Replace Us metrics with cm and kilo's

```python
height_transform = lambda x:x * 2.54
weight_transform = lambda x:x * 0.45359237
data['Height'] = height_transform(data['Height'])
data['Weight'] = weight_transform(data['Weight'])
```

Visulize:

```python
data.show(view="Scatter Plot", x = "Height", y = "Weight")
```

![linear-ipython](img/ipython1.png)

Train a linear regression model

```python
lm_model = graphlab.linear_regression.create(data, target="Weight", features=['Gender', 'Height'])
```

we can get that rmse = 4.54
predict weight based on male, height 170, female 170
firstly construct a SFrame

```python
predict_data = graphlab.SFrame({'Gender':['Male','Female'], "Height":[170,170]})
```

![linear-ipython-2](img/ipython2.png)

Predict based on data

```python
lm_model.predict(predict_data)
```

![linear-ipython-3](img/ipython3.png)