# Machine learning using Scikit-learn
## Linear Regression
Importing and reading csv data
```python
dataLocation=r'C:\Users\Yanbo Huang\Desktop\Python exercises\OLS_Regression_Example_3.csv'
data=pd.read_csv(dataLocation)
```
Data visualization

![scatter plot](https://raw.githubusercontent.com/fptudelft/MachineLearningSamples/master/Scikit_learn/imgs/lr1.jpg?token=ALUsnLiC34GZooFQnLFBiZVhriTokN9hks5WaGvpwA%3D%3D)

Changing gender property to numeric data, _inch_ to _cm_ and _bound_ to _kg_
```python
data['Gender']=(data['Gender']!='Male').astype(int)
data['Height']*=2.54
data['Weight']*=0.45359237
```
Linear regression model building
```python
Features=data[['Gender','Height']].values
Label=data['Weight'].values
regr=linear_model.LinearRegression()
regr.fit(Features,Label)
```
Weight rediction for women and man with 170cm hight  
```python
featuresTest=[[0,170.0],[1,170.0]]
print 'Input features:\n',featuresTest
print 'predictions:\n',regr.predict(featuresTest)
```
Model coefficients and weight predicitons with unit in _kg_
![prediction](https://raw.githubusercontent.com/fptudelft/MachineLearningSamples/master/Scikit_learn/imgs/lr2.jpg?token=ALUsnDXdWUClZASD1B3tFXMGOAont-34ks5WaGxTwA%3D%3D)


