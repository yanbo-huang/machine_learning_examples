# Machine learning using golearn

This Readme file illustrates how to implement several machine learning algorithms using golang's golearn library based on the [Mike's blog](https://xyclade.github.io/MachineLearning/).

* [Enviroment Configurations](#enviroment-configurations)
* [K-Nearest Neighbors](#k-nearest-neighbors)
## Enviroment Configurations

1. Go Enviroment
Please refer to [this blog](https://github.com/astaxie/build-web-application-with-golang/blob/master/en/01.1.md) to  install and configure your own Go development environment.It is much easier to use the second method *Using the standard installation packages*.
**Note:** Remember to write your *GOPATH*,*GOROOT*,and current *PATH* to your **bashrc**

2. Go development tool 
Install *GoSublime*, *SidebarEnhancements* and *Go Build* in your Sublime Text.
You can still refer to [the same blog](https://github.com/astaxie/build-web-application-with-golang/blob/master/en/01.4.md) for details.

3. Start Golearn
Follow the instruction in [this blog](https://github.com/sjwhitworth/golearn)
**Note:** Don't forget to change the dataset path when you test the first *knn* example.
Till now, you can run some machine learning code in the example folder.

## K-Nearest Neighbors
Parsing CSV data:
```go
rawData, err := base.ParseCSVToInstances("KNN_Example_1.csv", true)
	if err != nil {
		panic(err)
	}
```
Initialising a new KNN classifier using **euclidean** distance for voting weight and picking **k=3**
```go

cls := knn.NewKnnClassifier("euclidean", 3)
```

Since the dataset is only 100 points, we only do 2-fold cross validation which just splits the original dataset into train and test parts.Using train part to get the model and picking test data to predict the classification.
```go
	//Do a 2-fold cross validation
	trainData, testData := base.InstancesTrainTestSplit(rawData, 0.50)
	cls.Fit(trainData)

	//predict the classification of test data
	predictions := cls.Predict(testData)
```
Here we print the precision/recall model for a better evaluation of our knn model:
```go
	// Prints precision/recall metrics
	confusionMat, err := evaluation.GetConfusionMatrix(testData, predictions)
	if err != nil {
		panic(fmt.Sprintf("Unable to get confusion matrix: %s", err.Error()))
	}
```

Reference Class|True Positives|False Positives|True Negatives|Precision	|Recall|	F1 Score
---------|:--:|:--:|:--:|:--:|:--:|:--:|
1.00000000000000|26	|	3|		25|		0.8966|		0.8667|	0.8814
0.00000000000000|25	|	4|		26|		0.8621	|	0.8929|	0.8772

Then based on this table,we generate a Overall accuracy to discribe the reliability of our model:
```go
fmt.Println(evaluation.GetSummary(confusionMat))
```
In this example, *Overall accuracy* is **0.9310**

