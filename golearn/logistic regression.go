package main

import (
	"fmt"
	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/evaluation"
	"github.com/sjwhitworth/golearn/linear_models"
)

func main() {
	// read train dataset
	rawData, err := base.ParseCSVToInstances("iris1.csv", true)
	if err != nil {
		panic(err)
	}

	trainData, testData := base.InstancesTrainTestSplit(rawData, 0.10)
	lr, err := linear_models.NewLogisticRegression("l2", 1.0, 1e-6)
	lr.Fit(trainData)
	predictions, err := lr.Predict(testData)
	if err != nil {
		panic(err)
	}

	// Evaluate
	fmt.Println("logistic regression performance")
	cf, err := evaluation.GetConfusionMatrix(testData, predictions)
	if err != nil {
		panic(fmt.Sprintf("Unable to get confusion matrix: %s", err.Error()))
	}
	fmt.Println(evaluation.GetSummary(cf))

}
