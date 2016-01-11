package main

import (
	"fmt"
	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/linear_models"
)

func main() {
	// read train dataset
	trainData, err := base.ParseCSVToInstances("datasets/linear regression.csv", true)
	if err != nil {
		panic(err)
	}
	// read test dataset
	testData, err := base.ParseCSVToInstances("datasets/linear regression test.csv", true)
	if err != nil {
		panic(err)
	}
	// create a new linear regression model
	lr := linear_models.NewLinearRegression()
	// fit model
	lr.Fit(trainData)
	fmt.Println(testData)
	// test model and print predictions
	predictions, err := lr.Predict(testData)
	fmt.Println(predictions)

}
