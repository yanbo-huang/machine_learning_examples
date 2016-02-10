package main

import (
	"fmt"
	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/ensemble"
	"github.com/sjwhitworth/golearn/evaluation"
	"github.com/sjwhitworth/golearn/trees"
	"math/rand"
)

func main() {

	var tree base.Classifier
	rand.Seed(44111342)

	iris, err := base.ParseCSVToInstances("iris_headers.csv", true)
	if err != nil {
		panic(err)
	}
	// training-test split
	trainData, testData := base.InstancesTrainTestSplit(iris, 0.60)
	fmt.Println(trainData)
	//
	// ID3
	//
	tree = trees.NewID3DecisionTree(0.6)
	// (Parameter controls train-prune split.)

	// Train the ID3 tree
	err = tree.Fit(trainData)
	if err != nil {
		panic(err)
	}

	// Generate predictions
	predictions, err := tree.Predict(testData)
	if err != nil {
		panic(err)
	}

	// Evaluate
	fmt.Println("ID3 Performance (information gain)")
	cf, err := evaluation.GetConfusionMatrix(testData, predictions)
	if err != nil {
		panic(fmt.Sprintf("Unable to get confusion matrix: %s", err.Error()))
	}
	fmt.Println(evaluation.GetSummary(cf))

	//
	// Random Forests
	//
	tree = ensemble.NewRandomForest(70, 3)
	err = tree.Fit(trainData)
	if err != nil {
		panic(err)
	}
	predictions, err = tree.Predict(testData)
	if err != nil {
		panic(err)
	}
	fmt.Println("RandomForest Performance")
	cf, err = evaluation.GetConfusionMatrix(testData, predictions)
	if err != nil {
		panic(fmt.Sprintf("Unable to get confusion matrix: %s", err.Error()))
	}
	fmt.Println(evaluation.GetSummary(cf))
}
