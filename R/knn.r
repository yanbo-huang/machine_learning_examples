library(ggplot2)
#load data
knn.data <- read.csv("Downloads/MachineLearning-master/Example Data/KNN_Example_1.csv")
#take a look at data
ggplot(data) + geom_point(aes(X, Y, color = Label))
