library(ggplot2)
library(class)
#load data
knn.data <- read.csv("Downloads/MachineLearning-master/Example Data/KNN_Example_1.csv")
#take a look at data
ggplot(knn.data) + geom_point(aes(X, Y, color = Label))
#split data into 2 fold
sub <- sample(2, nrow(knn.data), replace=TRUE, prob=c(0.5, 0.5))
data.train <- knn.data[sub == 1, 1:2]
data.test <- knn.data[sub == 2, 1:2]
data.train.label <- knn.data[sub == 1, 3]
data.test.label <- knn.data[sub == 2, 3]
#train a knn model
knn.pred <- knn(train = data.train, test = data.test, cl = data.train.label, k = 3)
#show accuracy
knn.pred.dataframe <- as.data.frame(knn.pred)
knn.test.label <- as.data.frame(data.test.label)
knn.accuracy <- cbind(knn.pred.dataframe, knn.test.label)
#predict 5.3, 4.3
d1 <- 5.3
d2 <- 4.3
unknown <- cbind(d1, d2)
colnames(unknown) <- c("X", "Y")
knn.pred <- knn(train = data.train, test = unknown, cl = data.train.label, k = 3)
