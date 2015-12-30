library(ggplot2)
library(reshape2)
library(e1071)
#load data
train.data = read.csv("Downloads/MachineLearning-master/Example Data/SVM_Example_2.csv", stringsAsFactors = F)
test.data = read.csv("Downloads/MachineLearning-master/Example Data/SVM_Example_2_Test_data.csv", stringsAsFactors = F)
#visualize data
ggplot(train.data) + geom_line(aes(x = x, y = y, color = factor(label)))
ggplot(test.data) + geom_line(aes(x = x, y = y, color = factor(label)))
#change format of label from int to factor
train.data$label <- as.factor(train.data$label)
#train a simple model and predict
model <- svm(label~., data = train.data, kernel = "polynomial", degree = 2)
pred <- predict(model, test.data[,1:2])
table(pred, test.data[,3])
#get the best degree / cost parameter based on Mike's blog
cost <- c(0.001, 0.01, 0.1, 0.2, 0.5,1.0, 2.0, 3.0, 10.0, 100.0)
degree <- c(2, 3, 4, 5) 
tuned <- tune.svm(label~., data = train.data, cost = cost, kernel = "polynomial", degree = degree)
#best performance cost and degree train a new model
new.model <- svm(label~. ,data = train.data, kernel = "polynomial", cost = 100, degree = 3)
pred <- predict(model, test.data[,1:2])
table(pred, test.data[,3])
#print the best parameter
summary(tuned)


