library(ggplot2)
library(reshape2)
library(e1071)
#load data
train.data <- read.csv("Downloads/MachineLearning-master/Example Data/SVM_Example_1.csv", stringsAsFactors=F)
#take a galance
plot1 <- ggplot(train.data) + geom_point(aes(x, y, color = label))
#change format of label from int to factor
train.data$label <- as.factor(train.data$label)
#train a simple model and predict
model <- svm(label~., data = train.data)
pred <- predict(model, train.data[,1:2])
table(pred, train.data[,3])
#get the best sigma / cost parameter based on Mike's blog
sigma <- c(0.001, 0.01, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 10.0)
cost <- c(0.001, 0.01, 0.1, 0.2, 0.5,1.0, 2.0, 3.0, 10.0, 100.0)
tuned <- tune.svm(label~., data = train.data, gamma = sigma, cost = cost)
#print the best parameter
summary(tuned)
#train a new model with the best sigma and cost
model.new <- svm(label~., data = train.data, gamma = 0.5, cost = 3)
#combine predict result with original data
df <- cbind(train.data, data.frame(ifelse(as.numeric(predict(model.new))>1,1,0)))
colnames(df) <- c("x", "y", "label", "predict")
#remove label column
df <- df[,c(1,2,4)]
#visulize prediction
predictions <- melt(df, id.vars = c("x", "y"))
plot2 <- ggplot(predictions) + geom_point(aes(x, y, color = value))
#visualize two plots in a single screen
multiplot(plot1, plot2, cols = 2)

