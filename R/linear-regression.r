library(ggplot2)
#load data
data <- read.csv("Downloads/MachineLearning-master/Example Data/OLS_Regression_Example_3.csv", stringsAsFactors=F))
#replace gender with 0 and 1, male and female respectively
data$Gender[which(data$Gender=="Male")] = 0
data$Gender[which(data$Gender=="Female")] = 1
#change us metrics to eu matrics
data$Height = data$Height * 2.54
data$Weight = data$Weight * 0.45359237
#plot data (human)
human.plot <- ggplot(data)
human.plot + geom_point(aes(x = Height, y = Weight)) + labs(title = "Weight and heights for male and females")
#plot data (female and male separate)
maleFemale.plot <- ggplot(data)
maleFemale.plot + geom_point(aes(x = Height, y = Weight, color = Gender)) + labs(title = "Weight and heights for male and females")
#train a linear model
lm_model <- lm(Weight~Gender+Height, data=data)
summary(lm_model)
#predict a test data
test.data <- data.frame(0, 170)
colnames(test.data) <- c("Gender", "Height")
test.data$Gender <- as.factor(test.data$Gender)
predict(lm_model, test.data)