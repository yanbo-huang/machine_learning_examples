library(stats)
data <- read.csv("Desktop/Q1 Course/FP/MachineLearningSamples/extra-data/snsdata.csv", header = T)
#check missing values
sapply(data, function(x) sum(is.na(x)))
#use dummy coding to create 2 new columns as there are many missing values in gender column
data$female <- ifelse(data$gender == "F" & !is.na(data$gender), 1, 0)
data$no_gender <- ifelse(is.na(data$gender), 1, 0)
#impute missing value with mean value of each graduation year
ave_age <- ave(data$age, data$gradyear, FUN = function(x) mean(x, na.rm = TRUE))
data$age <- ifelse(is.na(data$age), ave_age, teens$age)
#remove 2 column
data <- data[, -2]
#train a k-means model
sns_model <- kmeans(data, 5)
#take a look at model
sns_model$clusters
sns_model$centers