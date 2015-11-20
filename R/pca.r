library(ggplot2)
library(dplyr)
library(reshape2)
#load data
data <- read.csv("Downloads/MachineLearning-master/Example Data/PCA_Example_1.csv", stringsAsFactors=F)
#change the first column format as Date
data$Date = as.Date(data$Date)
#transform stock data into Date Stock1  Stock2 .... Stock24
data <- reshape(data, idvar = "Date", timevar = "Stock", direction = "wide")
#sort data by date, asc
data <- arrange(data, Date)
#change column id(sort by A, B....Z)
data<-data[,order(colnames(data),decreasing=F)]
#apply PCA 
pca.model = prcomp(data[,1:ncol(data)-1])
#Get PCA component PCA component 1:
PC1 <- pca.model$x[,"PC1"]
#add id as duration days:
duration <- 1:length(PC1)
#transform into data frame and combine
PC1 <- as.data.frame(PC1)
duration <- as.data.frame(duration)
PC1 <- cbind(PC1, duration)
colnames(PC1) <- c("feature", "duration")
#draw plot 
pc1_plot <- qplot(PC1$duration, PC1$feature)
#verify data path
data.verify <- read.csv("Downloads/MachineLearning-master/Example Data/PCA_Example_2.csv", stringsAsFactors = F)
data.verify$Date <- as.Date(data.verify$Date)
#subset data, only contains 2 columns, date and close
data.verify <- data.verify[,c(1,5)]
#sort by date
data.verify <- arrange(data.verify, Date)
#add duration
duration.verify <- 1:nrow(data.verify)
duration.verify <- as.data.frame(duration.verify)
data.verify <- cbind(duration.verify, data.verify)
#normalize data
max.value <- max(data.verify$Close)
min.value <- min(data.verify$Close)
range.value <- max.value - min.value
data.verify$Close <- data.verify$Close/range.value
#plot data
qplot(data.verify$duration.verify, data.verify$Close)
#normalize to the same scale
#I wrote the normalize part into the readme.md in R folder