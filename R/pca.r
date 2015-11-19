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
#apply PCA 
pca.model = prcomp(data[,2:ncol(data)])
#Get PCA component PCA component 1:
PC1 <- pca.model$x[,"PC1"]
#add id as duration days:
duration <- 1:nrow(PC1)
#transform into data frame and combine
PC1 <- as.data.frame(PC1)
duration <- as.data.frame(duration)
PC1 <- cbind(PC1, duration)
colnames(PC1) <- c("feature", "duration")
#draw plot 
pc1_plot <- qplot(PC1$duration, PC1$feature)