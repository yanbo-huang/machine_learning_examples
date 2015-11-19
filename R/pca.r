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
pca.model = prcomp(data)