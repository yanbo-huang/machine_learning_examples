library(tm)
library(wordcloud)
library(e1071)
library(gmodels)
#load naive bayes data, sms spam/ham
nb.data <- read.csv("Downloads/Machine-Learning-With-R-master/sms_spam/sms_spam.csv", stringsAsFactors = F)
nb.data$type <- factor(nb.data$type)
#take a look at spam/ham num
table(nb.data$type)
#split messages into bag-of-words
nb.corpus <- Corpus(VectorSource(nb.data$text))
#clearn remove stop words
corpus <- tm_map(corpus, content_transformer(tolower))
nb.corpus <- tm_map(nb.corpus, removeNumbers)
nb.corpus <- tm_map(nb.corpus, removeWords, stopwords())
nb.corpus <- tm_map(nb.corpus, removePunctuation)
nb.corpus <- tm_map(nb.corpus, stripWhitespace)
#create term-document-matrix
nb.dtm <- DocumentTermMatrix(nb.corpus)
#split into train and test data
nb.train <- nb.corpus[1:4000]
nb.test <- nb.corpus[4001:5559]
nb.dtm.train <- nb.dtm[1:4000, ]
nb.dtm.test <- nb.dtm[4001:5559, ]
nb.raw.train <- nb.data[1:4000, ]
nb.raw.test <- nb.data[4001:5559, ]
spam <- subset(nb.raw.train, type == "spam")
ham <- subset(nb.raw.test, type == "ham")
#visualize
wordcloud(spam$text, max.words = 40, scale = c(3, 0.5))
wordcloud(ham$text, max.words = 40, scale = c(3, 0.5))
nb.freq <- findFreqTerms(nb.dtm.train, 5)
train.data <- DocumentTermMatrix(nb.train, list(dictionary = nb.freq))
test.data <- DocumentTermMatrix(nb.test, list(dictionary = nb.freq))
nb.train <- apply(train.data, MARGIN = 2, convert_counts)
nb.test <- apply(test.data, MARGIN = 2, convert_counts)
#build a classifier with naive bayes
nb.classifier <- naiveBayes(nb.train, nb.raw.train$type)
#make prediction
nb.pred <- predict(nb.classifier, nb.test)
#evaluate the classifier
CrossTable(nb.pred, nb.raw.test$type, prop.chisq = FALSE)

convert_counts <- function(x) {
    x <- ifelse(x > 0, 1, 0)
    x <- factor(x, levels = c(0, 1), labels = c("No", "Yes"))
    return(x)
 }
