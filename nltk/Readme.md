# NLTK
This Readme file illustrates how to implement naive bayes using NLTK based on the [Mike's blog](https://xyclade.github.io/MachineLearning/). 

[NLTK](http://www.nltk.org/), Natural Language Toolkit is a useful toolkit to work with natural language
* [Naive Bayes](#naive-bayes)

##Naive Bayes

Get files from directory
```python
def getFilesFromDir (path):
    dir_content = os.listdir(path)
    dir_clean = filter(lambda x: (".DS_Store" not in x) and ("cmds" not in x), dir_content)
    return dir_clean
```
Get message from file
```python
def getMessage (path):
#...
```
Get and filter useful words from message
```python
def getMessageWords(file_msg, stopwords = []):
    file_msg = ''.join(file_msg)
    file_msg = re.sub('3D', '', file_msg)
    file_msg = re.sub(r'([^\s\w]|_)+', '', file_msg)
    
    file_msg_words = wordpunct_tokenize(file_msg.replace('=\n', '').lower())
    file_msg_words = filter(lambda x: x not in stopwords, file_msg_words)
    file_msg_words = [w for w in file_msg_words if re.search('[a-zA-Z]', w) and len(w) > 1]
    return file_msg_words
```
Get stop words
```python
def getStopWords (path):
    fo = open (path)
    lines = fo.readlines()
    lines_clean = map(lambda x: str.replace(x, '\n', ''), lines)
    fo.close()
    return lines_clean
```
Get top features
```python
def getFeatures(file_msg, **kwargs):
    file_msg_words = getMessageWords(file_msg, **kwargs)
    words_list = nltk.FreqDist(file_msg_words)
    words_list_common = words_list.most_common()
    topFeatures = ["" for x in range(amountOfFeaturesPerSet)]
    for index in range(amountOfFeaturesPerSet):
        topFeatures[index]= words_list_common[index][0]
    return topFeatures
```
Get feature labels
```python
def getFeaturesLabel(file_msg, label, allFeature, feature_extractor, **kwargs):
    features_label = []
    for w in file_msg:
        features = feature_extractor(w, allFeature, **kwargs)
        features_label.append((features, label))
    return features_label
```
Words Indicator: true for words in the dictionary
```python
def wordsIndicator(file_msg, allFeature, **kwargs):
    file_msg_words = getMessageWords2(file_msg, **kwargs)
    featureWords = allFeature
    features_dict = defaultdict(list)
    
    for w in file_msg_words:
        if w in allFeature:
            features_dict[w] = True
    return features_dict
```
Train the naive bayes classifier and print classify accuracy
```python
naive_bayes_classifier = NaiveBayesClassifier.train(trainFeature)
print ('Test Spam accuracy: %.4f' %nltk.classify.accuracy(naive_bayes_classifier, spamTestFeature))
print ('Test Ham accuracy: %.4f' %nltk.classify.accuracy(naive_bayes_classifier, hamTestFeature))
```
Run the code several times and change amount of features to take, we get the following results:

###Ham Result
| Amount of Features to take        | Ham (Correct)           | Spam  |
| ----------------------------------|:-----------------------:| -----:|
| 50                                | 80.43%                  | 19.57%|
| 100                               | 85.07%                  | 14.93%|
| 200                               | 89.86%                  | 10.14%|
| 400                               | 92.50%                  | 7.50% |
###Spam Result
| Amount of Features to take        | Spam (Correct)          | Ham   |
| ----------------------------------|:-----------------------:| -----:|
| 50                                | 96.21%                  | 3.79% |
| 100                               | 97.92%                  | 2.08% |
| 200                               | 98.85%                  | 1.15% |
| 400                               | 99.07%                  | 0.93% |