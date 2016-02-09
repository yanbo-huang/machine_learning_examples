# NLTK
This Readme file illustrates how to implement naive bayes using NLTK based on the [Mike's blog](https://xyclade.github.io/MachineLearning/). 

[NLTK](http://www.nltk.org/), Natural Language Toolkit is a useful toolkit to work with natural language
* [Naive Bayes](#naive-bayes)
* [Recommendation System](#recommend-system)

##Naive Bayes
Some part of the code referred from [Machine Learning for Hackers Chapter 3: Naive Bayes Text Classification] (http://slendermeans.org/ml4h-ch3.html)

Get files from directory
```python
def getFilesFromDir (path):
    dir_content = os.listdir(path)
    dir_clean = filter(lambda x: (".DS_Store" not in x) and ("cmds" not in x), dir_content)
    msg = map(lambda x: getMessage(path + '/' + x), dir_clean)
    return msg
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
    topFeatures = map(lambda x: x[0], words_list_common[:amountOfFeaturesPerSet])
    return topFeatures
```
Get feature labels
```python
def getFeaturesLabel(file_msg, label, allFeature, feature_extractor, **kwargs):
    features = map(lambda x: feature_extractor(x, allFeature, **kwargs), file_msg)
    features_label = map(lambda x: (x, label), features)
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


##Recommendation System

Firsty, we need to process the email to get the information we need.

Get files from directory
```python
def getFilesFromDir (path):
    dir_content = os.listdir(path)
    dir_clean = filter(lambda x: (".DS_Store" not in x) and ("cmds" not in x), dir_content)
    msg = map(lambda x: getFullMessage(path + '/' + x), dir_clean)
    return msg
```
Get full email message 
```python
def getFullMessage (path):
    fo = open (path)
    lines = fo.readlines()
    fo.close()
    lines = ''.join(lines)
    return lines
```
Get email subject
```python
def getSubjectFromEmail (path):
    lines = getFilesFromDir(path)
    #subject index and subject end index
    subject = map(lambda x: x[x.index('Subject:')+8: len(x)], lines)
    subject = map(lambda x: x[: x.index('\n')], subject)
    subject = map(lambda x: x.lower(), subject)
    subject = map(lambda x: re.sub('re:', '', x), subject)
    return subject
```
Get email sender
```python
def getSenderFromEmail (path):
    lines = getFilesFromDir(path)
    sender = map(lambda x: x[x.index('From:'): len(x)], lines)
    sender = map(lambda x: x[: x.index('\n')], sender)
    for index in range(len(sender)):
        if '<' in sender[index]:
            sender[index] = sender[index][sender[index].index('<')+1: sender[index].index('>')]
        else:
            sender[index] = sender[index][sender[index].index('From:') + 5: ]
            if '(' in sender[index]:
                sender[index] = sender[index][: sender[index].index('(')]
        sender[index] = re.sub(' ', '', sender[index])
    return sender
```
Get email date
```python
def getDateFromEmail (path):
    lines = getFilesFromDir(path)
    date = map(lambda x: x[x.index('Date:')+5: len(x)], lines)
    date = map(lambda x: x[: x.index('\n')], date)
    return date
``` 
Get stop words
```python
def getStopWords (path):
    fo = open (path)
    lines = fo.readlines()
    lines = map(lambda x: str.replace(x, '\n', ''), lines)
    fo.close()
    return lines
```
Then, we group the email by sender, in this part, we import the pandas and numpy package to deal with data, and import the matplotlib to plot the bar chart. Also, from the data, we find the numeric value ranges from (0.69, 6.43).

Sender bar plot
```python
x = df['sender_describe'].tolist()
y = df['sender_values'].tolist()
y_array = np.array(y)
#Use numpy function log1p to re-scale the data
y_array = map(lambda value: np.log1p(value), y_array)
index = np.arange(len(y_array))
#...
bar_sender = plt.bar(index, y_array, bar_width, alpha=opacity, color='b')
```
<img src='imgs\bar1.png' height='300'>

Next, we group the email by subject. Also, after we use the log1p to re-scale the data, we find the numeric value ranges from (0.69, 3.33). The code part is similar as the one in the previous part.
Subject bar plot
```python
x = df_subject['subject_describe'].tolist()
y = df_subject['subject_values'].tolist()

y_array = np.array(y)
y_array = map(lambda value: np.log1p(value), y_array)
index = np.arange(len(y_array))
#...
bar_subject = plt.bar(index, y_array, bar_width, alpha=opacity, color='b')
```
<img src='imgs\bar2.png' height='300'>