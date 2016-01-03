###Machine Learning with Apache Spark

####Intellij idea configuration for spark:

*defult: spark is already in your computer*

- Install scala plugin 
- At the root directory of spark, run "sbt/sbt gen-idea"
- Create a scala project and write code in the scala script
- Add spark assembly
  + Select *File*
  + Select *Project Structure*
  + Select *Libraries*
  + click *+*, add three file types, include *javadocs, classes and jar*, click ok.
+  In the scala script: 

```scala
    val conf = new SparkConf().setAppName("test").setMaster("local")
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    val sc = new SparkContext(conf)
    var t0 = System.currentTimeMillis
    // Main code here //
    // Main code end //
    val et = (System.currentTimeMillis - t0) / 1000
    val mins = et / 60
    val secs = et % 60
    println( "{Time taken = %d mins %d secs}".format(mins, secs) )
```

####Linear Regression

Load data from csv file and remove the first row:

```scala
val csvPath = "/Users/wbcha/Downloads/MachineLearning-master/Example Data/OLS_Regression_Example_3.csv"
val rawData = sc.textFile(csvPath)
val dataWithoutHead = rawData.mapPartitionsWithIndex{(idx, iter) => if (idx == 0) iter.drop(1) else iter}
```

Split data with comma and transformation to US metrics:

```scala
val dataNewMetrics = dataWithoutHead.map(s => (s.split(",")(0), s.split(",")(1).toDouble * 2.54, s.split(",")(2).toDouble * 0.45359237))
```

Replace Male and Female with 1 and 2:

```scala
val dataFinal = dataNewMetrics.map{s=>
    var sexual = 2
    if(s._1 equals "\"Male\""){
    sexual = 1
    }
    (sexual, s._2, s._3)
}
```

Generate train data into **LabeledPoint** format

```scala
val trainData = dataFinal.map{ s =>
    //Label Point, construct a matrix, first arg is label to be predicted,
    //second argument is a vector, argument type must be double
    LabeledPoint(s._3, Vectors.dense(s._1, s._2))
}.cache()
```

Train model using function **LinearRegressionWithSGD**, in **MLlib**, we need to tune parameters(such as learning rate, iteration times) ourselves, we tried different learning rate from 1 decreased to 0.0003, finnaly got a relative low train error 10.73:

```scala
val stepSize = 0.0003
val numIterations = 10000
val model = LinearRegressionWithSGD.train(trainData, numIterations, stepSize)
// Evaluate model on training examples and compute training error
val valuesAndPreds = trainData.map { s =>
  val prediction = model.predict(s.features)
  (s.label, prediction)
}
val MSE = valuesAndPreds.map{case(v, p) => math.pow((v - p), 2)}.mean()
val ModelError = math.sqrt(MSE)
println("Model Error = " + ModelError)
```

####PCA


####Logistic Regression

As apache MLlib is under construction, they do not provide kernel svm, knn. So we'll implement some existed algorithms with apache spark, first one is Logistic Regression use [IRIS](https://en.wikipedia.org/wiki/Iris_flower_data_set) dataset from [UCI](https://archive.ics.uci.edu/ml/datasets/Iris) machine learning repostory. 

First of all, import data and remove the head line:

```scala
val csvPath = "/Users/wbcha/Desktop/Q1 Course/FP/MachineLearningSamples/extradata/iris.csv"
val rawData = sc.textFile(csvPath)
val dataWithoutHead = rawData.mapPartitionsWithIndex { (idx, iter) => if (idx == 0) iter.drop(1) else iter }
```

Split data by *comma* and generate new RDD.

```scala
val splitData = dataWithoutHead.map(s => (s.split(",")(0), s.split(",")(1), s.split(",")(2), s.split(",")(3), s.split(",")(4)))
```

Because Mllib only alow double format label, so we need to replace flower species into digits, in this case, Iris-setosa -> class 1, Iris-virginica -> class 2, Iris-versicolor -> class 3.

```scala
val dataSpecies = splitData.map{s =>
  var species = 0
  if(s._5 equals "Iris-setosa"){
    species = 0
  }else if(s._5 equals "Iris-virginica"){
    species = 1
  }else{
    species = 2
  }
  (s._1.toDouble, s._2.toDouble, s._3.toDouble, s._4.toDouble, species.toDouble)
}
```

Then we split data into train set and test set in order to validate our model:

```scala
val splits = dataSpecies.randomSplit(Array(0.6, 0.4), seed = 11L)
val training = splits(0)
val test = splits(1)
```

And transform them into *LabeledPoint* format:

```scala
val passedTrainData = training.map{s =>
  LabeledPoint(s._5, Vectors.dense(s._1, s._2, s._3, s._4))
}
val passedTestData = test.map{s =>
  LabeledPoint(s._5, Vectors.dense(s._1, s._2, s._3, s._4))
}
```

Next, we trained a logistic regression model, as we know that flowers can be grouped into 3 species, set *setNumClass* equals to 3.

```scala
val model = new LogisticRegressionWithLBFGS().setNumClasses(3).run(passedTrainData)
```

Predict label based on test data:

```scala
val predictionAndLabels = passedTestData.map { case LabeledPoint(label, features) =>
  val prediction = model.predict(features)
  (prediction, label)
}
```

And finally, evaluate precision rate:

```scala
val metrics = new MulticlassMetrics(predictionAndLabels)
val precision = metrics.precision
println("Precision = " + precision)
```

The precision rate equals to 92.98%. This is just an implementation of Logistic Regression with Spark, if we use cross validation set and do some feature engineering work, the precision rate is likely to be larger.

###Linear SVM

As we said before, Spark MLlib do not provide kernel SVM yet, so we continually use sexual-height-weight(OLS_Regression_Example_3.csv) dataset, try to use Linear SVM(Large-Margin Classifier) to seperate sexual according to different weight and height.

The reason why we use this dataset is that Linear-SVM on MLlib only support **Binary Classification** at the moment.

First of all, load data and remove header line

```scala
val csvPath = "/Users/wbcha/Downloads/MachineLearning-master/Example Data/OLS_Regression_Example_3.csv"
val rawData = sc.textFile(csvPath)
//remove the first line (csv head)
val dataWithoutHead = rawData.mapPartitionsWithIndex { (idx, iter) => if (idx == 0) iter.drop(1) else iter }
```

Generate new RDD by split data with comma, then replace Male with 0, replace Female with 1:

```scala
val splitData = dataWithoutHead.map(s => (s.split(",")(0), s.split(",")(1), s.split(",")(2)))
val dataSpecies = splitData.map{s =>
  var sexual = 0
  if(s._1 equals "\"Male\""){
    sexual = 0
  }else{
    sexual = 1
  }
  (s._2.toDouble, s._3.toDouble, sexual.toDouble)
}
```

Split data into train set and test set, transform then into **LabeledPoint** format:

```scala
//split data to train and test, training (60%) and test (40%).
val splits = dataSpecies.randomSplit(Array(0.6, 0.4), seed = 11L)
val training = splits(0)
val test = splits(1)
//prepare train and test data with labeled point format
val passedTrainData = training.map { s =>
  LabeledPoint(s._3, Vectors.dense(s._1, s._2))
}
val passedTestData = test.map{s =>
  LabeledPoint(s._3, Vectors.dense(s._1, s._2))
}
```

Train a linear-SVM model:

```scala
val numIterations = 200
val model = SVMWithSGD.train(passedTrainData, numIterations)
```

We use 200 as iteration times because we tried 100, 200 and 250 three numbers, and get accuracy rate: 71.96%, 91.68% and 91.60% respectively. So we selected the iteration times with the best performance.

Follow is the code for evaluate our model:

```scala
val metrics = new MulticlassMetrics(predictionAndLabels)
val precision = metrics.precision
println("Precision = " + precision)
```

###Kmeans

We continue implement existed algorithms with apache spark, this one is **K-means** use [IRIS](https://en.wikipedia.org/wiki/Iris_flower_data_set) dataset from [UCI](https://archive.ics.uci.edu/ml/datasets/Iris) machine learning repostory. 

As Kmeans is an unsupervised machine learning algorithm, we drop the label column of iris dataset to group dataset.

First of all, load data and remove header.

```scala
val csvPath = "/Users/wbcha/Desktop/Q1 Course/FP/MachineLearningSamples/extradata/iris.csv"
val rawData = sc.textFile(csvPath)
//remove the first line (csv head)
val dataWithoutHead = rawData.mapPartitionsWithIndex { (idx, iter) => if (idx == 0) iter.drop(1) else iter }
```

Split data with comma, remove the label from dataset to use kmeans:

```scala
val splitData = dataWithoutHead.map(s => (s.split(",")(0), s.split(",")(1), s.split(",")(2), s.split(",")(3)))
val parsedData = splitData.map(s => Vectors.dense(s._1.toDouble, s._2.toDouble, s._3.toDouble, s._4.toDouble)).cache()
```

Train a K-means model:

```scala
val numOfClusters = 3
val numIterations = 100
val clusters = KMeans.train(parsedData, numOfClusters, numIterations)
```

We set **NumOfClusters** equals to 3 because we already knows that there are 3 kinds of flowers in our dataset. And of course, we tried different numIterations, as we keep increase iteration time from 100, the model error remain the same.

Evaluate model error(Within Set Sum of Squared Errors):

```scala
val WSSSE = clusters.computeCost(parsedData)
println("Within Set Sum of Squared Errors = " + WSSSE)
```

Model error equals to 78.86.





