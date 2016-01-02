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
LabeledPoint(s._3, Vectors.dense(s._1, s._2))}.cache()
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
