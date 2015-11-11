/**
  * Created by yuliang on 10/11/15.
  */
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.mllib.linalg.Vectors


object test
{
  def main(args: Array[String])
  {
    val conf = new SparkConf().setAppName("test").setMaster("local")
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    val sc = new SparkContext(conf)
    var t0 = System.currentTimeMillis
    // Main code here //
    val csvPath = "/Users/yuliang/Documents/MachineLearning-master/ExampleData/OLS_Regression_Example_3.csv"
    val data = sc.textFile(csvPath)

    //split data into train/test
    val splitData = data.map(s => (s.split(",")(0), s.split(",")(1), s.split(",")(2)))
    val splits = splitData.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0)
    val test = splits(1)

    //training data: male and female
    val train_maleData = training.filter(s =>(s._1 == "Male")).map(s=>(s._2, s._3))
    val train_femaleData = training.filter(s =>(s._1 == "Female")).map(s=>(s._2, s._3))

    //train data model
    val train_maleData_used = train_maleData.map { s =>
      val height = s._1
      val weight = s._2
      LabeledPoint(weight.toDouble, Vectors.dense(height.toDouble))
    }.cache()

    val train_femaleData_used = train_femaleData.map{ s =>
      val height = s._1
      val weight = s._2
      LabeledPoint(weight.toDouble, Vectors.dense(height.toDouble))
    }.cache()

    val stepsize = 0.001
    val numIterations = 2000
    val model_male = LinearRegressionWithSGD.train(train_maleData_used,numIterations,stepsize)
    val model_female = LinearRegressionWithSGD.train(train_femaleData_used,numIterations,stepsize)


    //test data: male and female
    val test_maleData = test.filter(s=>(s._1 == "Male"))
    val test_femaleData = training.filter(s =>(s._1 == "Female"))

    // Evaluate model on training examples and compute training error
    val valuesAndPreds_1 = train_maleData_used.map { s =>
      val prediction = model_male.predict(s.features)
      (s.label, prediction)
    }
    valuesAndPreds_1.take(10).foreach(println)

    val valuesAndPreds_2 = train_femaleData_used.map { s =>
      val prediction = model_female.predict(s.features)
      (s.label, prediction)
    }
    valuesAndPreds_2.take(10).foreach(println)

    val MSE_1 = valuesAndPreds_1.map{case(v, p) => math.pow((v - p), 2)}.mean()
    val MSE_2 = valuesAndPreds_1.map{case(v, p) => math.pow((v - p), 2)}.mean()
    println("training male Mean Squared Error = " + MSE_1)
    println("training female Mean Squared Error = " + MSE_2)
    // Main code end //

    val et = (System.currentTimeMillis - t0) / 1000
    val mins = et / 60
    val secs = et % 60
    println( "Time taken = %d mins %d secs".format(mins, secs) )
  }
}
