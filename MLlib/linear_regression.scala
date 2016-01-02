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
    // Main code start //
    val csvPath = "/Users/wbcha/Downloads/MachineLearning-master/Example Data/OLS_Regression_Example_3.csv"
    val rawData = sc.textFile(csvPath)
    //remove the first line (csv head)
    val dataWithoutHead = rawData.mapPartitionsWithIndex{(idx, iter) => if (idx == 0) iter.drop(1) else iter}
    //split comma between string and transform rawdata into RDD
    //Since the data is in US metrics
    //inch and pounds we will recalculate this to cm and kilo's
    val dataNewMetrics = dataWithoutHead.map(s => (s.split(",")(0), s.split(",")(1).toDouble * 2.54, s.split(",")(2).toDouble * 0.45359237))
    //Replace male with 1 and female with 2
    val dataFinal = dataNewMetrics.map{s=>
      var sexual = 2
      if(s._1 equals "\"Male\""){
        sexual = 1
      }
      (sexual, s._2, s._3)
    }
    val trainData = dataFinal.map{ s =>
      //Label Point, construct a matrix, first arg is label to be predicted,
      //second argument is a vector, argument type must be double
      LabeledPoint(s._3, Vectors.dense(s._1, s._2))
    }.cache()
    trainData.take(8000).foreach(println)
    //we tried different step size
    //first tried 1 and 0.001, 1 returns Nan and 0.001 returns Nan
    //then we tried 0.001 model error still very large
    //tried 0.0005 model error is 11
    //tried 0.0004 model error is 10.73
    //tried 0.0003 model remains to be 10.73
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

    val testdata_male = Vectors.dense(1.0, 170.0)
    val testdata_female = Vectors.dense(2.0, 170.0)
    println("------------------------")
    println("Male, height 170 is predicted to be:" + model.predict(testdata_male))
    println("Female, height 170 is predicted to be:" + model.predict(testdata_female))
    println("------------------------")
    
    // Main code end //

    val et = (System.currentTimeMillis - t0) / 1000
    val mins = et / 60
    val secs = et % 60
    println( "Time taken = %d mins %d secs".format(mins, secs) )
  }
}