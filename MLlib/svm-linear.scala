import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.evaluation.MulticlassMetrics

object svm
{
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("test").setMaster("local")
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    val sc = new SparkContext(conf)
    var t0 = System.currentTimeMillis
    // Main code start //
    val csvPath = "/Users/wbcha/Downloads/MachineLearning-master/Example Data/OLS_Regression_Example_3.csv"
    val rawData = sc.textFile(csvPath)
    //remove the first line (csv head)
    val dataWithoutHead = rawData.mapPartitionsWithIndex { (idx, iter) => if (idx == 0) iter.drop(1) else iter }
    //split data with comma
    val splitData = dataWithoutHead.map(s => (s.split(",")(0), s.split(",")(1), s.split(",")(2)))
    //replace different sexual with different numbers
    //Male -> class 0, Female -> class 1
    //Because MLLib SVM Classification labels should be 0 or 1
    val dataSpecies = splitData.map{s =>
      var sexual = 0
      if(s._1 equals "\"Male\""){
        sexual = 0
      }else{
        sexual = 1
      }
      (s._2.toDouble, s._3.toDouble, sexual.toDouble)
    }
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
    //build a Linear SVM model
    val numIterations = 250
    val model = SVMWithSGD.train(passedTrainData, numIterations)

    //predict based on test data
    val predictionAndLabels = passedTestData.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }
    // Get evaluation metrics.
    val metrics = new MulticlassMetrics(predictionAndLabels)
    val precision = metrics.precision
    println("Precision = " + precision)
    // Main code end //

    val et = (System.currentTimeMillis - t0) / 1000
    val mins = et / 60
    val secs = et % 60
    println("Time taken = %d mins %d secs".format(mins, secs))
  }
}