import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint

object evaluation
{
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("test").setMaster("local")
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    val sc = new SparkContext(conf)
    var t0 = System.currentTimeMillis
    // Main code start //
    val csvPath = "/Users/wbcha/Desktop/Q1 Course/FP/MachineLearningSamples/extra-data/iris.csv"
    val rawData = sc.textFile(csvPath)
    //remove the first line (csv head)
    val dataWithoutHead = rawData.mapPartitionsWithIndex { (idx, iter) => if (idx == 0) iter.drop(1) else iter }
    //split data with comma
    val splitData = dataWithoutHead.map(s => (s.split(",")(0), s.split(",")(1), s.split(",")(2), s.split(",")(3), s.split(",")(4)))
    //replace different species of flowers with different numbers
    //Iris-setosa -> class 1, Iris-virginica -> class 2, Iris-versicolor -> class 3
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

    //split data to train and test, training (60%) and test (40%).
    val splits = dataSpecies.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0)
    val test = splits(1)

    val passedTrainData = training.map{s =>
      LabeledPoint(s._5, Vectors.dense(s._1, s._2, s._3, s._4))
    }
    //build a logistic regression model
    val model = new LogisticRegressionWithLBFGS().setNumClasses(3).run(passedTrainData)
    //train model on test set
    val passedTestData = test.map{s =>
      LabeledPoint(s._5, Vectors.dense(s._1, s._2, s._3, s._4))
    }
    //predict based on test data
    val predictionAndLabels = passedTestData.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }
    // Get evaluation metrics.
    // If classification task is binary classification, use BinaryClassificationMetrics, else, use MulticlassMetrics
    val metrics = new MulticlassMetrics(predictionAndLabels)
    //confusion matrix
    val cfm = metrics.confusionMatrix
    //evaluate with precision
    val precision = metrics.precision
    //evaluate with recall
    val recall = metrics.recall
    //evaluate with F1 score
    val fscore = metrics.fMeasure
    println("---------------------------")
    println("confusion matrix: ")
    println(cfm)
    println("Precision = " + precision)
    println("recall = " + recall)
    println("F1 Score = " + fscore)
    println("---------------------------")
    // Main code end //

    val et = (System.currentTimeMillis - t0) / 1000
    val mins = et / 60
    val secs = et % 60
    println("Time taken = %d mins %d secs".format(mins, secs))
  }
}