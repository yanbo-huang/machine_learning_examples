import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint

object kmeans
{
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("test").setMaster("local")
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    val sc = new SparkContext(conf)
    var t0 = System.currentTimeMillis
    // Main code start //
    val csvPath = "/Users/wbcha/Desktop/Q1 Course/FP/MachineLearningSamples/extradata/iris.csv"
    val rawData = sc.textFile(csvPath)
    //remove the first line (csv head)
    val dataWithoutHead = rawData.mapPartitionsWithIndex { (idx, iter) => if (idx == 0) iter.drop(1) else iter }
    //split data with comma, remove the label column to change it to a unsupervised problem
    val splitData = dataWithoutHead.map(s => (s.split(",")(0), s.split(",")(1), s.split(",")(2), s.split(",")(3)))
    val parsedData = splitData.map(s => Vectors.dense(s._1.toDouble, s._2.toDouble, s._3.toDouble, s._4.toDouble)).cache()
    //cluster data into 3 classes with kmeans
    val numOfClusters = 3
    val numIterations = 100
    val clusters = KMeans.train(parsedData, numOfClusters, numIterations)
    // Evaluate clustering by computing Within Set Sum of Squared Errors
    val WSSSE = clusters.computeCost(parsedData)
    println("Within Set Sum of Squared Errors = " + WSSSE)
    // Main code end //

    val et = (System.currentTimeMillis - t0) / 1000
    val mins = et / 60
    val secs = et % 60
    println("Time taken = %d mins %d secs".format(mins, secs))
  }
}