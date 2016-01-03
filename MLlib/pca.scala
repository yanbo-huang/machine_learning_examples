import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.feature.PCA

object pca
{
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("test").setMaster("local")
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    val sc = new SparkContext(conf)
    var t0 = System.currentTimeMillis
    // Main code start //
    val csvPath = "/Users/wbcha/Downloads/MachineLearning-master/Example Data/PCA_Example_1.csv"
    val rawData = sc.textFile(csvPath)
    //remove the first line (csv head)
    val dataWithoutHead = rawData.mapPartitionsWithIndex { (idx, iter) => if (idx == 0) iter.drop(1) else iter }
    //split data according to comma
    val dataRdd = dataWithoutHead.map(s => (s.split(",")(0), s.split(",")(1), s.split(",")(2).toDouble))
    //group data by date
    val groupByDate = dataRdd.groupBy(s => s._1).sortBy(s => s._1)
    //get date tuple and double tuple
    val dateTuples = groupByDate.map(s => (s._1))
    val rawDoubleTuples = groupByDate.map(s => s._2)
    val doubleTuples = rawDoubleTuples.map(s => s.map(t => t._3))
    //combine dateTuples and doubleTuples
    val finalRdd = dateTuples.zip(doubleTuples)
    //generate Vectors for build matrix
    val rows = finalRdd.map(s => s._2).map{line =>
      val valuesString = line.mkString(",")
      val values = valuesString.split(",").map(_.toDouble)
      Vectors.dense(values)
    }
    //compute principal component, in this case project 24 dimensionals into 1 dimensional space
    val matrix = new RowMatrix(rows)
    val pc: Matrix = matrix.computePrincipalComponents(1)
    println("Principal components are:\n" + pc)
    // Project the rows to the linear space spanned by 1 principal components.
    val projected: RowMatrix = matrix.multiply(pc)
    projected.rows.foreach(println)
    // Main code end //

    val et = (System.currentTimeMillis - t0) / 1000
    val mins = et / 60
    val secs = et % 60
    println("Time taken = %d mins %d secs".format(mins, secs))
  }
}