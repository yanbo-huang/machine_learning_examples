/**
  * Created by yuliang on 09/01/16.
  */

import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import scala.io.Source
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}


object naive_bayes
{
  def main(args: Array[String])
  {
    val conf = new SparkConf().setAppName("test").setMaster("local")
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    val sc = new SparkContext(conf)
    val t0 = System.currentTimeMillis
    // Main code start //
    //path
    val pathBase = "/Users/yuliang/Downloads/naive_bayes"
    val pathHamTrain = pathBase + "/easy_ham"
    val pathHamTest = pathBase + "/easy_ham_2"
    val pathSpamTrain = pathBase + "/spam"
    val pathSpamTest = pathBase + "/spam_2"
    val pathStopWords = pathBase + "/stopwords.txt"
    //file amount and feature amount to take
    val fileAmount = 500
    val featureAmount = 100
    //get stop words
    def getStopWords(stopWordsPath:String): List[String] =
    {
      val stopWordsSource = Source.fromFile(stopWordsPath, "latin1")
      val stopWordsLines = stopWordsSource.mkString.split("\n")
      return stopWordsLines.toList
    }
    val stopWords = getStopWords(pathStopWords)
    //get documents
    def getDocuments(emailPath: String): RDD[String] =
    {
     val emailDocuments =  sc.wholeTextFiles(emailPath).filter(x => !x._1.toString.contains("cmd") && !x._1.toString.contains(".DS_Store"))
                                                       .map(x => x._2.mkString
                                                                     .substring(x._2.indexOf("\n\n"))
                                                                     .replace("\n", " ")
                                                                     .replace("\t", " ")
                                                                     .replace("3D", "")
                                                                     .replaceAll("[^a-zA-Z\\s]", "")
                                                                     .toLowerCase

                                                       )
     return emailDocuments
    }

    val hamDocumentsTrain = getDocuments(pathHamTrain)
    val hamDocumentsTest = getDocuments(pathHamTest)
    val spamDocumentsTrain = getDocuments(pathSpamTrain)
    val spamDocumentsTest = getDocuments(pathSpamTest)

    //Ham Top Features
    val hamDictionary = hamDocumentsTrain.flatMap(x => x.split(" ")).filter(s => s.nonEmpty && !stopWords.contains(s))
    val hamFeatures = hamDictionary.groupBy(w => w).mapValues(_.size).sortBy(_._2, ascending = false)
    val hamTopFeatures = hamFeatures.map(x => x._1).take(featureAmount)

    //Spam Top Features
    val spamDictionary = spamDocumentsTrain.flatMap(x => x.split(" ")).filter(s => s.nonEmpty && !stopWords.contains(s))
    val spamFeatures = spamDictionary.groupBy(w => w).mapValues(_.size).sortBy(_._2, ascending = false)
    val spamTopFeatures = spamFeatures.map(x => x._1).take(featureAmount)

    val emailFeatures = (hamTopFeatures ++ spamTopFeatures).toSet
    emailFeatures.foreach(println)

    // Main code end //
    val et = (System.currentTimeMillis - t0) / 1000
    val mins = et / 60
    val secs = et % 60
    println("Time taken = %d mins %d secs".format(mins, secs))
  }

}
