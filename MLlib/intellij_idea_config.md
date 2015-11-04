intellij idea configuration for spark app:
defult: spark is already in your computer
1. scala plugin 
2. at the root directory of spark, run "sbt/sbt gen-idea"
3. create a scala project and write code in a scala script
4. file->project structure-> library->+
    choose java
    browse : spark->assembly->target->scala-2.xx->spark-assembly-xxx-hadoopxxx.jar
    choose ok
    choose file type: javadocs, classes, jar
5.  in the scala script: 

    val conf = new SparkConf().setAppName("test").setMaster("local")
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    val sc = new SparkContext(conf)
    var t0 = System.currentTimeMillis

    // Main code here ////////////////////////////////////////////////////

    val et = (System.currentTimeMillis - t0) / 1000
    val mins = et / 60
    val secs = et % 60
    println( "{Time taken = %d mins %d secs}".format(mins, secs) )
