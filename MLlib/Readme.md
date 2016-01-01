###Machine Learning with Apache Spark

###Intellij idea configuration for spark app:

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
