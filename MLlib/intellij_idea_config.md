    val conf = new SparkConf().setAppName("test").setMaster("local")
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    val sc = new SparkContext(conf)

   ##test
    var t0 = System.currentTimeMillis

    // Main code here ////////////////////////////////////////////////////
    val count = sc.parallelize(1 to 5000000).map{i =>
      val x = Math.random()
      val y = Math.random()
      if(x*x + y*y < 1) 1 else 0
    }.reduce(_+_)
    println("---------------------------------------")
    println(4.0*count/5000000)
    println("---------------------------------------")




    val et = (System.currentTimeMillis - t0) / 1000
    val mins = et / 60
    val secs = et % 60
    println( "{Time taken = %d mins %d secs}".format(mins, secs) )
