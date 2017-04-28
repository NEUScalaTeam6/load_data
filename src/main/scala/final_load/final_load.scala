package final_load

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

import org.apache.spark.sql.DataFrame

import org.apache.spark.ml.{Pipeline, PipelineModel}

import org.apache.spark.ml.feature.{IndexToString, MaxAbsScaler, OneHotEncoder, StringIndexer, StringIndexerModel, VectorAssembler, VectorIndexer}
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.StringIndexer


/**
  * Created by liushengchen on 4/28/17.
  */
object final_load {

  def  main(args: Array[String]) {

    // Create a SparkSession.
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("Spark SQL basic example")
      .config("spark.some.config.option", "some-value")
      .getOrCreate()


    import spark.implicits._


    val file ="/Users/liushengchen/course/Scala/final_project/data/CurryFinalDatatest2s2.csv"
    val shotData = spark.read
      .format("com.databricks.spark.csv")
      .option("header", "true") // Use first line of all files as header
      .option("inferSchema", "true") // Automatically infer data types
      .load(file)

    val df_raw = shotData.select("index","period","minutes_remaining","seconds_remaining","action_type","loc_x", "loc_y", "shot_zone_basic","shot_zone_area","defender_id","defender_distance","htm","vtm","SCOREMARGIN","shot_made_flag")
    df_raw.show(1)
    df_raw.printSchema


    // full court -> half court

    // Define the UDF
    val isFullCourt_x = udf((x: Int) => {
      if (x > 47) 94-x
      else x
    })

    val isFullCourt_y= udf((x: Int, y:Int) => {
      if (x > 47) 50-y
      else y
    })

    val df_half = df_raw.withColumn("x_half",isFullCourt_x($"loc_x"))
      .withColumn("y_half",isFullCourt_y($"loc_x",$"loc_y"))
    df_half.show(2)

    /*
    2.Score margin
    "score margin" means the score difference between the home team and the visit team.
    In raw data, when the score margin is '0', the value is shown as 'TIE'.
    We need to filter out this value and convert it into '0'.
    */

    //2.1 In raw data, when the score margin is ‘0’, the value is shown as ‘TIE’. We need to filter out this value and convert it into ‘0’.

    val df_raw_score = df_half.withColumn("score_margin", when(col("SCOREMARGIN").equalTo("TIE"),0).otherwise(col("SCOREMARGIN")))


    //2.2 Categorize score margins into several groups: score_margin_category = score_margin
    // change column type
    val toInt  = udf[Int, Double]( _.toInt)
    val df_divide = df_raw_score.withColumn("score_margin_category",col("score_margin")/5)
    val df_score_category =df_divide.withColumn("score_margin_category",toInt(df_divide("score_margin_category")))
      .withColumn("defender_distance_category",toInt(df_divide("defender_distance")))

    //
    // 2.3 Consider score margin larger than 15 as a same group.
    // Define the UDF

    val isBeyond = udf((score: Int) => {
      if (score > 3) 3
      else if (score < -3) -3
      else score
    })
    val df_score_1 = df_score_category.withColumn("score_margin_category", isBeyond($"score_margin_category"))


    /*
   3. Defender distance
   // Consider score margin larger than 15 as a same group.
    */

    // Consider score margin larger than 15 as a same group.
    // Define the UDF
    val groupDistance = udf((distance: Double) => {
      if (distance < 1) 0
      else if (distance < 3) 1
      else if (distance < 5) 2
      else if (distance < 7) 3
      else if (distance < 10) 4
      else if (distance < 15) 5
      else 6
    })
    val df_ddistance = df_score_1.withColumn("defender_distance_category", groupDistance($"defender_distance_category"))


    /*
    4. Query "home team" or "visit team"
     */

    val data_select = df_ddistance.withColumn("h", when(col("htm").equalTo("GSW"),1).otherwise(0 ))
      .withColumn("v", when(col("vtm").equalTo("GSW"),1).otherwise(0 ))

    /*
    5. Dealing with categorical values
     */

    //5.1 StringIndexer helps you convert String categorical values into integer indexes of those values.

    // 5.1 Action type

    // shows distinct "action types"
    val distinctValuesDF = data_select.select(data_select("action_type")).distinct
    val distinctValuesArray = distinctValuesDF.select("action_type").rdd.map(r => r(0).asInstanceOf[String]).collect()



    def udfActionToCategory=udf((action_type: String) => {
      action_type match {
        case t if t.contains("Tip") => "Tip Shot"
        case t if t.contains("Layup") => "Layup Shot"
        case t if t.contains("Jump") => "Jump Shot"
        case t if t.contains("Bank") => "Bank Shot"
        case t if t.contains("Dunk") => "Dunk Shot"
        case t if t.contains("Hook") => "Hook Shot"
        case t if t.contains("Fadeaway") => "Fadeaway Shot"
        case _ => "Else"
      }})
    val transformed_category = data_select.withColumn("action_type_category", udfActionToCategory(data_select("action_type")))


    val indexer = new StringIndexer()
      .setInputCol("action_type_category")
      .setOutputCol("action_categoryIndex")

    // Defind a method: indexStringsColumns

    def indexStringColumns(df:DataFrame, cols:Array[String]):DataFrame ={
      var newdf = df
      for (col <- cols){
        val si = new StringIndexer().setInputCol(col).setOutputCol(col+ "-num")
        val sm:StringIndexerModel = si.fit(newdf)
        newdf = sm.transform(newdf).drop(col)
        newdf = newdf.withColumnRenamed(col+"-num",col)
      }
      newdf
    }

    val dfnumeric = indexer.fit(transformed_category).transform(transformed_category)

    //Encoding the "action_type" with OneHotEncoder
    def oneHotEncodeColumns(df:DataFrame, cols:Array[String]):DataFrame ={
      var newdf = df
      for (c<-cols){
        val onehotenc = new OneHotEncoder().setInputCol(c).setOutputCol("action_type_onehot").setDropLast(false)
        newdf =onehotenc.transform(newdf)
      }
      newdf
    }
    val dfhot = oneHotEncodeColumns(dfnumeric, Array("action_categoryIndex"))


// period
    val df_period = dfhot.withColumn("period", when($"period">5, 5).otherwise(col("period")))

    //Encoding the "period" with OneHotEncoder
    def oneHotEncodeColumns_period(df:DataFrame, cols:Array[String]):DataFrame ={
      var newdf = df
      for (c<-cols){
        val onehotenc = new OneHotEncoder().setInputCol(c).setOutputCol("period_onehot").setDropLast(false)
        newdf =onehotenc.transform(newdf)
      }
      newdf
    }
    val df_hot_period = oneHotEncodeColumns_period(df_period, Array("period"))


    // shot zone
    val getConcatenated = udf( (first: String, second: String) => { first + " " + second } )

    //use withColumn method to add a new column called newColName
    val df_shotZone = df_hot_period.withColumn("shot_zone", getConcatenated($"shot_zone_basic",$"shot_zone_area"))

    val df_shotZone_numeric = indexStringColumns(df_shotZone, Array("shot_zone"))
    //df_shotZone_numeric.select("shot_zone","shot_zone-num").show(2,false)


    //Encoding the "shot_zone" with OneHotEncoder
    def oneHotEncodeColumns_shotzone(df:DataFrame, cols:Array[String]):DataFrame ={
      var newdf = df
      for (c<-cols){
        val onehotenc = new OneHotEncoder().setInputCol(c).setOutputCol("shot_zone_onehot").setDropLast(false)
        newdf =onehotenc.transform(newdf)
      }
      newdf
    }
    val df_hot_shotzone = oneHotEncodeColumns_shotzone(df_shotZone_numeric, Array("shot_zone"))


    //defender_distance

    //Encoding the "defender_distance" with OneHotEncoder
    def oneHotEncodeColumns_ddfender(df:DataFrame, cols:Array[String]):DataFrame ={
      var newdf = df
      for (c<-cols){
        val onehotenc = new OneHotEncoder().setInputCol(c).setOutputCol("d_distance_onehot").setDropLast(false)
        newdf =onehotenc.transform(newdf)
      }
      newdf
    }
    val df_hot_ddistance =oneHotEncodeColumns_ddfender(df_hot_shotzone, Array("defender_distance_category"))

    // find centers for each zone

    val clusterCenters =df_hot_ddistance.groupBy("shot_zone").agg(avg($"x_half"),avg($"y_half")).toDF()
    val columnsRenamed = Seq("shot_zone","x_center","y_center")
    val clusterCenters_1 = clusterCenters.toDF(columnsRenamed: _*)


    //StandardScaler

    // In regression it is recommended that the input variables have a mean of 0. It’s easy to achieve by using the StandardScaler from Spark ML.

    val df_xyJoined = df_hot_ddistance.join(clusterCenters_1, Seq("shot_zone"))
      .withColumn("x_center_scaled",($"x_center"/47))
      .withColumn("y_center_scaled",(($"y_center"-25)/25))
      .withColumn("score_margin_scaled",(($"score_margin_category")/3))


    ///combine to a feature vector
    val assembler_1 = new VectorAssembler()
      // .setInputCols(Array("x_center","y_center","period_onehot","h","v","action_type_onehot","d_distance_onehot","score_margin_category"))
      .setInputCols(Array("x_center_scaled","y_center_scaled","d_distance_onehot","score_margin_scaled","period_onehot"))
      .setOutputCol("features")

    val data_all = assembler_1.transform(df_xyJoined)

    //Split the data into training and test sets (30% held out for testing)

    val Array(trainingData, testData) =data_all.randomSplit(Array(0.7, 0.3))


    // shooting percentage
    /// training data
    val df_grouped_train = trainingData.groupBy("features","shot_made_flag").
      agg(count("*").alias("shots")).orderBy("features","shot_made_flag")

    val df_made_miss_train =  trainingData.groupBy("features").agg(count("*").alias("total")).orderBy("features")

    // append the column "total attempts" to "df_grouped", drop duplicated columns.  Join key :"features"
    val df_joined_train = df_grouped_train.join(df_made_miss_train, Seq("features"))

    // Drop total attempts that are smaller than 1 times
    val df_joined_cut_train = df_joined_train.filter($"total" >= 2 )

    // Calulate FG% for each feature
    //Define a udf to concatenate two passed in string values
    val getPercentage = udf( (first: Float, second: Float) => "%.2f".format(first / second).toDouble  )

    //use withColumn method to add a new column called newColName
    val df_FG_train = df_joined_cut_train.withColumn("FG%", getPercentage($"shots", $"total")).select("features","shot_made_flag","shots","total","FG%")

    //add rows with all-miss situations

    // filter rows where all shots are missed
    val df_filtered_train = df_FG_train.filter($"shot_made_flag" === 0 && $"FG%" === 1 ).toDF("features", "shot_made_flag","shots","total","FG%")

    // create a dataframe "df_missall" that adds records of "made shots" FG equals to  0
    val df_missall_train = df_filtered_train.withColumn("shot_made_flag",lit(1)).withColumn("FG%",lit(0)).withColumn("shots",lit(0))


    // concatenate "df_missall" to "df_FG"
    val dfUnion_train =  df_FG_train.union(df_missall_train).toDF("features", "shot_made_flag","shots","total","FG%")

    val dfUnion_train_filtered = dfUnion_train.filter($"shot_made_flag" === 1).toDF





    // test data
    val df_grouped_test = testData.groupBy("features","shot_made_flag").
      agg(count("*").alias("shots")).orderBy("features","shot_made_flag")

    val df_made_miss_test =  testData.groupBy("features").agg(count("*").alias("total")).orderBy("features")

    // append the column "total attempts" to "df_grouped", drop duplicated columns.  Join key :"features"
    val df_joined_test = df_grouped_test.join(df_made_miss_test, Seq("features"))

    // Drop total attempts that are smaller than 1 times
    val df_joined_cut_test = df_joined_test.filter($"total" >= 2 )

    // Calulate FG% for each feature
    //Define a udf to concatenate two passed in string values
    //val getPercentage = udf( (first: Float, second: Float) => "%.2f".format(first / second).toDouble  )

    //use withColumn method to add a new column called newColName
    val df_FG_test = df_joined_cut_test.withColumn("FG%", getPercentage($"shots", $"total")).select("features","shot_made_flag","shots","total","FG%")

    //add rows with all-miss situations

    // filter rows where all shots are missed
    val df_filtered_test = df_FG_test.filter($"shot_made_flag" === 0 && $"FG%" === 1 ).toDF("features", "shot_made_flag","shots","total","FG%")

    // create a dataframe "df_missall" that adds records of "made shots" FG equals to  0
    val df_missall_test = df_filtered_test.withColumn("shot_made_flag",lit(1)).withColumn("FG%",lit(0)).withColumn("shots",lit(0))


    // concatenate "df_missall" to "df_FG"
    val dfUnion_test =  df_FG_test.union(df_missall_test).toDF("features", "shot_made_flag","shots","total","FG%")

    val dfUnion_test_filtered = dfUnion_test.filter($"shot_made_flag" === 1).toDF


    // Random forest

    // Automatically identify categorical features, and index them.
    // Set maxCategories so features with > 4 distinct values are treated as continuous.


/*
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(dfUnion_train_filtered)
    println("passed")

*/
    // Train a RandomForest model.
    val rf = new RandomForestRegressor()
      .setLabelCol("FG%")
      .setFeaturesCol("features")
      .setNumTrees(20)


    // Chain indexer and forest in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(rf))

    // Train model. This also runs the indexer.
    val model_RF = pipeline.fit(dfUnion_train_filtered)

    // Make predictions.
    val predictions = model_RF.transform(dfUnion_test_filtered)


    // Select example rows to display.
    predictions.select("prediction", "FG%", "features").show(5)

    val evaluator = new RegressionEvaluator()
      .setLabelCol("FG%")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val rmse = evaluator.evaluate(predictions)
    println("Root Mean Squared Error (RMSE) on test data = " + rmse)

    val rfModel = model_RF.stages(0).asInstanceOf[RandomForestRegressionModel]
    //println("Learned regression forest model:\n" + rfModel.toDebugString)
    //rfModel.write.overwrite().save("/Users/liushengchen/course/Scala/final_project/loadData/rfModel")
    rfModel.write.overwrite().save("rfModel_1")



  }

}
