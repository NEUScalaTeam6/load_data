package load_data

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.commons.io.IOUtils
import java.net.URL
import java.nio.charset.Charset

import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types._
import org.apache.spark.sql.Row
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, MaxAbsScaler, OneHotEncoder, StringIndexer, StringIndexerModel, VectorAssembler, VectorIndexer}
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.ml.evaluation.RegressionEvaluator


/**
  * Created by liushengchen on 4/24/17.
  */
object loadData_ml {

 def  main(args: Array[String]) {

    // Create a SparkSession.
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("Spark SQL basic example")
      .config("spark.some.config.option", "some-value")
      .getOrCreate()


    import spark.implicits._


    val file = "/Users/liushengchen/Google_Drive/Scala_group/Project/Data/CurryFinalData.csv"
    val shotData = spark.read
      .format("com.databricks.spark.csv")
      .option("header", "true") // Use first line of all files as header
      .option("inferSchema", "true") // Automatically infer data types
      .load(file)

    val df_raw = shotData.select("_c0", "period", "minutes_remaining", "seconds_remaining", "action_type", "loc_x", "loc_y",
      "defender_id", "defender_distance", "htm", "vtm", "SCOREMARGIN", "shot_made_flag")

    df_raw.show(1)
    df_raw.printSchema

    // Drop "Hail Marry" shot: shot "y" location larger thant 47 feet (half court)
    val df_raw_1 = df_raw.filter($"loc_y" < 47 ).toDF

    /*
    2.Score margin
    "score margin" means the score difference between the home team and the visit team.
    In raw data, when the score margin is '0', the value is shown as 'TIE'.
    We need to filter out this value and convert it into '0'.
    */

    val df_raw_score = df_raw_1.withColumn("score_margin", when(col("SCOREMARGIN").equalTo("TIE"),0).otherwise(col("SCOREMARGIN")))

    // 2.2 Categorize score margins into several groups: score_margin_category = score_margin / 5
    val toInt  = udf[Int, Double]( _.toInt)
    val df_divide = df_raw_score.withColumn("score_margin_category",col("score_margin")/5)
    val df_score_category =df_divide.withColumn("score_margin_category",toInt(df_divide("score_margin_category")))
      .withColumn("defender_distance_category",toInt(df_divide("defender_distance")))

    // Consider score margin larger than 15 as a same group.
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
    5. k-means of loc_x, loc_y

     */

    //5.1 Combine “loc_x” and “loc_y” into “feature_xy”

    val assembler = new VectorAssembler()
      .setInputCols(Array("loc_x", "loc_y"))
      .setOutputCol("feature_xy")

    val data_assembled = assembler.transform(data_select)
    val data_featurexy= data_assembled.select("feature_xy", "period","action_type","h","v","defender_id","defender_distance_category","score_margin_category","shot_made_flag")

    //5.2 Compute Cost function to find the optimal k values
    // Trains a k-means model

    val k_num = 20
    val kmeans = new KMeans()
      .setK(k_num)                              // set number of clusters
      .setFeaturesCol("feature_xy")
      .setPredictionCol("shot_zone")
    val kMeansModel = kmeans.fit(data_featurexy)

    val transformed = kMeansModel.transform(data_featurexy)

    // 5.3 cluster centers
    val clusterCenters =  kMeansModel.clusterCenters map (_.toArray)
    //println("The Cluster Centers are = " + clusterCenters)

    val d=spark.sparkContext.parallelize(clusterCenters).toDF("clusterCenters")
    //Extract schema for further usage
    val schema = d.schema
    // Add id field
    val rows = d.rdd.zipWithIndex.map{
      case (r: Row, id: Long) => Row.fromSeq(id +: r.toSeq)}

    val dfWithID = spark.sqlContext.createDataFrame(
      rows, StructType(StructField("shot_zone", LongType, false) +: schema.fields))

    // convert Array[Double] to Vector
    val seqAsVector = udf((xs: Seq[Double]) => Vectors.dense(xs.toArray))

    val df_cluster = dfWithID.withColumn("clusterCenters", seqAsVector(col("clusterCenters")))

    // Join centers to the dataframe
    val df_xyJoined =  transformed.join(df_cluster, Seq("shot_zone"))
    df_xyJoined.show(1)

    /*
    6. Dealing with categorical values

        StringIndexer helps you convert String categorical values into integer indexes of those values.
     */

    //6.1 Action type

    // shows distinct "action types"
    val distinctValuesDF = df_xyJoined.select(df_xyJoined("action_type")).distinct
    val distinctValuesArray = distinctValuesDF.select("action_type").rdd.map(r => r(0).asInstanceOf[String]).collect()

    //Because "action type" has 35 different values, which is too many. We merge those values into 8 categories:

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
    val transformed_category = df_xyJoined.withColumn("action_type_category", udfActionToCategory(df_xyJoined("action_type")))

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

    val dfnumeric = indexStringColumns(transformed_category, Array("action_type_category"))

    //Encoding the "shot_zone" with OneHotEncoder
    def oneHotEncodeColumns(df:DataFrame, cols:Array[String]):DataFrame ={
      var newdf = df
      for (c<-cols){
        val onehotenc = new OneHotEncoder().setInputCol(c).setOutputCol("action_type_onehot").setDropLast(false)
        newdf =onehotenc.transform(newdf)
      }
      newdf
    }
    val dfhot = oneHotEncodeColumns(dfnumeric, Array("action_type_category"))


    // 6.2 period
    //Encoding the "period" with OneHotEncoder
    def oneHotEncodeColumns_period(df:DataFrame, cols:Array[String]):DataFrame ={
      var newdf = df
      for (c<-cols){
        val onehotenc = new OneHotEncoder().setInputCol(c).setOutputCol("period_onehot").setDropLast(false)
        newdf =onehotenc.transform(newdf)
      }
      newdf
    }
    val df_hot_period = oneHotEncodeColumns_period(dfhot, Array("period"))


    // 6.3 shot_zone

    //Encoding the "shot_zone" with OneHotEncoder
    def oneHotEncodeColumns_shotzone(df:DataFrame, cols:Array[String]):DataFrame ={
      var newdf = df
      for (c<-cols){
        val onehotenc = new OneHotEncoder().setInputCol(c).setOutputCol("shot_zone_onehot").setDropLast(false)
        newdf =onehotenc.transform(newdf)
      }
      newdf
    }
    val df_hot_shotzone = oneHotEncodeColumns_shotzone(df_hot_period, Array("shot_zone"))


    //6.4 defender_distance
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


    /*
    7. Construct "features" vector
     */
    val vectorFirst = udf{ x:DenseVector => "%.2f".format(x(0)).toDouble }
    val vectorSecond = udf{ x:DenseVector =>"%.2f".format(x(1)).toDouble }
    val df_1 =df_hot_ddistance.withColumn( "x_loc",vectorFirst(df_hot_ddistance("feature_xy")))
      .withColumn("y_loc",vectorSecond(df_hot_ddistance("feature_xy")))
      .withColumn("x_center",vectorFirst(df_hot_ddistance("clusterCenters")))
      .withColumn("y_center",vectorSecond(df_hot_ddistance("clusterCenters")))


    val assembler_1 = new VectorAssembler()
      // .setInputCols(Array("x_center","y_center","period_onehot","h","v","action_type_onehot","d_distance_onehot","score_margin_category"))
      .setInputCols(Array("x_center","y_center","d_distance_onehot","score_margin_category","period_onehot"))
      .setOutputCol("features")

    val output_1 = assembler_1.transform(df_1)


    /*
    8. Caluculate shooting percentage
     */

    val df_grouped = output_1.groupBy("features","shot_made_flag").
      agg(count("*").alias("shots")).orderBy("features","shot_made_flag")

    val df_made_miss = output_1.groupBy("features").agg(count("*").alias("total")).orderBy("features")

    // append the column "total attempts" to "df_grouped", drop duplicated columns.  Join key :"features"
    val df_joined = df_grouped.join(df_made_miss, Seq("features"))

    // Drop total attempts that are smaller than 1 times
    val df_joined_cut = df_joined.filter($"total" >= 2 )

    // Calulate FG% for each feature
    //Define a udf to concatenate two passed in string values
    val getPercentage = udf( (first: Float, second: Float) => "%.2f".format(first / second).toDouble  )

    //use withColumn method to add a new column called newColName
    val df_FG = df_joined_cut.withColumn("FG%", getPercentage($"shots", $"total")).select("features","shot_made_flag"
      ,"shots","total","FG%")

    //add rows with all-miss situations

    // filter rows where all shots are missed
    val df_filtered = df_FG.filter($"shot_made_flag" === 0 && $"FG%" === 1 ).toDF("features", "shot_made_flag","shots"
      ,"total","FG%")

    // create a dataframe "df_missall" that adds records of "made shots" FG equals to  0
    val df_missall = df_filtered.withColumn("shot_made_flag",lit(1)).withColumn("FG%",lit(0)).withColumn("shots",lit(0))

    // concatenate "df_missall" to "df_FG"
    val dfUnion =  df_FG.union(df_missall).toDF("features", "shot_made_flag","shots","total","FG%")



    /*
    9. MaxAbsScaler

      In regression it is recommended that the input variables are rescaled each feature to range [-1, 1]
      It’s easy to achieve by using the MaxAbsScaler from Spark ML.
     */

    val scaler = new MaxAbsScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")

    // Compute summary statistics and generate MaxAbsScalerModel
    val scalerModel = scaler.fit(dfUnion)

    // rescale each feature to range [-1, 1]
    val scaledData = scalerModel.transform(dfUnion)
    scaledData.select("features", "scaledFeatures").show(2,false)

    println("Total records:", scaledData.count())

    // Split the data into training and test sets (30% held out for testing).
    val Array(trainingData, testData) = scaledData.randomSplit(Array(0.7, 0.3))


    /*
    10. Random forest regression
     */

    // Automatically identify categorical features, and index them.
    // Set maxCategories so features with > 4 distinct values are treated as continuous.
    val featureIndexer = new VectorIndexer()
      .setInputCol("scaledFeatures")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(scaledData)


    // Train a RandomForest model.
    val rf = new RandomForestRegressor()
      .setLabelCol("FG%")
      .setFeaturesCol("indexedFeatures")
      .setNumTrees(20)


    // Chain indexer and forest in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(featureIndexer,rf))

    // Train model. This also runs the indexer.
    val model_RF = pipeline.fit(trainingData)

    // Make predictions.
    val predictions = model_RF.transform(testData)

    // Select example rows to display.
    predictions.select("prediction", "FG%", "features").show(5)

    val evaluator = new RegressionEvaluator()
      .setLabelCol("FG%")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val rmse = evaluator.evaluate(predictions)
    println("Root Mean Squared Error (RMSE) on test data = " + rmse)

    val rfModel = model_RF.stages(1).asInstanceOf[RandomForestRegressionModel]


    // save model
    rfModel.write.overwrite().save("rfModel")
    model_RF.write.overwrite().save("rfPipeline")
    //model_RF.write.overwrite().save("/Users/liushengchen/course/Scala/final_project/loadData/ml_model/workflow")

    val sameModel =RandomForestRegressionModel.load("rfModel")
    val samePipeline =PipelineModel.load("rfPipeline")



    //"x_center","y_center","d_distance_onehot","score_margin_category","period_onehot"
    val x = 0.5
    val y = 0.6
    val d_distance = 0
    val score_margin = 2
    val period = 1

    // Convert 1-based indices to 0-based.
    val d_distance_onehot = Vectors.sparse(7, Array(d_distance), Array(1))
    val period_onehot = Vectors.sparse(6, Array(period - 1), Array(1))

    val df_prediction = spark.createDataFrame(Seq(
      (x, y, d_distance_onehot, score_margin, period_onehot)
    )).toDF("x", "y", "d_distance_onehot", "score_margin", "period_onehot")

    //df_prediction.show

    val assembler_load = new VectorAssembler()
      // .setInputCols(Array("x_center","y_center","period_onehot","h","v","action_type_onehot","d_distance_onehot","score_margin_category"))
      .setInputCols(Array("x", "y", "d_distance_onehot", "score_margin", "period_onehot"))
      .setOutputCol("indexedFeatures")

    val output_p = assembler_load.transform(df_prediction)

    val predictions_test = sameModel.transform(output_p)
    predictions_test.show()


  }
}
