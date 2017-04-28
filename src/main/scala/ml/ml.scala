package ml

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
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer,VectorAssembler,StringIndexerModel,
OneHotEncoder,MaxAbsScaler}
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.ml.evaluation.RegressionEvaluator

/**
  * Created by liushengchen on 4/24/17.
  */
object ml {

  def main(args: Array[String]): Unit = {

    // Create a SparkSession.
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("Spark SQL basic example")
      .config("spark.some.config.option", "some-value")
      .getOrCreate()


    import spark.implicits._

    val file = "data/dfUnion/dfUnion.csv"
    val dfUnion= spark.read
      .format("com.databricks.spark.csv")
      .option("header", "true") // Use first line of all files as header
      .option("inferSchema", "true") // Automatically infer data types
      .load(file)

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


    // Chain indexer and forest in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(featureIndexer, rf))

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
    rfModel.write.overwrite().save("/model/rfModel")



  }

}
