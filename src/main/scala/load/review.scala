package load

import org.apache.spark.sql.SparkSession

/**
  * Created by liushengchen on 4/28/17.
  */
object review extends App{

  //def main(args: Array[String]) {

    // Create a SparkSession.
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("Spark SQL basic example")
      .config("spark.some.config.option", "some-value")
      .getOrCreate()


    import org.apache.spark.ml.feature.VectorAssembler
    import org.apache.spark.ml.linalg.Vectors
    import org.apache.spark.ml.regression.RandomForestRegressionModel

    // "x_center_scaled", "y_center_scaled","d_distance_onehot","score_margin_scaled","period_onehot
    val x = 0.4  //  [0,1]
    val y = -0.0534 // [-1,1]
    val d_distance = 0 //[0,6] onehot
    val score_margin = 2/3 // [-3,3]
    val period = 1 // [1,5] onehot

    // Convert 1-based indices to 0-based.
    val d_distance_onehot = Vectors.sparse(7, Array(d_distance), Array(1))
    val period_onehot = Vectors.sparse(6, Array(period - 1), Array(1))

    val df_prediction = spark.createDataFrame(Seq(
      (x, y, d_distance_onehot, score_margin, period_onehot)
    )).toDF("x", "y", "d_distance_onehot", "score_margin", "period_onehot")

    //df_prediction.show

    val assembler = new VectorAssembler()
      // .setInputCols(Array("x_center","y_center","period_onehot","h","v","action_type_onehot","d_distance_onehot","score_margin_category"))
      .setInputCols(Array("x", "y", "d_distance_onehot", "score_margin", "period_onehot"))
      .setOutputCol("features")

    val output_1 = assembler.transform(df_prediction)


    //val model_GBT = PipelineModel.load("workflow")

    //val sameModel =GBTRegressionModel.load("gbtModel")

    val sameModel =RandomForestRegressionModel.read.load("rfModel_1")
    val predictions_test = sameModel.transform(output_1)
    val prediction = predictions_test.select("prediction").head().getDouble(0)
    println( "%.1f".format(prediction *100 ) + "%")
    predictions_test.show


    def FG:Double = prediction

}