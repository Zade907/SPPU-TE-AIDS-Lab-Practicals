import org.apache.spark.sql.SparkSession

object WordCountApp {
  def main(args: Array[String]): Unit = {

    // Step 1: Create Spark Session
    val spark = SparkSession.builder()
      .appName("Word Count Example")
      .master("local[*]")   // runs locally
      .getOrCreate()

    // Step 2: Read text file
    val textFile = spark.read.textFile("input.txt")

    // Step 3: Split words and count
    val wordCounts = textFile
      .flatMap(line => line.split(" "))
      .groupBy(word => word)
      .count()

    // Step 4: Show output
    wordCounts.collect().foreach(println)

    // Stop Spark
    spark.stop()
  }
}