from pyspark.sql import SparkSession
from pyspark.conf import SparkConf

def spark_session():
    """OnToma works on Spark dataframes."""
    # Stop any existing Spark sessions to avoid conflicts
    try:
        SparkSession.getActiveSession().stop()
    except Exception:
        pass
    config = (
        SparkConf()
        .set("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:5.0.0")
        .set("spark.driver.memory", "16g")
        .set("spark.executor.memory", "16g")
        .set("spark.driver.maxResultSize", "8g")
        .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .set("spark.kryoserializer.buffer.max", "1g")
        .set("spark.sql.adaptive.enabled", "true")
        .set("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .set("spark.sql.adaptive.advisoryPartitionSizeInBytes", "64MB")
        .set("spark.sql.shuffle.partitions", "400")
        .set("spark.default.parallelism", "8")
        .set("spark.sql.execution.arrow.pyspark.enabled", "true")
        .set("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
        .set("spark.ui.enabled", "false")
        .set("spark.driver.host", "localhost")
        .set("spark.driver.bindAddress", "localhost")
    )
    return (
        SparkSession.builder.appName("clinical_mining_entity_mapping")
        .master("local[*]")
        .config(conf=config)
        .getOrCreate()
    )

