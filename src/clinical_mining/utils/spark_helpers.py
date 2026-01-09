from pyspark.sql import SparkSession
from pyspark.conf import SparkConf

def spark_session():
    """OnToma works on Spark dataframes."""
    try:
        SparkSession.getActiveSession().stop()
    except Exception:
        pass
    
    config = (
        SparkConf()
        .set("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:5.0.0")
        .set("spark.driver.memory", "10g")
        .set("spark.driver.maxResultSize", "4g")
        .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .set("spark.kryoserializer.buffer.max", "512m")
        .set("spark.sql.shuffle.partitions", "50")
        .set("spark.default.parallelism", "4")
        .set("spark.sql.adaptive.enabled", "true")
        .set("spark.ui.enabled", "false")
        .set("spark.driver.host", "localhost")
        .set("spark.driver.bindAddress", "127.0.0.1")
    )
    return (
        SparkSession.builder
        .appName("clinical_mining_entity_mapping")
        .master("local[4]")
        .config(conf=config)
        .getOrCreate()
    )