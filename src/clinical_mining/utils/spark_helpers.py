from pyspark.sql import SparkSession
import sparknlp
from ontoma.ner._pipelines import get_device


def spark_session() -> SparkSession:
    """OnToma works on Spark dataframes."""
    try:
        SparkSession.getActiveSession().stop()
    except Exception:
        pass
    params = {
        "spark.driver.memory":"10g",
        "spark.driver.maxResultSize":"4g",
        "spark.serializer":"org.apache.spark.serializer.KryoSerializer",
        "spark.kryoserializer.buffer.max":"512m",
        "spark.sql.shuffle.partitions": "50",
        "spark.default.parallelism": "4",
        "spark.sql.adaptive.enabled": "true",
        "spark.ui.enabled": "false",
    }
    is_apple_silicon = get_device() == "mps"
    return sparknlp.start(params=params, apple_silicon=is_apple_silicon)