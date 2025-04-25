from psutil import virtual_memory
from pyspark.sql import SparkSession as PySparkSession

def detect_spark_memory_limit() -> int:
    """Detects available memory (in GiB) and reserves 90% for Spark."""
    mem_gib = virtual_memory().total >> 30
    return int(mem_gib * 0.9)

class SparkSession:
    """
    Manages a Spark session with sensible memory defaults and user config."""
    def __init__(self, config: dict[str, str] = {}):
        spark_mem_limit = detect_spark_memory_limit()
        default_config = {
            "spark.driver.memory": f"{spark_mem_limit}g",
            "spark.executor.memory": f"{spark_mem_limit}g",
        }
        merged_config = {**default_config, **config}
        builder = PySparkSession.builder.appName("Clinical Mining")
        for k, v in merged_config.items():
            builder = builder.config(k, v)
        self._session = builder.getOrCreate()

    @property
    def session(self) -> PySparkSession:
        return self._session

    def stop(self):
        self._session.stop()