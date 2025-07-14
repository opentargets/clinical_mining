from psutil import virtual_memory
from pyspark.sql import DataFrame, SparkSession as PySparkSession

from functools import reduce


def detect_spark_memory_limit() -> int:
    """Detects available memory (in GiB) and reserves 90% for Spark."""
    mem_gib = virtual_memory().total >> 30
    return int(mem_gib * 0.9)


class SparkSession:
    """Manages a Spark session with sensible memory defaults and user config, with an optional PostgreSQL connection."""

    def __init__(
        self,
        db_url: str | None = None,
        user: str | None = None,
        password: str | None = None,
        schema: str | None = None,
        config: dict[str, str] = {},
    ):
        spark_mem_limit = detect_spark_memory_limit()
        default_config = {
            "spark.driver.memory": f"{spark_mem_limit}g",
            "spark.executor.memory": f"{spark_mem_limit}g",
        }
        if db_url and user and password:
            default_config["spark.jars.packages"] = "org.postgresql:postgresql:42.6.0"
            self.jdbc_url = f"jdbc:postgresql://{db_url}"
            self.connection_properties = {
                "user": user,
                "password": password,
                "driver": "org.postgresql.Driver",
                "ssl": "true",
                "sslmode": "require",
            }
            self.schema = schema

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

    def load_table(
        self,
        table_name: str,
        limit: int | None = None,
        select_cols: list[str] | str | None = None,
    ) -> DataFrame:
        """
        Load a table from a database.

        Args:
            table_name: Name of the table to load
            limit: Optional row limit for testing

        Returns:
            A Spark DataFrame containing the table data
        """
        if not self.jdbc_url or not self.connection_properties or not self.schema:
            raise ConnectionError(
                "Database connection not configured. Please provide db_url, user, and password during SparkSession initialization."
            )

        full_table_name = f"{self.schema}.{table_name}"

        if select_cols is not None:
            select_cols = (
                ", ".join(select_cols) if isinstance(select_cols, list) else select_cols
            )
            query = f"SELECT {select_cols} FROM {full_table_name}"
        else:
            query = f"SELECT * FROM {full_table_name}"
        if limit is not None:
            query += f" LIMIT {limit}"
        subquery = f"({query}) as t"
        return self.session.read.jdbc(
            url=self.jdbc_url, table=subquery, properties=self.connection_properties
        ).distinct()

    def print_table_schema(self, table_name: str) -> None:
        """Get schema for a table"""
        limited_df = self.load_table(table_name, limit=1)
        limited_df.printSchema()


def join_dfs(
    dfs: list[DataFrame],
    join_on: str = "nct_id",
    how: str = "left",
) -> DataFrame:
    """Join a list of DataFrames on a common column.

    Args:
        dfs: List of DataFrames to join
        join_on: Column to join on
        how: Type of join to use
    Returns:
        DataFrame: The joined DataFrame
    """
    return reduce(lambda df1, df2: df1.join(df2, on=join_on, how=how), dfs)


def union_dfs(
    dfs: list[DataFrame],
) -> DataFrame:
    """Union a list of DataFrames.

    Args:
        dfs: List of DataFrames to union
    Returns:
        DataFrame: The unioned DataFrame
    """
    return reduce(lambda df1, df2: df1.unionByName(df2, allowMissingColumns=True), dfs)
