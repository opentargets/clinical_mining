from pyspark.sql import DataFrame

from clinical_mining.utils.spark import SparkSession


class AACTConnector:
    """A connection manager for AACT database."""

    def __init__(self, db_url: str, user: str, password: str, schema: str = "ctgov"):
        """
        Initialize the connection manager.

        Args:
            db_url: Database host and port (e.g., "aact-db.ctti-clinicaltrials.org:5432/aact")
            user: Database username
            password: Database password
            schema: Database schema (defaults to "ctgov")
        """
        self.jdbc_url = f"jdbc:postgresql://{db_url}"
        self.connection_properties = {
            "user": user,
            "password": password,
            "driver": "org.postgresql.Driver",
            "ssl": "true",
            "sslmode": "require",
        }
        self.schema = schema
        self.spark = SparkSession(
            {"spark.jars.packages": "org.postgresql:postgresql:42.6.0"}
        )

    def load_table(
        self,
        table_name: str,
        limit: int | None = None,
        select_cols: list[str] | str | None = None,
    ) -> DataFrame:
        """
        Load a table from AACT database.

        Args:
            table_name: Name of the table to load
            limit: Optional row limit for testing

        Returns:
            A Spark DataFrame containing the table data
        """
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
        return self.spark.session.read.jdbc(
            url=self.jdbc_url, table=subquery, properties=self.connection_properties
        ).distinct()

    def print_table_schema(self, table_name: str) -> None:
        """Get schema for a table"""
        limited_df = self.load_table(table_name, limit=1)
        limited_df.printSchema()
