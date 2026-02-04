"""Polars-specific helper functions for the pipeline."""

import polars as pl
from pyspark.sql import SparkSession, DataFrame


def convert_polars_to_spark(
    polars_df: pl.DataFrame, spark: SparkSession, chunk_size: int = 100000
) -> DataFrame:
    """
    Convert large Polars DataFrame to Spark DataFrame using optimised chunking to avoid serialization and memory issues.

    Args:
        polars_df: Input Polars DataFrame
        spark: Active SparkSession
        chunk_size: Number of rows per chunk (100k works best for 1M+ row DataFrames)

    Returns:
        Spark DataFrame with optimized partitioning
    """
    total_rows = len(polars_df)

    if total_rows <= chunk_size:
        # For small DataFrames, use direct conversion
        return spark.createDataFrame(polars_df.to_pandas())

    # Process in chunks for large DataFrames
    spark_chunks = []

    for i in range(0, total_rows, chunk_size):
        chunk = polars_df.slice(i, chunk_size)
        chunk_spark = spark.createDataFrame(chunk.to_pandas())
        spark_chunks.append(chunk_spark)

    # Union all chunks into single DataFrame
    result_df = spark_chunks[0]
    for chunk_df in spark_chunks[1:]:
        result_df = result_df.union(chunk_df)

    # Optimize final partitioning to prevent large tasks
    optimal_partitions = min(
        max(total_rows // 50000, 8), 400
    )  # Between 8-400 partitions
    return result_df.repartition(optimal_partitions)


def union_dfs(dfs: list[pl.DataFrame]) -> pl.DataFrame:
    """Concatenates a list of Polars DataFrames, filling missing column values with null.

    Args:
        dfs (List[pl.DataFrame]): A list of Polars DataFrames to concatenate.

    Returns:
        pl.DataFrame: A single concatenated DataFrame.
    """
    return pl.concat(dfs, how="diagonal")


def join_dfs(dfs: list[pl.DataFrame], join_on: str, how: str = "inner") -> pl.DataFrame:
    """Joins a list of Polars DataFrames on a common key.

    Args:
        dfs (List[pl.DataFrame]): A list of Polars DataFrames to join.
        join_on (str): The column name to join on.
        how (str, optional): The type of join. Defaults to "inner".

    Returns:
        pl.DataFrame: The joined DataFrame.
    """
    if not dfs:
        return pl.DataFrame()

    joined_df = dfs[0]
    for i in range(1, len(dfs)):
        joined_df = joined_df.join(dfs[i], on=join_on, how=how, coalesce=True)

    return joined_df


def coalesce_column(
    df: pl.DataFrame, output_column_name: str, input_column_names: list[str], drop: bool = False
) -> pl.DataFrame:
    """Safely coalesces multiple columns into a single column, filling missing values with null. Note that order of input columns is important.
    
    Raises:
        ValueError: If none of the input_column_names exist in the DataFrame
    """
    existing_columns = [col for col in input_column_names if col in df.columns]
    
    # Raise error if none of the columns exist
    if not existing_columns:
        raise ValueError(
            f"None of the input columns {input_column_names} exist in the DataFrame. "
            f"Available columns: {df.columns}"
        )
    
    # Coalesce only the existing columns
    df = df.with_columns(
        pl.coalesce(*existing_columns).alias(output_column_name)
    )
    return df.drop(existing_columns) if drop else df

def filter_df(df: pl.DataFrame, expr: str | pl.Expr) -> pl.DataFrame:
    """Filters a Polars DataFrame based on a condition.
    
    Args:
        df (pl.DataFrame): The DataFrame to filter.
        expr (str | pl.Expr): The condition to filter by. Can be a string expression
    
    Returns:
        pl.DataFrame: The filtered DataFrame.
    """
    if isinstance(expr, pl.Expr):
        return df.filter(expr)

    ctx = pl.SQLContext()
    ctx.register("df", df)

    return ctx.execute(f"SELECT * FROM df WHERE {expr}").collect()


