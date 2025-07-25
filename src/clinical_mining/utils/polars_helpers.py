"""Polars-specific helper functions for the pipeline."""

import polars as pl


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
        joined_df = joined_df.join(dfs[i], on=join_on, how=how)

    return joined_df
