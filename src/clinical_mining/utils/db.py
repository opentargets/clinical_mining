import polars as pl


def construct_db_url(
    db_type: str, db_uri: str, db_user: str | None, db_password: str | None
):
    """Constructs a database URI from the given parameters."""
    if db_user and db_password:
        return f"{db_type}://{db_user}:{db_password}@{db_uri}"
    return f"{db_type}://{db_uri}"


def load_db_table(
    table_name: str,
    db_url: str,
    db_schema: str,
    select_cols: list[str] | str | None = None,
    limit: int | None = None,
) -> pl.DataFrame:
    """Connects to a db and returns the query results as a Polars DataFrame."""
    if select_cols is not None:
        select_cols = (
            ", ".join(select_cols) if isinstance(select_cols, list) else select_cols
        )
        query = f"SELECT {select_cols} FROM {db_schema}.{table_name}"
    else:
        query = f"SELECT * FROM {db_schema}.{table_name}"
    if limit is not None:
        query += f" LIMIT {limit}"
    return pl.read_database_uri(query=query, uri=db_url)


def print_table_schema(table_name: str, db_uri: str, db_schema) -> None:
    """Get schema for a table"""
    print(load_db_table(table_name, db_uri, db_schema, limit=1).schema)
