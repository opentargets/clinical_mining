from loguru import logger
import polars as pl


def construct_db_uri(
    db_type: str, db_uri: str, db_user: str | None, db_password: str | None
):
    """Constructs a database URI from the given parameters."""
    if db_user and db_password:
        return f"{db_type}://{db_user}:{db_password}@{db_uri}"
    return f"{db_type}://{db_uri}"


def _build_select_query(
    table_name: str,
    db_schema: str,
    select_cols: list[str] | str = "*",
    limit: int | None = None,
    dialect: str = "generic",
) -> str:
    """Form select query to extract data from a table.

    Args:
        table_name: Table to read.
        db_schema: Schema name (will be used as <schema>.<table>).
        select_cols: List of columns or a raw 
        limit: Optional row limit.
        dialect: "generic" uses SQL LIMIT. "oracle" uses FETCH FIRST.

    Returns:
        SQL query string.
    """
    if isinstance(select_cols, str):
        fields = select_cols
    elif isinstance(select_cols, list):
        fields = ", ".join(select_cols)

    qualified = f"{db_schema}.{table_name}" if db_schema else table_name
    query = f"SELECT DISTINCT {fields} FROM {qualified}"

    if limit is not None:
        if dialect.lower() == "oracle":
            query += f" FETCH FIRST {limit} ROWS ONLY"
        else:
            query += f" LIMIT {limit}"

    return query


def load_db_table(
    table_name: str,
    db_url: str,
    db_schema: str,
    select_cols: list[str] | str = "*",
    limit: int | None = None,
) -> pl.DataFrame:
    """Connects to a db and returns the query results as a Polars DataFrame."""
    dialect = "oracle" if "ora" in db_url else "generic"
    query = _build_select_query(
        table_name=table_name,
        db_schema=db_schema,
        select_cols=select_cols,
        limit=limit,
        dialect=dialect,
    )
    return pl.read_database_uri(query=query, uri=db_url)


def print_table_schema(table_name: str, db_uri: str, db_schema: str) -> None:
    """Get schema for a table"""
    logger.info(load_db_table(table_name, db_uri, db_schema, limit=1).schema)


# --- Oracle utilities -------------------------------------------------------

def _init_oracle_client(lib_dir: str | None = None) -> None:
    """Initialize the Oracle client once if a lib_dir is provided.

    Safe to call multiple times; cx_Oracle will ignore subsequent inits.
    """
    if lib_dir:
        import cx_Oracle

        try:
            cx_Oracle.init_oracle_client(lib_dir=lib_dir)
        except Exception:
            pass


def load_oracle_query(
    user: str,
    password: str,
    host: str,
    port: int,
    service: str,
    query: str,
    init_client_lib_dir: str | None = None,
) -> pl.DataFrame:
    """Execute a SQL query against Oracle and return a Polars DataFrame.

    Uses cx_Oracle connection with Polars' read_database().
    """
    import cx_Oracle

    # Create connection
    _init_oracle_client(init_client_lib_dir)
    dsn = cx_Oracle.makedsn(host, port, service_name=service)
    connection = cx_Oracle.connect(user=user, password=password, dsn=dsn)

    try:
        return pl.read_database(query, connection)
    finally:
        try:
            connection.close()
        except Exception:
            pass


def load_oracle_table(
    table_name: str,
    user: str,
    password: str,
    host: str,
    port: int,
    service: str,
    db_schema: str = "",
    select_cols: list[str] | str = "*",
    limit: int | None = None,
    init_client_lib_dir: str | None = None,
) -> pl.DataFrame:
    """Execute a query in an Oracle table and return a Polars DataFrame.

    Example:
        load_oracle_table(
            table_name="CT_NCTID_CONDITION_EFO",
            user="opentargets",
            password="...",
            host="ora-vm-089",
            port=1531,
            service="CHEMPRO",
            db_schema="DRUGBASE_CURATION",
            limit=5,
            init_client_lib_dir="/opt/oracle/instantclient_23_3",
        )
    """
    query = _build_select_query(
        table_name=table_name,
        db_schema=db_schema,
        select_cols=select_cols,
        limit=limit,
        dialect="oracle",
    )
    import cx_Oracle

    # Create connection
    _init_oracle_client(init_client_lib_dir)
    dsn = cx_Oracle.makedsn(host, port, service_name=service)
    connection = cx_Oracle.connect(user=user, password=password, dsn=dsn)

    try:
        return pl.read_database(query, connection)
    finally:
        try:
            connection.close()
        except Exception:
            pass