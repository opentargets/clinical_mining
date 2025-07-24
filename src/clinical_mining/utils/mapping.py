import polars as pl


def assign_drug_id(
    df: pl.DataFrame, molecule: pl.DataFrame, verbose: bool = True
) -> pl.DataFrame:
    """Assigns drug IDs by joining with a molecule mapping table."""
    df_w_id = (
        df.with_columns(pl.col("drug_name").str.to_lowercase())
        .join(
            molecule.explode("drug_names").rename({"drug_names": "drug_name"}),
            on="drug_name",
            how="left",
        )
        .with_columns(drug_id=pl.coalesce(["drug_id", "mapped_drug_id"]))
        .drop("mapped_drug_id")
    )
    if verbose:
        unmapped_count = df_w_id.filter(pl.col("drug_id").is_null()).height
        mapped_count = df_w_id.filter(pl.col("drug_id").is_not_null()).height
        print(f"Unmapped drug rows: {unmapped_count}")
        print(f"Mapped drug rows: {mapped_count}")
    return df_w_id


def assign_disease_id(
    df: pl.DataFrame, diseases: pl.DataFrame, verbose: bool = True
) -> pl.DataFrame:
    """Assigns disease IDs by joining with a disease mapping table."""
    df_w_id = (
        df.with_columns(pl.col("disease_name").str.to_lowercase())
        .join(
            diseases.explode("disease_names").rename({"disease_names": "disease_name"}),
            on="disease_name",
            how="left",
        )
        .with_columns(disease_id=pl.coalesce(["disease_id", "mapped_disease_id"]))
        .drop("mapped_disease_id")
    )
    if verbose:
        unmapped_count = df_w_id.filter(pl.col("disease_id").is_null()).height
        mapped_count = df_w_id.filter(pl.col("disease_id").is_not_null()).height
        print(f"Unmapped disease rows: {unmapped_count}")
        print(f"Mapped disease rows: {mapped_count}")
    return df_w_id


def process_molecule(path: str) -> pl.DataFrame:
    """Loads and processes molecule data from a Parquet file."""
    return (
        pl.read_parquet(path)
        .select(
            pl.col("id").alias("mapped_drug_id"),
            pl.col("name").str.to_lowercase(),
            pl.col("synonyms").list.eval(pl.element().str.to_lowercase()),
        )
        .with_columns(
            drug_names=pl.concat_list(
                [pl.col("synonyms"), pl.col("name")]
            ).list.unique()
        )
        .select("mapped_drug_id", "drug_names")
    )


def process_disease(path: str) -> pl.DataFrame:
    """Loads and processes disease data from a Parquet file."""
    return (
        pl.read_parquet(path)
        .select(
            pl.col("id").alias("mapped_disease_id"),
            pl.col("name").str.to_lowercase(),
            pl.col("synonyms").struct.field("hasExactSynonym").alias("synonyms"),
        )
        .with_columns(
            synonyms=pl.col("synonyms").list.eval(pl.element().str.to_lowercase())
        )
        .with_columns(
            disease_names=pl.concat_list(
                [pl.col("synonyms"), pl.col("name")]
            ).list.unique()
        )
        .select("mapped_disease_id", "disease_names")
        .unique()
    )
