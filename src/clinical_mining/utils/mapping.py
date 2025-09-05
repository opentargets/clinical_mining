import polars as pl


def normalise_label(label: pl.Series) -> pl.Series:
    return label.str.to_lowercase().str.replace(r"[^a-z0-9]", "")


def assign_entity_id(
    df: pl.DataFrame,
    mapping_index: pl.DataFrame,
    chembl_curation: pl.DataFrame,
    entity_type: str,
    verbose: bool = True,
) -> pl.DataFrame:
    """Assigns entity IDs (drug or disease) by joining with a mapping table and ChEMBL's curation data."""
    # TODO: Refactor to use ChEMBL curation into OnToma

    name_col = f"{entity_type}_name"
    names_col = f"{entity_type}_names"
    id_col = f"{entity_type}_id"
    normalised_name_col = f"normalised_{entity_type}_name"
    mapped_id_col = f"mapped_{entity_type}_id"

    lut = pl.concat(
        [
            mapping_index.explode(names_col).rename({names_col: name_col}),
            chembl_curation.filter(pl.col(name_col).is_not_null()),
        ],
        how="diagonal",
    ).select(
        pl.col(id_col).alias(mapped_id_col),
        normalise_label(pl.col(name_col)).alias(normalised_name_col),
    )

    df_w_id = (
        df.with_columns(normalise_label(pl.col(name_col)).alias(normalised_name_col))
        .join(
            lut,
            on=normalised_name_col,
            how="left",
        )
        .with_columns(pl.coalesce(pl.col(id_col), pl.col(mapped_id_col)).alias(id_col))
        .drop(mapped_id_col, normalised_name_col)
    )

    if verbose:
        unmapped_count = df_w_id.filter(pl.col(id_col).is_null()).height
        mapped_count = df_w_id.filter(pl.col(id_col).is_not_null()).height
        print(f"Unmapped {entity_type} rows: {unmapped_count}")
        print(f"Mapped {entity_type} rows: {mapped_count}")

    return df_w_id


def process_molecule(path: str) -> pl.DataFrame:
    """Loads and processes molecule data from a Parquet file."""
    return (
        pl.read_parquet(path)
        .with_columns(
            drug_names=pl.concat_list(
                [pl.col("synonyms"), pl.col("name"), pl.col("tradeNames")]
            ).list.unique()
        )
        .select(pl.col("id").alias("drug_id"), "drug_names")
    )


def process_disease(path: str) -> pl.DataFrame:
    """Loads and processes disease data from a Parquet file."""
    return (
        pl.read_parquet(path)
        .with_columns(
            disease_names=pl.concat_list(
                [
                    pl.col("synonyms"),
                    pl.col("name"),
                    pl.col("synonyms").struct.field("hasExactSynonym"),
                ]
            ).list.unique()
        )
        .select(pl.col("id").alias("disease_id"), "disease_names")
        .unique()
    )
