import polars as pl


def normalise_label(label: pl.Series) -> pl.Series:
    return label.str.to_lowercase().str.replace(r"[^a-z0-9]", "")


def assign_drug_id(
    df: pl.DataFrame,
    molecule: pl.DataFrame,
    chembl_curation: pl.DataFrame,
    verbose: bool = True,
) -> pl.DataFrame:
    """Assigns drug IDs by joining with a molecule mapping table + ChEMBL clinical trial curation."""
    # TODO: Refactor to use ChEMBL curation into OnToma
    drug_lut = pl.concat(
        [
            molecule.explode("drug_names").with_columns(
                normalised_drug_name=normalise_label(pl.col("drug_names")).alias(
                    "normalised_drug_name"
                )
            ),
            chembl_curation.filter(pl.col("drug_name").is_not_null()).select(
                normalise_label(pl.col("drug_name")).alias("normalised_drug_name"),
                pl.col("drug_id").alias("mapped_drug_id"),
            ),
        ],
        how="diagonal",
    )
    df_w_id = (
        df.with_columns(
            normalised_drug_name=normalise_label(pl.col("drug_name")).alias(
                "normalised_drug_name"
            )
        )
        .join(
            drug_lut,
            on="normalised_drug_name",
            how="left",
        )
        .with_columns(drug_id=pl.coalesce(["drug_id", "mapped_drug_id"]))
        .drop("mapped_drug_id", "normalised_drug_name")
    )
    if verbose:
        unmapped_count = df_w_id.filter(pl.col("drug_id").is_null()).height
        mapped_count = df_w_id.filter(pl.col("drug_id").is_not_null()).height
        print(f"Unmapped drug rows: {unmapped_count}")
        print(f"Mapped drug rows: {mapped_count}")
    return df_w_id


def assign_disease_id(
    df: pl.DataFrame,
    diseases: pl.DataFrame,
    chembl_curation: pl.DataFrame,
    verbose: bool = True,
) -> pl.DataFrame:
    """Assigns disease IDs by joining with a disease mapping table + ChEMBL clinical trial curation."""
    # TODO: Refactor to use ChEMBL curation into OnToma
    diseases_lut = pl.concat(
        [
            diseases.explode("disease_names").with_columns(
                normalised_disease_name=normalise_label(pl.col("disease_names")).alias(
                    "normalised_disease_name"
                )
            ),
            chembl_curation.filter(pl.col("disease_name").is_not_null()).select(
                normalise_label(pl.col("disease_name")).alias(
                    "normalised_disease_name"
                ),
                pl.col("disease_id").alias("mapped_disease_id"),
            ),
        ],
        how="diagonal",
    )
    df_w_id = (
        df.with_columns(
            normalised_disease_name=normalise_label(pl.col("disease_name")).alias(
                "normalised_disease_name"
            )
        )
        .join(
            diseases_lut,
            on="normalised_disease_name",
            how="left",
        )
        .with_columns(disease_id=pl.coalesce(["disease_id", "mapped_disease_id"]))
        .drop("mapped_disease_id", "normalised_disease_name")
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
        .with_columns(
            drug_names=pl.concat_list(
                [pl.col("synonyms"), pl.col("name"), pl.col("tradeNames")]
            ).list.unique()
        )
        .select(pl.col("id").alias("mapped_drug_id"), "drug_names")
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
        .select(pl.col("id").alias("mapped_disease_id"), "disease_names")
        .unique()
    )
