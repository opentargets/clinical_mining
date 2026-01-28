"""Extraction of drug/indication relationships grounded to EFO and ChEMBL IDs from ChEMBL Clinical Trials Curation tables."""

import polars as pl

from clinical_mining.utils.db import load_oracle_table


def extract_chembl_ct_curation(
    db_host: str,
    db_port: int,
    db_service: str,
    db_user: str,
    db_password: str,
    oracle_client_path: str,
) -> pl.DataFrame:
    """Reads the following tables from ChEMBL Clinical Trials Curation database:
    
    CT_NCTID_CONDITION_EFO: Clinical trial to EFO mapping
    CT_NCT_ID2MOLREGNO: Clinical trial to MOLREGNO mapping (internal ChEMBL drug ID)
    MOLECULE_DICTIONARY: MOLREGNO to ChEMBL ID mapping

    Returns:
        pl.DataFrame: DataFrame with drug/indication relationships grounded to EFO and ChEMBL IDs for each clinical trial
    """
    trial_to_efo = (
        load_oracle_table(
            table_name="CT_NCTID_CONDITION_EFO",
            select_cols=["NCT_ID", "CONDITION_1", "EFO", "IS_INDICATION"],
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port,
            service=db_service,
            db_schema="DRUGBASE_CURATION",
            limit=None,
            init_client_lib_dir=oracle_client_path,
        )
        .filter(~pl.col("EFO").is_in(["-NOT-CONDITION-", "-ABSENT-"]))
        .select(
            pl.col("NCT_ID").alias("studyId"),
            pl.col("CONDITION_1").alias("diseaseFromSource"),
            pl.col("EFO").str.replace(":", "_").alias("diseaseId"),
        )
        .drop_nulls()
        .unique()
    )

    trial_to_drug = (
        load_oracle_table(
            table_name="CT_NCT_ID2MOLREGNO",
            select_cols=["NCT_ID", "INTERVENTION_NAME", "MOLREGNO", "ARM_GROUP_TYPE"],
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port,
            service=db_service,
            db_schema="DRUGBASE_CURATION",
            limit=None,
            init_client_lib_dir=oracle_client_path,
        )
        .filter(
            ~pl.col("MOLREGNO").is_in(
                ["-NOT-DRUG-", "-ABSENT-", "-NOT-SPECIFIED-DRUG/S-"]
            )
        )
        .select(
            pl.col("NCT_ID").alias("studyId"),
            pl.col("INTERVENTION_NAME").alias("drugFromSource"),
            "MOLREGNO",
        )
        .drop_nulls()
        .unique()
    )

    chembl = (
        load_oracle_table(
            table_name="MOLECULE_DICTIONARY",
            select_cols=["MOLREGNO", "CHEMBL_ID"],
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port,
            service=db_service,
            db_schema="CHEMBL_36",
            limit=None,
            init_client_lib_dir=oracle_client_path,
        )
        .select(pl.col("MOLREGNO").cast(pl.Utf8), pl.col("CHEMBL_ID").alias("drugId"))
        .drop_nulls()
        .unique()
    )
    return (
        trial_to_drug.join(trial_to_efo, "studyId", how="full")
        .join(chembl, "MOLREGNO", how="inner")
        .select("studyId", "drugId", "diseaseId", "drugFromSource", "diseaseFromSource")
        .unique()
    )
