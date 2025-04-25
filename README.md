# Clinical Trial Mining

## Motivation

Mining clinical trials is essential for accelerating drug discovery and biomedical research. ClinicalTrials.gov contains a wealth of information on ongoing and completed studies, including interventions, conditions, and outcomes. Systematic extraction and integration of this data enables:

- Mapping of drug–disease relationships
- Identification of drug repurposing opportunities
- Analysis of intervention efficacy and safety

## Project Overview

This project provides tools to fetch, process, and annotate clinical trial data directly from the AACT (Aggregate Analysis of ClinicalTrials.gov) database. It is designed to facilitate large-scale mining and integration of clinical trials for drug discovery applications.

### Key Features

- **Direct Connection to AACT:** Uses a robust connector to securely access the AACT PostgreSQL database.
- **Automated Table Loading:** Loads and filters relevant tables (studies, interventions, conditions, etc.) using Spark for scalable processing.
- **Drug and Disease Mapping:** Integrates external drug and disease vocabularies to annotate interventions and indications in trials.
- **Output:** Produces harmonized datasets suitable for downstream analytics, including drug–disease mappings.

## How It Works

1. **Database Connection:**  
   The [AACTConnector](cci:2://file:///Users/irenelopez/EBI/repos/clinical_mining/src/clinical_mining/utils/db.py:5:0-72:32) class manages secure connections to the AACT PostgreSQL database using JDBC and Spark. Credentials and connection parameters are provided via configuration.

2. **Data Extraction:**  
   The pipeline loads core tables (e.g., `studies`, `interventions`, `conditions`) and applies filters (e.g., selecting drug interventions, excluding placebos).

3. **Annotation:**  
   - **Drug Mapping:** Interventions are mapped to standardized drug identifiers using the Open Targets molecule dataset.
   - **Disease Mapping:** Study conditions are mapped to standardized disease identifiers using the Open Targets disease dataset.

4. **Output Generation:**  
   The final, annotated dataset is saved in Parquet format for further analysis.

## Usage

1. **Configure Connection:**  
   Update your configuration with the AACT database host, user, password, and schema.

2. **Run the Pipeline:**  
   Execute the main script to fetch and process data:
   ```bash
   python src/clinical_mining/aact/main.py
   ```