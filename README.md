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

## Data Sources

1. **AACT Database**  

    AACT database is a PostgreSQL database containing clinical trial data from the ClinicalTrials.gov database.
    Spark Session is connected to the AACT PostgreSQL database using JDBC. Credentials and connection parameters are provided via configuration.

2. **ChEMBL Drug Indication Data**

   JSON file storing indications for drugs, and clinical candidate drugs, from a variety of sources (e.g., FDA, EMA, WHO ATC, ClinicalTrials.gov, INN, USAN).

## Usage

### 1. Configure Your Environment

- **Database Credentials**: Open `src/clinical_mining/config.yaml` and fill in your AACT database credentials under the `db_properties` section.
- **File Paths**: Ensure the paths under the `datasets` section point to the correct locations for your input data files.

### 2. Run the Pipeline

Execute the main script from the root directory of the project:

```bash
uv run clinical_mining
```

You can override configuration parameters from the command line if needed:

```bash
uv run clinical_mining db_properties.user=<your_user> db_properties.password=<your_password>
```

## Pipeline Configuration

This pipeline is designed to be **config-driven**, allowing you to rearrange the steps to produce different outputs without changing the core Python code. The entire workflow is defined in `src/clinical_mining/config.yaml`.

### Pipeline Structure (DAG)

The pipeline is structured as a Directed Acyclic Graph (DAG) with three main stages, executed in order:

1.  **`setup`**: Steps used to prepare any data needed for later processes, such as loading mapping tables or performing initial filtering on large datasets.
2.  **`union`**: Steps that generate the primary drug/indication DataFrames. The outputs of all steps in this section are combined into a single DataFrame.
3.  **`process`**: Final transformation steps that run sequentially on the unified DataFrame.

The final, annotated dataset is saved in Parquet format for further analysis.

### Configuring a Pipeline Step

Each step within a stage is a dictionary with the following keys:

-   `name`: A unique name for the step. The output of this step will be stored and can be referenced by this name in later steps.
-   `function`: The full Python path to the function you want to execute.
-   `parameters`: A dictionary of arguments to pass to the function.
    -   **Reference other DataFrames**: To pass the output of a previous step or an initial input source as a parameter, use a `$` prefix.
    -   **Literal values**: Any value without a `$` prefix is treated as a literal.

The pipeline can be customised by editing the `pipeline` section of `config.yaml`. You can reorder steps, add new steps that call existing functions, or change parameters.

### Contribute

If you need to add a new data transformation or a new input source, please submit a pull request with the new functionality. Once merged, your function can be integrated into the pipeline via the configuration file.
