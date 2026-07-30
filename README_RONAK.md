Overview of files modified/added for the LLM Outcome Extraction project

src/clinical_mining/data_sources/aact/llm_extractor.py: The build_prompts_nct_combined_abstracts function contains the logic for building the prompts to be used as input in the LLM pipeline. This is accompanied by the helper function clean_or_none for normalization.

src/clinical_mining/prompts/outcome_analysis_llm.txt: Full prompt given to LLMs for classification

src/clinical_mining/recipe/outcome_classifier.yaml: Recipe file for running the full pipeline

src/clinical_mining/schemas.py: Full schema used for validating LLM output. The exact schemas relevant to this project are - Therapy, Condition, KeyReconcilliation, TrialOutcome, TrialExtraction.

src/clinical_mining/workflows/llm.py - Slight modifications for prompt caching functionality have been added.

scripts/run_openai_batch.py - Same as the recipe YAML file, but this file specifically uses the Open AI Batch API to produce the same output dataset.

src/clinical_mining/utils/processing_llm.py - Post-processing function for reshaping the output from run_openai_batch.py. Converts LLM output schema into desired format for downstream analysis.

src/clinical_mining/utils/mapping_llm.py - Mapping function for the output of the processing_llm.py file. Exact purpose is to include the Open Targets IDs for better integration into the platform.
