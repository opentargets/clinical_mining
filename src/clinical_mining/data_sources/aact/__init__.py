"""Backward-compatible re-exports for AACT data source."""

from .clinical_report import (
    extract_clinical_report,
    process_interventions,
    process_conditions,
)

__all__ = [
    "extract_clinical_report",
    "process_interventions",
    "process_conditions",
]
