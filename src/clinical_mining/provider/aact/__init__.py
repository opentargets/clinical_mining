"""Backward-compatible re-exports for the AACT provider."""

from .clinical_report import (
    extract_clinical_report,
    process_conditions,
    process_interventions,
)

__all__ = [
    "extract_clinical_report",
    "process_interventions",
    "process_conditions",
]
