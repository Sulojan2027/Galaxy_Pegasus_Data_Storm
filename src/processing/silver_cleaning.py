"""
Module for Silver Layer (Cleaned Data).

Responsibilities:
- Apply DE checks and data cleaning logic from data_quality.py.
- Quarantine failed records into a separate rejected records store.
- Document failure reasons for quarantined records.
- Produce sanitized datasets, separating true market signals from system artifacts.
"""
