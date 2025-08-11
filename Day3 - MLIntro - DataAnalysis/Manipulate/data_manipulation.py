from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class ManipulationResult:
    """Container for data manipulation results."""
    original_missing_count: int
    replacement_value: float
    rows_affected: int
    output_file: str
    success: bool


@dataclass
class DuplicateAggregationResult:
    """Container for duplicate aggregation results."""
    original_row_count: int
    duplicate_count: int
    final_row_count: int
    output_file: str
    success: bool


class DataManipulator:
    """Unified data manipulation class for the hardcoded CSV dataset."""

    def __init__(self):
        self.input_file_path: str = "talep_tahminleme.csv"
        self.output_file_path: str = "manipulated_data.csv"
        self._df: Optional[pd.DataFrame] = None

    def _load_dataframe(self) -> pd.DataFrame:
        """Load the CSV once and cache it for subsequent calls."""
        if self._df is None:
            try:
                self._df = pd.read_csv(self.input_file_path)
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"CSV file not found at '{self.input_file_path}'"
                ) from exc
        return self._df

    def replace_missing_ihtiyac_kg(self, replacement_value: float = 23.354):
        """Find and replace missing values in 'İhtiyaç Kg' column, then save to new CSV."""

        df = self._load_dataframe().copy() 
        column_name = "İhtiyaç Kg"
        
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in dataset. Available columns: {list(df.columns)}")
        
        # Count original missing values
        original_missing = df[column_name].isna().sum()
        
        # Replace missing values
        df[column_name] = df[column_name].fillna(replacement_value)
        
        # Count how many rows were actually affected
        rows_affected = int(original_missing)
        
        # Save the manipulated data
        try:
            df.to_csv(self.output_file_path, index=False)
            success = True
        except Exception as exc:
            raise IOError(f"Failed to save manipulated data to '{self.output_file_path}': {exc}") from exc
        
        return ManipulationResult(
            original_missing_count=int(original_missing),
            replacement_value=replacement_value,
            rows_affected=rows_affected,
            output_file=self.output_file_path,
            success=success
        )

    def aggregate_duplicate_rows(self) -> DuplicateAggregationResult:
        """Aggregate duplicate rows by summing 'İhtiyaç Kg' values and save to a new CSV file."""
        
        df = self._load_dataframe().copy() 
        
        # Count original rows
        original_count = len(df)
        
        if "İhtiyaç Kg" not in df.columns:
            raise ValueError(f"Column 'İhtiyaç Kg' not found in dataset. Available columns: {list(df.columns)}")
        
        # Group duplicated columns
        grouping_columns = [col for col in df.columns if col != "İhtiyaç Kg"]
        
        # Group each duplicated column in itself and sum the values
        # dropna=False ensures NA values in grouping columns are preserved
        # min_count=1 ensures that if all values in a group are NA, the result remains NA
        df_aggregated = df.groupby(grouping_columns, as_index=False, dropna=False)["İhtiyaç Kg"].sum(min_count=1)
        
        final_count = len(df_aggregated)
        duplicates_aggregated = original_count - final_count
        
        # Save the aggregated data
        try:
            df_aggregated.to_csv(self.output_file_path, index=False)
            success = True
        except Exception as e:
            raise e
        
        return DuplicateAggregationResult(
            original_row_count=original_count,
            duplicate_count=duplicates_aggregated,
            final_row_count=final_count,
            output_file=self.output_file_path,
            success=success
        )
