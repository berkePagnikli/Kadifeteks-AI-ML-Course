from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import numpy as np


@dataclass
class CorrelationAnalysisResult:
    """Container for correlation analysis results."""
    target_column: str
    correlation_with_target: Dict[str, float]
    correlation_matrix: pd.DataFrame
    numeric_features: List[str]
    plots_created: List[str]
    success: bool


@dataclass
class CategoricalAnalysisResult:
    """Container for categorical feature analysis results."""
    target_column: str
    categorical_features: List[str]
    feature_importance: Dict[str, float]
    group_statistics: Dict[str, Dict]
    plots_created: List[str]
    success: bool


@dataclass
class StatisticalSummary:
    """Container for statistical analysis results."""
    column_name: str
    count: int
    mean: float
    median: float
    min_value: float
    max_value: float
    std: float
    percentile_25: float
    percentile_50: float
    percentile_75: float
    percentile_90: float
    iqr: float


@dataclass
class DatasetOverview:
    """A small container for basic dataset info."""

    num_rows: int
    num_features: int
    feature_names: List[str]


class DataAnalyzer:
    """Unified analyzer for the hardcoded CSV dataset."""

    def __init__(self):
        file_path = "manipulated_data.csv"
        self.file_path: str = file_path
        self._df: Optional[pd.DataFrame] = None


    def _load_dataframe(self):
        """Load the CSV once and cache it for subsequent calls."""
        if self._df is None:
            try:
                self._df = pd.read_csv(self.file_path)
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"CSV file not found at '{self.file_path}'"
                ) from exc
        return self._df

    def get_dataset_overview(self):
        """Return number of rows, number of features, and the feature names."""

        df = self._load_dataframe()
        feature_names = list(df.columns.astype(str))
        return DatasetOverview(
            num_rows=int(df.shape[0]),
            num_features=int(df.shape[1]),
            feature_names=feature_names,
        )

    def get_missing_value(self):
        """List where missing values occur as row/column pairs."""

        df = self._load_dataframe()
        if df.empty:
            return pd.DataFrame(columns=["row_index", "column"])

        mask = df.isna()
        if not mask.values.any():
            return pd.DataFrame(columns=["row_index", "column"])

        stacked = mask.stack()
        missing = stacked[stacked]
        out = missing.reset_index()[["level_0", "level_1"]]
        out.columns = ["row_index", "column"]
        return out

    def get_duplicate_rows(self):
        """Return the duplicated data points (rows), excluding the first occurrence."""

        df = self._load_dataframe()
        if df.empty:
            return df.copy()

        dup_mask = df.duplicated(keep="first")
        duplicates = df[dup_mask].copy()
        return duplicates

    def analyze_ihtiyac_statistics(self) -> StatisticalSummary:
        """Analyze the 'İhtiyaç Kg' column for statistical summary and spike detection."""
        df = self._load_dataframe()
        column_name = "İhtiyaç Kg"
        
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in dataset. Available columns: {list(df.columns)}")
        
        # Extract the column and remove any non-numeric or null values
        series = pd.to_numeric(df[column_name], errors='coerce').dropna()
        
        if series.empty:
            raise ValueError(f"No valid numeric data found in column '{column_name}'")
        
        # Calculate basic statistics
        count = len(series)
        mean_val = float(series.mean())
        median_val = float(series.median())
        min_val = float(series.min())
        max_val = float(series.max())
        std_val = float(series.std())
        
        # Calculate percentiles
        percentiles = series.quantile([0.25, 0.50, 0.75, 0.90])
        q1 = float(percentiles[0.25])
        q2 = float(percentiles[0.50])  # median
        q3 = float(percentiles[0.75])
        p90 = float(percentiles[0.90])
        
        # Calculate IQR and identify potential outliers using IQR method
        iqr = q3 - q1
        
        return StatisticalSummary(
            column_name=column_name,
            count=count,
            mean=mean_val,
            median=median_val,
            min_value=min_val,
            max_value=max_val,
            std=std_val,
            percentile_25=q1,
            percentile_50=q2,
            percentile_75=q3,
            percentile_90=p90,
            iqr=iqr,
        )

    def analyze_feature_correlations(self, target_column: str = "İhtiyaç Kg"):
        """Analyze correlations between features and create visualization plots."""
        df = self._load_dataframe()
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset. Available columns: {list(df.columns)}")
        
        # Select only numeric columns for correlation analysis
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            raise ValueError("No numeric columns found in the dataset for correlation analysis")
        
        if target_column not in numeric_df.columns:
            raise ValueError(f"Target column '{target_column}' is not numeric and cannot be used for correlation analysis")
        
        # Calculate correlation matrix
        correlation_matrix = numeric_df.corr()
        
        # Get correlations with target column
        target_correlations = correlation_matrix[target_column].drop(target_column)  # Remove self-correlation
        correlation_dict = target_correlations.to_dict()
        
        # Create visualizations
        plots_created = []
        
        try:
            # Set up the plotting style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Plot 1: Correlation Heatmap
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # Mask upper triangle
            sns.heatmap(correlation_matrix, 
                       annot=True, 
                       cmap='RdBu_r', 
                       center=0,
                       square=True,
                       mask=mask,
                       fmt='.2f',
                       cbar_kws={"shrink": .8})
            plt.title(f'Feature Correlation Matrix\n(Lower Triangle Only)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            heatmap_file = 'correlation_heatmap.png'
            plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
            plt.close()
            plots_created.append(heatmap_file)
            
            success = True
            
        except Exception as e:
            print(f"Warning: Could not create some plots due to: {e}")
            success = len(plots_created) > 0  # Partial success if some plots were created
        
        return CorrelationAnalysisResult(
            target_column=target_column,
            correlation_with_target=correlation_dict,
            correlation_matrix=correlation_matrix,
            numeric_features=list(numeric_df.columns),
            plots_created=plots_created,
            success=success
        )

    def analyze_categorical_contribution(self, target_column: str = "İhtiyaç Kg", max_categories: int = 20):
        """Analyze how categorical features contribute to the target variable."""
        
        df = self._load_dataframe()
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset. Available columns: {list(df.columns)}")
        
        # Ensure target is numeric
        target_series = pd.to_numeric(df[target_column], errors='coerce').dropna()
        if target_series.empty:
            raise ValueError(f"Target column '{target_column}' contains no valid numeric data")
        
        # Get categorical columns (non-numeric, excluding target)
        categorical_columns = []
        for col in df.columns:
            if col != target_column:
                if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                    categorical_columns.append(col)
        
        if not categorical_columns:
            raise ValueError(f"No suitable categorical columns found (max {max_categories} categories per column)")
        
        feature_importance = {}
        group_statistics = {}
        plots_created = []
        
        try:
            # Calculate ANOVA F-statistic for each categorical feature
            for cat_col in categorical_columns:
                # Get complete cases (no missing values in either column)
                mask = df[cat_col].notna() & df[target_column].notna()
                cat_data = df.loc[mask, cat_col]
                target_data = pd.to_numeric(df.loc[mask, target_column], errors='coerce')
                
                if len(cat_data) < 2 or target_data.isna().all():
                    continue
                
                # Group target values by categorical values
                groups = [target_data[cat_data == cat].dropna() for cat in cat_data.unique() if len(target_data[cat_data == cat].dropna()) > 0]
                
                if len(groups) < 2:
                    feature_importance[cat_col] = 0.0
                    continue
                
                # Perform ANOVA
                try:
                    f_stat, p_value = stats.f_oneway(*groups)
                    # Use F-statistic as importance score (higher = more important)
                    feature_importance[cat_col] = float(f_stat) if not np.isnan(f_stat) else 0.0
                except:
                    feature_importance[cat_col] = 0.0
                
                # Calculate group statistics
                group_stats = {}
                for category in cat_data.unique():
                    category_values = target_data[cat_data == category].dropna()
                    if len(category_values) > 0:
                        group_stats[str(category)] = {
                            'count': len(category_values),
                            'mean': float(category_values.mean()),
                            'std': float(category_values.std()) if len(category_values) > 1 else 0.0,
                            'min': float(category_values.min()),
                            'max': float(category_values.max())
                        }
                group_statistics[cat_col] = group_stats
            
            # Create visualizations for top important features
            if feature_importance:
                # Sort features by importance
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                
                # Plot 1: Feature Importance Bar Chart
                plt.figure(figsize=(12, 8))
                features, scores = zip(*sorted_features)
                bars = plt.barh(range(len(features)), scores, color='skyblue', alpha=0.7)
                plt.yticks(range(len(features)), features)
                plt.xlabel('ANOVA F-Statistic (Importance Score)', fontweight='bold')
                plt.title(f'Categorical Feature Importance for "{target_column}"', fontsize=14, fontweight='bold')
                
                # Add value labels
                for i, (bar, score) in enumerate(zip(bars, scores)):
                    plt.text(score + max(scores) * 0.01, i, f'{score:.2f}', 
                            va='center', ha='left', fontweight='bold')
                
                plt.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                importance_file = f'categorical_importance_{target_column.replace(" ", "_")}.png'
                plt.savefig(importance_file, dpi=300, bbox_inches='tight')
                plt.close()
                plots_created.append(importance_file)
            
            success = True
            
        except Exception as e:
            print(f"Warning: Could not complete categorical analysis due to: {e}")
            success = len(plots_created) > 0
        
        return CategoricalAnalysisResult(
            target_column=target_column,
            categorical_features=categorical_columns,
            feature_importance=feature_importance,
            group_statistics=group_statistics,
            plots_created=plots_created,
            success=success
        )
    
    def plot_ihtiyac_distribution(self):

        df = self._load_dataframe()
        
        column_name = "İhtiyaç Kg"
        series = pd.to_numeric(df[column_name], errors='coerce').dropna()

        sorted_values = series.sort_values(ascending=False).reset_index(drop=True)

        plt.figure(figsize=(14, 8))
        plt.plot(range(len(sorted_values)), sorted_values, 
                linewidth=2, color='steelblue', alpha=0.8)
        
        plt.title('İhtiyaç Kg Veri Dağılımı (Büyükten Küçüğe Sıralı)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Veri Noktası Sırası', fontsize=12)
        plt.ylabel('İhtiyaç Kg Değeri', fontsize=12)
        plt.grid(True, alpha=0.3, linestyle='--')

        mean_val = sorted_values.mean()
        median_val = sorted_values.median()
        
        plt.axhline(y=mean_val, color='red', linestyle='--', alpha=0.7, 
                   label=f'Ortalama: {mean_val:.2f}')
        plt.axhline(y=median_val, color='green', linestyle='--', alpha=0.7, 
                   label=f'Medyan: {median_val:.2f}')
        
        plt.legend(loc='upper right')
        
        # Layout düzenle
        plt.tight_layout()
        data_frequency = 'frequency_graph.png'
        plt.savefig(data_frequency, dpi=300, bbox_inches='tight')
        plt.close()
        return sorted_values