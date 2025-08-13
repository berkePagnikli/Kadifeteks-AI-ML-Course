import pandas as pd
import numpy as np

# Function to winsorize a column
def winsorize_column(column, lower_percentile=5, upper_percentile=95):
    lower_threshold = np.percentile(column, lower_percentile)
    upper_threshold = np.percentile(column, upper_percentile)
    return np.clip(column, lower_threshold, upper_threshold)

# Load dataset
file_path = 'DataSets\Concrete_Data.csv'
data = pd.read_csv(file_path)

# Winsorize each numeric column in the dataset
for column in data.columns:
    data[column] = winsorize_column(data[column])

# Save the winsorized data
output_file_path = 'DataSets\Concrete_Data_Winsorized.csv'
data.to_csv(output_file_path, index=False)