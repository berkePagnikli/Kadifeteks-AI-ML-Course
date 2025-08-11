import pandas as pd
import os
from pathlib import Path


def excel_to_csv(input_file, output_file=None):
    """Convert the first sheet of an Excel file to CSV format."""
    try:
        # Check if input file exists
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file '{input_file}' not found.")
        
        # Read only the first sheet of the Excel file
        print(f"Reading Excel file: {input_file}")
        df = pd.read_excel(input_file, sheet_name=0)  # sheet_name=0 gets the first sheet
        
        # If no output file specified, create one based on input filename
        if output_file is None:
            input_path = Path(input_file)
            output_file = input_path.with_suffix('.csv')
        
        # Convert to CSV
        print(f"Converting to CSV: {output_file}")
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"Successfully converted! Output saved as: {output_file}")
        print(f"Data shape: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Display first few rows as preview
        print("\nPreview of the data:")
        print(df.head())
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")


def main():
    """Main function."""

    input_file = "Talep Tahminleme Veri.xlsx"
    output_file = "talep2.csv"
    
    excel_to_csv(input_file, output_file)

if __name__ == "__main__":
    main()
