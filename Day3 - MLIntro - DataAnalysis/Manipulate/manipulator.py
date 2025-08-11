from data_manipulation import DataManipulator

manipulator = DataManipulator()
"""
# USE THIS FUNCTION FIRST, WHICH WILL AGGRAGATE THE DUPLICATED COLUMNS AS REQUESTED
duplicate_result = manipulator.aggregate_duplicate_rows()
print(f"\nDuplicate removal success: {duplicate_result.success}")
"""
# IF YOU WANT TO REPLACE THE MISSING VALUES PRESENT IN THE DATASET, COMMENT OUT a.k.a PUT CLOSE/DELETE THE AGGRAGATOR FUNCTION ABOVE
# THE DATA WHICH RESIDES IN manipulated_data.csv IS ALREADY AGGRAGATED + NA FREE
"""
# Replace missing values in 'İhtiyaç Kg' column
result = manipulator.replace_missing_ihtiyac_kg()
print(f"Missing value replacement success: {result.success}")
"""