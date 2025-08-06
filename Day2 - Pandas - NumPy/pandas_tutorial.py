"""
Pandas - Beginner Tutorial
Pandas: Powerful data manipulation and analysis library
Key structures: Series (1D) and DataFrame (2D)
"""

# Import necessary libraries
import pandas as pd
import numpy as np

print("=" * 50)
print("PANDAS TUTORIAL FOR BEGINNERS")
print("=" * 50)

# Check pandas version
print(f"Pandas version: {pd.__version__}")

# =====================================
# PART 1: PANDAS SERIES
# =====================================
print("\n1. PANDAS SERIES (1D Data)")
print("-" * 30)

# Creating Series
print("\n1.1 Creating Series:")
# From list
numbers = pd.Series([1, 2, 3, 4, 5])
print("Series from list:")
print(numbers)

# With custom index
fruits = pd.Series(['apple', 'banana', 'orange'], index=['a', 'b', 'c'])
print("\nSeries with custom index:")
print(fruits)

# From dictionary
scores = pd.Series({'math': 95, 'physics': 87, 'chemistry': 92})
print("\nSeries from dictionary:")
print(scores)

# Series properties
print("\n1.2 Series Properties:")
print(f"Values: {scores.values}")
print(f"Index: {scores.index}")
print(f"Size: {scores.size}")
print(f"Data type: {scores.dtype}")

# Accessing elements
print("\n1.3 Accessing Series Elements:")
print(f"Math score: {scores['math']}")
print(f"First score: {scores.iloc[0]}")
print(f"Scores > 90: \n{scores[scores > 90]}")

# =====================================
# PART 2: PANDAS DATAFRAME
# =====================================
print("\n\n2. PANDAS DATAFRAME (2D Data)")
print("-" * 30)

# Creating DataFrames
print("\n2.1 Creating DataFrames:")

# From dictionary
data_dict = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [25, 30, 35, 28],
    'City': ['New York', 'London', 'Tokyo', 'Paris'],
    'Salary': [50000, 60000, 75000, 55000]
}
df = pd.DataFrame(data_dict)
print("DataFrame from dictionary:")
print(df)

# From list of dictionaries
data_list = [
    {'Name': 'Eve', 'Age': 32, 'City': 'Berlin', 'Salary': 65000},
    {'Name': 'Frank', 'Age': 29, 'City': 'Madrid', 'Salary': 58000}
]
df2 = pd.DataFrame(data_list)
print("\nDataFrame from list of dictionaries:")
print(df2)

# DataFrame properties
print("\n2.2 DataFrame Properties:")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Index: {df.index.tolist()}")
print(f"Data types:\n{df.dtypes}")

# Basic information
print("\n2.3 DataFrame Info:")
print("\ndf.info():")
df.info()

print("\ndf.describe():")
print(df.describe())

# =====================================
# PART 3: SELECTING DATA
# =====================================
print("\n\n3. SELECTING DATA")
print("-" * 30)

# Selecting columns
print("\n3.1 Selecting Columns:")
print("Names column:")
print(df['Name'])

print("\nMultiple columns:")
print(df[['Name', 'Age']])

# Selecting rows
print("\n3.2 Selecting Rows:")
print("First row:")
print(df.iloc[0])

print("\nFirst two rows:")
print(df.iloc[0:2])

print("\nRows by condition:")
print(df[df['Age'] > 30])

# Selecting specific cells
print("\n3.3 Selecting Specific Cells:")
print(f"Alice's age: {df.loc[0, 'Age']}")
print(f"Age of people from New York: {df.loc[df['City'] == 'New York', 'Age'].values}")

# =====================================
# PART 4: ADDING AND MODIFYING DATA
# =====================================
print("\n\n4. ADDING AND MODIFYING DATA")
print("-" * 30)

# Adding new columns
print("\n4.1 Adding New Columns:")
df['Experience'] = [3, 5, 8, 4]  # Add experience column
df['Salary_K'] = df['Salary'] / 1000  # Salary in thousands
print("DataFrame with new columns:")
print(df)

# Modifying existing data
print("\n4.2 Modifying Data:")
df.loc[df['Name'] == 'Bob', 'Age'] = 31  # Update Bob's age
print("After updating Bob's age:")
print(df[['Name', 'Age']])

# Adding new rows
print("\n4.3 Adding New Rows:")
new_person = pd.DataFrame({
    'Name': ['Grace'],
    'Age': [27],
    'City': ['Sydney'],
    'Salary': [62000],
    'Experience': [4],
    'Salary_K': [62.0]
})
df = pd.concat([df, new_person], ignore_index=True)
print("After adding new person:")
print(df)

# =====================================
# PART 5: DATA FILTERING AND SORTING
# =====================================
print("\n\n5. DATA FILTERING AND SORTING")
print("-" * 30)

# Filtering data
print("\n5.1 Filtering Data:")
high_earners = df[df['Salary'] > 60000]
print("High earners (Salary > 60000):")
print(high_earners[['Name', 'Salary']])

experienced = df[df['Experience'] >= 5]
print("\nExperienced employees (Experience >= 5):")
print(experienced[['Name', 'Experience']])

# Multiple conditions
young_high_earners = df[(df['Age'] < 30) & (df['Salary'] > 55000)]
print("\nYoung high earners:")
print(young_high_earners[['Name', 'Age', 'Salary']])

# Sorting data
print("\n5.2 Sorting Data:")
df_sorted_age = df.sort_values('Age')
print("Sorted by age:")
print(df_sorted_age[['Name', 'Age']])

df_sorted_salary = df.sort_values('Salary', ascending=False)
print("\nSorted by salary (descending):")
print(df_sorted_salary[['Name', 'Salary']])

# =====================================
# PART 6: GROUPING AND AGGREGATION
# =====================================
print("\n\n6. GROUPING AND AGGREGATION")
print("-" * 30)

# Create sample data for grouping
dept_data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank'],
    'Department': ['IT', 'HR', 'IT', 'Finance', 'IT', 'HR'],
    'Salary': [70000, 50000, 80000, 60000, 75000, 55000],
    'Experience': [5, 3, 7, 4, 6, 2]
}
dept_df = pd.DataFrame(dept_data)
print("Department data:")
print(dept_df)

# Group by department
print("\n6.1 Grouping by Department:")
dept_groups = dept_df.groupby('Department')
print("Average salary by department:")
print(dept_groups['Salary'].mean())

print("\nMultiple aggregations:")
dept_summary = dept_groups.agg({
    'Salary': ['mean', 'max', 'min'],
    'Experience': 'mean'
})
print(dept_summary)

# =====================================
# PART 7: HANDLING MISSING DATA
# =====================================
print("\n\n7. HANDLING MISSING DATA")
print("-" * 30)

# Create data with missing values
data_with_nan = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [25, None, 35, 28],
    'Salary': [50000, 60000, None, 55000],
    'City': ['New York', 'London', 'Tokyo', None]
}
df_nan = pd.DataFrame(data_with_nan)
print("DataFrame with missing values:")
print(df_nan)

# Check for missing values
print("\n7.1 Checking Missing Values:")
print("Missing values per column:")
print(df_nan.isnull().sum())

print("\nRows with any missing values:")
print(df_nan[df_nan.isnull().any(axis=1)])

# Handle missing values
print("\n7.2 Handling Missing Values:")

# Drop rows with missing values
df_dropna = df_nan.dropna()
print("After dropping rows with NaN:")
print(df_dropna)

# Fill missing values
df_filled = df_nan.fillna({
    'Age': df_nan['Age'].mean(),
    'Salary': df_nan['Salary'].median(),
    'City': 'Unknown'
})
print("\nAfter filling missing values:")
print(df_filled)

# =====================================
# PART 8: READING AND WRITING DATA
# =====================================
print("\n\n8. READING AND WRITING DATA")
print("-" * 30)

# Create sample data and save to CSV
print("\n8.1 Writing to CSV:")
sample_data = pd.DataFrame({
    'Product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor'],
    'Price': [999, 25, 75, 299],
    'Stock': [50, 200, 150, 75]
})
sample_data.to_csv('sample_data.csv', index=False)
print("Data saved to 'sample_data.csv'")
print(sample_data)

# Read from CSV
print("\n8.2 Reading from CSV:")
try:
    loaded_data = pd.read_csv('sample_data.csv')
    print("Data loaded from CSV:")
    print(loaded_data)
except FileNotFoundError:
    print("CSV file not found (this is normal in tutorial mode)")

# =====================================
# PRACTICAL EXAMPLES
# =====================================
print("\n\n9. PRACTICAL EXAMPLES")
print("-" * 30)

print("\n9.1 Sales Analysis:")
# Create sales data
sales_data = {
    'Date': pd.date_range('2024-01-01', periods=12, freq='ME'),
    'Product': ['Laptop', 'Mouse', 'Keyboard'] * 4,
    'Sales': [150, 300, 200, 180, 350, 250, 200, 400, 300, 170, 380, 280],
    'Revenue': [149700, 7500, 15000, 179460, 8750, 18750, 199800, 10000, 22500, 169830, 9500, 21000]
}
sales_df = pd.DataFrame(sales_data)
print("Sales data:")
print(sales_df.head())

# Analyze sales by product
product_summary = sales_df.groupby('Product').agg({
    'Sales': 'sum',
    'Revenue': 'sum'
}).round(2)
print("\nSales summary by product:")
print(product_summary)

# Find best performing month
monthly_sales = sales_df.groupby(sales_df['Date'].dt.month)['Revenue'].sum()
best_month = monthly_sales.idxmax()
print(f"\nBest performing month: {best_month} (Revenue: ${monthly_sales.max():,.2f})")

print("\n9.2 Student Performance Analysis:")
# Create student data
student_data = {
    'Student_ID': range(1, 11),
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 
             'Frank', 'Grace', 'Henry', 'Iris', 'Jack'],
    'Math': [85, 92, 78, 94, 88, 76, 90, 82, 95, 89],
    'Science': [90, 85, 82, 91, 86, 79, 93, 87, 92, 84],
    'English': [88, 89, 85, 87, 92, 83, 91, 86, 89, 90],
    'Grade': ['A', 'A', 'B', 'A', 'B', 'C', 'A', 'B', 'A', 'B']
}
student_df = pd.DataFrame(student_data)

# Calculate total scores
student_df['Total'] = student_df[['Math', 'Science', 'English']].sum(axis=1)
student_df['Average'] = student_df[['Math', 'Science', 'English']].mean(axis=1)

print("Student performance:")
print(student_df[['Name', 'Math', 'Science', 'English', 'Average', 'Grade']].head())

# Find top performers
top_students = student_df.nlargest(3, 'Average')
print("\nTop 3 students:")
print(top_students[['Name', 'Average']])

# Grade distribution
grade_dist = student_df['Grade'].value_counts()
print("\nGrade distribution:")
print(grade_dist)

# Subject averages
subject_avg = student_df[['Math', 'Science', 'English']].mean()
print("\nSubject averages:")
for subject, avg in subject_avg.items():
    print(f"{subject}: {avg:.1f}")

# =====================================
# WHY USE PANDAS?
# =====================================
print("\n\n10. WHY USE PANDAS?")
print("-" * 30)
print("""
PANDAS ADVANTAGES:
1. Easy data manipulation: Filter, sort, group data effortlessly
2. Handle missing data: Built-in methods for NaN values
3. File I/O: Read/write CSV, Excel, JSON, SQL databases
4. Data analysis: Statistical operations and aggregations
5. Integration: Works seamlessly with NumPy and matplotlib

WHEN TO USE PANDAS:
- Data cleaning and preprocessing
- Exploratory data analysis
- Working with structured data (tables)
- Time series analysis
- Data visualization preparation
""")

print("\n" + "=" * 50)
print("End of Pandas Tutorial")
print("=" * 50)