"""
NumPy (Numerical Python) - Beginner Tutorial
NumPy: Powerful library for numerical computing with arrays
Key features: Fast array operations, mathematical functions, broadcasting
"""

# Import numpy
import numpy as np

print("=" * 50)
print("NUMPY TUTORIAL FOR BEGINNERS")
print("=" * 50)

# =====================================
# PART 1: CREATING ARRAYS
# =====================================
print("\n1. CREATING NUMPY ARRAYS")
print("-" * 30)

# From Python lists
print("\n1.1 Creating Arrays from Lists:")
list_1d = [1, 2, 3, 4, 5]
array_1d = np.array(list_1d)
print(f"Python list: {list_1d}")
print(f"NumPy array: {array_1d}")
print(f"Array type: {type(array_1d)}")

# 2D arrays
list_2d = [[1, 2, 3], [4, 5, 6]]
array_2d = np.array(list_2d)
print(f"\n2D list: {list_2d}")
print(f"2D array:\n{array_2d}")

# Array properties
print("\n1.2 Array Properties:")
print(f"Shape: {array_2d.shape}")
print(f"Size: {array_2d.size}")
print(f"Dimensions: {array_2d.ndim}")
print(f"Data type: {array_2d.dtype}")

# Creating arrays with specific functions
print("\n1.3 Creating Arrays with Functions:")

# Zeros and ones
zeros_array = np.zeros((3, 4))
ones_array = np.ones((2, 3))
print(f"Zeros array (3x4):\n{zeros_array}")
print(f"Ones array (2x3):\n{ones_array}")

# Range arrays
range_array = np.arange(0, 10, 2)  # start, stop, step
linspace_array = np.linspace(0, 1, 5)  # start, stop, num_points
print(f"Range array (0 to 10, step 2): {range_array}")
print(f"Linspace array (0 to 1, 5 points): {linspace_array}")

# Random arrays
np.random.seed(42)  # For reproducible results
random_array = np.random.random((2, 3))
random_int_array = np.random.randint(1, 10, size=(2, 4))
print(f"Random array (2x3):\n{random_array}")
print(f"Random integers (2x4):\n{random_int_array}")

# =====================================
# PART 2: ARRAY INDEXING AND SLICING
# =====================================
print("\n\n2. ARRAY INDEXING AND SLICING")
print("-" * 30)

# 1D array indexing
print("\n2.1 1D Array Indexing:")
arr = np.array([10, 20, 30, 40, 50])
print(f"Array: {arr}")
print(f"First element: {arr[0]}")
print(f"Last element: {arr[-1]}")
print(f"Elements 1 to 3: {arr[1:4]}")

# 2D array indexing
print("\n2.2 2D Array Indexing:")
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"Matrix:\n{matrix}")
print(f"Element at row 1, col 2: {matrix[1, 2]}")
print(f"First row: {matrix[0, :]}")
print(f"Second column: {matrix[:, 1]}")
print(f"Submatrix:\n{matrix[0:2, 1:3]}")

# Boolean indexing
print("\n2.3 Boolean Indexing:")
data = np.array([1, 5, 3, 8, 2, 7])
condition = data > 4
print(f"Data: {data}")
print(f"Condition (> 4): {condition}")
print(f"Elements > 4: {data[condition]}")

# =====================================
# PART 3: ARRAY OPERATIONS
# =====================================
print("\n\n3. ARRAY OPERATIONS")
print("-" * 30)

# Basic arithmetic
print("\n3.1 Basic Arithmetic Operations:")
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])
print(f"Array a: {a}")
print(f"Array b: {b}")
print(f"Addition: {a + b}")
print(f"Subtraction: {a - b}")
print(f"Multiplication: {a * b}")
print(f"Division: {a / b}")

# Operations with scalars
print("\n3.2 Operations with Scalars:")
print(f"a + 10: {a + 10}")
print(f"a * 2: {a * 2}")
print(f"a ** 2: {a ** 2}")

# Mathematical functions
print("\n3.3 Mathematical Functions:")
angles = np.array([0, np.pi/2, np.pi])
print(f"Angles: {angles}")
print(f"Sine: {np.sin(angles)}")
print(f"Cosine: {np.cos(angles)}")

numbers = np.array([1, 4, 9, 16])
print(f"Numbers: {numbers}")
print(f"Square root: {np.sqrt(numbers)}")
print(f"Natural log: {np.log(numbers)}")

# =====================================
# PART 4: ARRAY STATISTICS
# =====================================
print("\n\n4. ARRAY STATISTICS")
print("-" * 30)

# Statistical functions
print("\n4.1 Statistical Functions:")
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(f"Data: {data}")
print(f"Sum: {np.sum(data)}")
print(f"Mean: {np.mean(data)}")
print(f"Median: {np.median(data)}")
print(f"Standard deviation: {np.std(data)}")
print(f"Minimum: {np.min(data)}")
print(f"Maximum: {np.max(data)}")

# 2D array statistics
print("\n4.2 2D Array Statistics:")
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"Matrix:\n{matrix}")
print(f"Sum of all elements: {np.sum(matrix)}")
print(f"Sum along rows (axis=0): {np.sum(matrix, axis=0)}")
print(f"Sum along columns (axis=1): {np.sum(matrix, axis=1)}")

# =====================================
# PART 5: ARRAY RESHAPING
# =====================================
print("\n\n5. ARRAY RESHAPING")
print("-" * 30)

# Reshape arrays
print("\n5.1 Reshaping Arrays:")
original = np.array([1, 2, 3, 4, 5, 6])
print(f"Original (6,): {original}")

reshaped_2x3 = original.reshape(2, 3)
print(f"Reshaped (2x3):\n{reshaped_2x3}")

reshaped_3x2 = original.reshape(3, 2)
print(f"Reshaped (3x2):\n{reshaped_3x2}")

# Flatten arrays
flattened = reshaped_2x3.flatten()
print(f"Flattened: {flattened}")

# Transpose
print("\n5.2 Transpose:")
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(f"Original matrix:\n{matrix}")
print(f"Transposed:\n{matrix.T}")

# =====================================
# PART 6: ARRAY CONCATENATION
# =====================================
print("\n\n6. ARRAY CONCATENATION")
print("-" * 30)

# Concatenating arrays
print("\n6.1 Concatenating Arrays:")
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
concatenated = np.concatenate([arr1, arr2])
print(f"Array 1: {arr1}")
print(f"Array 2: {arr2}")
print(f"Concatenated: {concatenated}")

# Stacking arrays
print("\n6.2 Stacking Arrays:")
vertical_stack = np.vstack([arr1, arr2])
horizontal_stack = np.hstack([arr1, arr2])
print(f"Vertical stack:\n{vertical_stack}")
print(f"Horizontal stack: {horizontal_stack}")

# =====================================
# PRACTICAL EXAMPLES
# =====================================
print("\n\n7. PRACTICAL EXAMPLES")
print("-" * 30)

print("\n7.1 Temperature Conversion:")
celsius = np.array([0, 20, 30, 40, 100])
fahrenheit = celsius * 9/5 + 32
print(f"Celsius: {celsius}")
print(f"Fahrenheit: {fahrenheit}")

print("\n7.2 Student Grades Analysis:")
# Grades for 4 students in 3 subjects
grades = np.array([
    [85, 92, 78],  # Student 1
    [90, 88, 94],  # Student 2
    [76, 85, 89],  # Student 3
    [88, 91, 87]   # Student 4
])

print("Grades matrix (students x subjects):")
print(grades)

student_averages = np.mean(grades, axis=1)
subject_averages = np.mean(grades, axis=0)

print(f"Student averages: {student_averages}")
print(f"Subject averages: {subject_averages}")
print(f"Class average: {np.mean(grades)}")

# Find best student
best_student = np.argmax(student_averages)
print(f"Best student (index {best_student}): {student_averages[best_student]:.1f}")

print("\n7.3 Sales Data Analysis:")
# Monthly sales data
months = np.array(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'])
sales = np.array([15000, 18000, 22000, 19000, 25000, 28000])

print(f"Months: {months}")
print(f"Sales: {sales}")

# Calculate growth
growth = np.diff(sales)  # Difference between consecutive elements
print(f"Monthly growth: {growth}")

# Find best and worst months
best_month_idx = np.argmax(sales)
worst_month_idx = np.argmin(sales)
print(f"Best month: {months[best_month_idx]} (${sales[best_month_idx]})")
print(f"Worst month: {months[worst_month_idx]} (${sales[worst_month_idx]})")

# Sales above average
avg_sales = np.mean(sales)
above_average = sales > avg_sales
print(f"Average sales: ${avg_sales:.0f}")
print(f"Months above average: {months[above_average]}")

print("\n7.4 Image Processing Simulation:")
# Simulate a small grayscale image (5x5 pixels)
image = np.random.randint(0, 256, size=(5, 5))
print("Original image (pixel values):")
print(image)

# Apply simple operations
brightened = np.clip(image + 50, 0, 255)  # Brighten and clip values
darkened = np.clip(image - 30, 0, 255)   # Darken and clip values

print("Brightened image:")
print(brightened)
print("Darkened image:")
print(darkened)

# =====================================
# WHY USE NUMPY?
# =====================================
print("\n\n8. WHY USE NUMPY?")
print("-" * 30)
print("""
NUMPY ADVANTAGES:
1. Speed: NumPy operations are much faster than pure Python
2. Memory efficient: Arrays use less memory than Python lists
3. Vectorization: Apply operations to entire arrays at once
4. Broadcasting: Operations between arrays of different shapes
5. Rich ecosystem: Foundation for pandas, scikit-learn, matplotlib

WHEN TO USE NUMPY:
- Numerical computations
- Array operations
- Scientific computing
- Data preprocessing
- Mathematical operations on large datasets
""")

# Simple performance demonstration
print("\n8.1 Performance Example:")
import time

# Python list operation
python_list = list(range(1000000))
start_time = time.time()
python_result = [x * 2 for x in python_list]
python_time = time.time() - start_time

# NumPy operation
numpy_array = np.arange(1000000)
start_time = time.time()
numpy_result = numpy_array * 2
numpy_time = time.time() - start_time

print(f"Python list time: {python_time:.4f} seconds")
print(f"NumPy array time: {numpy_time:.4f} seconds")
if numpy_time > 0:
    print(f"NumPy is {python_time/numpy_time:.1f}x faster!")
else:
    print("NumPy operation was too fast to measure accurately!")

print("\n" + "=" * 50)
print("End of NumPy Tutorial")
print("=" * 50)