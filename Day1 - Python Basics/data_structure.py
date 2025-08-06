# 1. LISTS

# Creating lists
fruits = ["apple", "banana", "orange"]
numbers = [1, 2, 3, 4, 5]
mixed_list = ["hello", 42, True, 3.14]

print(f"Fruits: {fruits}")
print(f"Numbers: {numbers}")
print(f"Mixed list: {mixed_list}")

# Accessing elements (indexing)
print(f"First fruit: {fruits[0]}")
print(f"Last fruit: {fruits[-1]}")

# APPEND
fruits.append("grape")
print(f"After append: {fruits}")

# INSERT
fruits.insert(1, "mango")
print(f"After insert: {fruits}")

# REMOVE
fruits.remove("banana")
print(f"After remove: {fruits}")

# POP
popped_fruit = fruits.pop()
print(f"Popped fruit: {popped_fruit}, Remaining: {fruits}")

# LENGTH
print(f"Length of fruits: {len(fruits)}")

# CHECK
print(f"Is 'apple' in fruits? {'apple' in fruits}")
print()

# 2. TUPLES

# Creating tuples
coordinates = (10, 20)

print(f"Coordinates: {coordinates}")

# Accessing elements
print(f"X coordinate: {coordinates[0]}")
print(f"Y coordinate: {coordinates[1]}")

# Tuple unpacking
x, y = coordinates

# Tuples are immutable - this would cause an error:
# coordinates[0] = 15  # TypeError!

# 3. DICTIONARIES

# Creating dictionaries
student = {
    "name": "Alice",
    "age": 20,
    "grade": "A",
    "subjects": ["Math", "Science", "English"]
}


print(f"Student: {student}")

# Using get() method (safer)
print(f"Student GPA: {student.get('gpa', 'Not available')}")

# Adding/updating values
student["gpa"] = 3.8
student["age"] = 21
print(f"Updated student: {student}")

# Dictionary methods
print(f"Dictionary keys: {list(student.keys())}")
print(f"Dictionary values: {list(student.values())}")
print(f"Dictionary items: {list(student.items())}")
print()