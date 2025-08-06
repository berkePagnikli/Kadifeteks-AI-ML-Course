# 1. BASIC FUNCTION DEFINITION
def greet():
    """A simple function that prints a greeting"""
    print("Hello, World!")

# Calling the function
greet()

# 2. FUNCTIONS WITH PARAMETERS
def greet_person(name):
    """Function with one parameter"""
    print(f"Hello, {name}!")

def add_numbers(a, b):
    """Function with multiple parameters"""
    result = a + b
    print(f"{a} + {b} = {result}")

# Calling functions with arguments
print("Calling functions with parameters:")
greet_person("Alice")
greet_person("Bob")
add_numbers(5, 3)
add_numbers(10, 15)
print()

# 3. FUNCTIONS WITH RETURN VALUES
print("3. FUNCTIONS WITH RETURN VALUES")
print("-" * 35)

def multiply(x, y):
    """Function that returns a value"""
    return x * y

def get_full_name(first_name, last_name):
    """Function that returns a string"""
    full_name = f"{first_name} {last_name}"
    return full_name

# Using return values
print("Using return values:")
result = multiply(4, 7)
print(f"multiply(4, 7) = {result}")

name = get_full_name("John", "Doe")
print(f"Full name: {name}")

# You can use return values directly
print(f"Direct use: {multiply(3, 9)}")
print()

# 4. DEFAULT PARAMETERS
def introduce(name, age=25, city="Unknown"):
    """Function with default parameter values"""
    print(f"Hi, I'm {name}, {age} years old, from {city}")

# Calling with different numbers of arguments
print("Calling with default parameters:")
introduce("Alice")  # Uses default age and city
introduce("Bob", 30)  # Uses default city
introduce("Charlie", 35, "New York")  # No defaults used
introduce("Diana", city="London")  # Using keyword argument
print()

# 5. KEYWORD ARGUMENTS
def create_profile(name, age, profession, hobbies="Reading"):
    """Function demonstrating keyword arguments"""
    profile = f"Name: {name}, Age: {age}, Job: {profession}, Hobbies: {hobbies}"
    return profile

# Different ways to call the function
print("Using keyword arguments:")
profile1 = create_profile("Alice", 28, "Engineer")
profile2 = create_profile(age=32, name="Bob", profession="Teacher", hobbies="Sports")
profile3 = create_profile("Charlie", profession="Doctor", age=45)

print(profile1)
print(profile2)
print(profile3)
print()