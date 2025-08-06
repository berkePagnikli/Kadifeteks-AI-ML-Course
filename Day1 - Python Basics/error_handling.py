# 1. Basic try-except with division by zero
try:
    result = 10 / 0
    print(f"Result: {result}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")

# 2. File not found error
try:
    with open("nonexistent_file.txt", "r") as file:
        content = file.read()
        print(content)
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")

# 3. Index out of range error
try:
    my_list = [1, 2, 3, 4, 5]
    print(f"Accessing index 10: {my_list[10]}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")

# 4. Key error with dictionaries
try:
    my_dict = {"name": "John", "age": 30}
    print(f"Height: {my_dict['height']}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")

# 5. Type error - string and integer operations
try:
    result = "Hello" + 5
    print(f"Result: {result}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")

# 6. Value error - invalid conversion
try:
    number = int("not_a_number")
    print(f"Converted number: {number}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")

# 7. Attribute error
try:
    my_string = "Hello World"
    my_string.nonexistent_method()
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")

# 8. Try-except with else clause
try:
    result = 10 / 2
    print(f"Division successful: {result}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")
else:
    print("No error occurred - this is the else block")

# 9. Try-except with finally clause
try:
    result = 10 / 0
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")
finally:
    print("This finally block always executes")

# 10. Nested try-except blocks
try:
    print("Outer try block")
    try:
        print("Inner try block")
        result = 10 / 0
    except Exception as e:
        print(f"Inner except: {type(e).__name__}: {e}")
        # Attempting another operation that might fail
        my_list = [1, 2, 3]
        print(my_list[10])
except Exception as e:
    print(f"Outer except: {type(e).__name__}: {e}")

# 11. Function with error handling
def safe_divide(a, b):
    """
    Safely divide two numbers with error handling
    """
    try:
        result = a / b
        return result
    except Exception as e:
        print(f"Error in safe_divide: {type(e).__name__}: {e}")
        return None

# Test the function
result1 = safe_divide(10, 2)
print(f"10 / 2 = {result1}")

result2 = safe_divide(10, 0)
print(f"10 / 0 = {result2}")

# 12. User input with error handling
def get_user_number():
    """
    Get a number from user with error handling
    """
    try:
        # For demonstration, we'll simulate user input
        user_input = "abc"  # This would normally be input("Enter a number: ")
        number = float(user_input)
        return number
    except Exception as e:
        print(f"Error converting input to number: {type(e).__name__}: {e}")
        return None

number = get_user_number()
if number is not None:
    print(f"You entered: {number}")
else:
    print("Invalid input provided")

# 13. Multiple operations with error handling
def perform_operations():
    """
    Perform multiple operations that might fail
    """
    operations = [
        lambda: 10 / 2,           # Should succeed
        lambda: 10 / 0,           # Division by zero
        lambda: [1, 2][5],        # Index error
        lambda: int("hello"),     # Value error
    ]
    
    for i, operation in enumerate(operations, 1):
        try:
            result = operation()
            print(f"Operation {i}: Success - Result: {result}")
        except Exception as e:
            print(f"Operation {i}: Failed - {type(e).__name__}: {e}")

perform_operations()

# 14. Class with error handling

class Calculator:
    """
    Simple calculator class with error handling
    """
    
    def divide(self, a, b):
        try:
            result = a / b
            return result
        except Exception as e:
            print(f"Calculator error: {type(e).__name__}: {e}")
            return None
    
    def get_list_item(self, my_list, index):
        try:
            return my_list[index]
        except Exception as e:
            print(f"List access error: {type(e).__name__}: {e}")
            return None

# Test the calculator class
calc = Calculator()
print(f"Calculator: 20 / 4 = {calc.divide(20, 4)}")
print(f"Calculator: 20 / 0 = {calc.divide(20, 0)}")

my_list = [10, 20, 30, 40, 50]
print(f"List item at index 2: {calc.get_list_item(my_list, 2)}")
print(f"List item at index 10: {calc.get_list_item(my_list, 10)}")

print("\n" + "=" * 60)
print("ERROR HANDLING DEMONSTRATIONS COMPLETED")
print("=" * 60)

print("\nKey Points:")
print("• Always use 'except Exception as e' to catch and handle errors")
print("• Use try-except to prevent program crashes")
print("• The 'e' variable contains the error information")
print("• 'else' block runs when no exception occurs")
print("• 'finally' block always runs, regardless of exceptions")
print("• Error handling makes programs more robust and user-friendly")
