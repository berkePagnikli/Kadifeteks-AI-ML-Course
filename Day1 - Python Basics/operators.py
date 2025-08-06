# 1. ARITHMETIC OPERATORS

# Basic arithmetic
a = 10
b = 3

print(f"a = {a}, b = {b}")
print(f"Addition: {a} + {b} = {a + b}")
print(f"Subtraction: {a} - {b} = {a - b}")
print(f"Multiplication: {a} * {b} = {a * b}")
print(f"Division: {a} / {b} = {a / b}")
print(f"Floor Division: {a} // {b} = {a // b}")
print(f"Modulus (remainder): {a} % {b} = {a % b}")
print(f"Exponent: {a} ** {b} = {a ** b}")
print()

# 2. COMPARISON OPERATORS

x = 5
y = 8

print(f"x = {x}, y = {y}")
print(f"Equal to: {x} == {y} is {x == y}")
print(f"Not equal to: {x} != {y} is {x != y}")
print(f"Greater than: {x} > {y} is {x > y}")
print(f"Less than: {x} < {y} is {x < y}")
print(f"Greater than or equal: {x} >= {y} is {x >= y}")
print(f"Less than or equal: {x} <= {y} is {x <= y}")
print()

# 3. LOGICAL OPERATORS
is_sunny = True
is_warm = False

print(f"is_sunny = {is_sunny}, is_warm = {is_warm}")
print(f"AND: is_sunny and is_warm = {is_sunny and is_warm}")
print(f"OR: is_sunny or is_warm = {is_sunny or is_warm}")
print(f"NOT: not is_sunny = {not is_sunny}")
print()

# 4. ASSIGNMENT OPERATORS

number = 10
print(f"Initial value: number = {number}")

number += 5  # Same as: number = number + 5
print(f"After += 5: number = {number}")

number -= 3  # Same as: number = number - 3
print(f"After -= 3: number = {number}")

number *= 2  # Same as: number = number * 2
print(f"After *= 2: number = {number}")

number /= 4  # Same as: number = number / 4
print(f"After /= 4: number = {number}")
print()