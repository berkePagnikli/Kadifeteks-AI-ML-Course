# 1. BASIC IF STATEMENT
age = 18
print(f"Age: {age}")

if age >= 18:
    print("You are an adult!")
print()

# 2. IF-ELSE STATEMENT
temperature = 25
print(f"Temperature: {temperature}°C")

if temperature > 30:
    print("It's hot outside!")
else:
    print("The weather is nice!")
print()

# 3. IF-ELIF-ELSE STATEMENT
grade = 85
print(f"Grade: {grade}")

if grade >= 90:
    print("Excellent! Grade: A")
elif grade >= 80:
    print("Great! Grade: B")
elif grade >= 70:
    print("Good! Grade: C")
elif grade >= 60:
    print("Pass! Grade: D")
else:
    print("Fail! Grade: F")
print()

# 4. NESTED IF STATEMENTS

weather = "sunny"
temperature = 28
print(f"Weather: {weather}, Temperature: {temperature}°C")

if weather == "sunny":
    if temperature > 25:
        print("Perfect day for a picnic!")
    else:
        print("Sunny but a bit cool.")
else:
    print("Maybe stay inside today.")
print()

# 5. MULTIPLE CONDITIONS WITH AND/OR
is_weekend = True
is_raining = False
print(f"Is weekend: {is_weekend}, Is raining: {is_raining}")

# Using AND
if is_weekend and not is_raining:
    print("Great day to go out!")

# Using OR
if is_weekend or not is_raining:
    print("Either it's weekend or not raining (or both)!")
print()

# 6. CONDITIONAL EXPRESSIONS (TERNARY OPERATOR)

score = 75
print(f"Score: {score}")

# Ternary operator: value_if_true if condition else value_if_false
result = "Pass" if score >= 60 else "Fail"
print(f"Result: {result}")

# Another example
number = 10
status = "Even" if number % 2 == 0 else "Odd"
print(f"Number {number} is {status}")
print()