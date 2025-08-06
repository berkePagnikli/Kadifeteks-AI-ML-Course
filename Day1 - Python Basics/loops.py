# 1. FOR LOOP - Iterate over sequences

# Loop through a list
fruits = ["apple", "banana", "orange", "grape"]
print("Fruits in our basket:")
for fruit in fruits:
    print(f"  - {fruit}")
print()

# Loop through a string
word = "Python"
print(f"Letters in '{word}':")
for letter in word:
    print(f"  {letter}")
print()

# 2. FOR LOOP WITH RANGE

# range(stop)
print("range(5):")
for i in range(5):
    print(f"  {i}")
print()

# range(start, stop)
print("range(2, 8):")
for i in range(2, 8):
    print(f"  {i}")
print()

# range(start, stop, step)
print("range(0, 10, 2) - even numbers:")
for i in range(0, 10, 2):
    print(f"  {i}")
print()

print("range(10, 0, -2) - countdown:")
for i in range(10, 0, -2):
    print(f"  {i}")
print()

# 3. FOR LOOP WITH ENUMERATE
colors = ["red", "green", "blue", "yellow"]
print("Colors with their index:")
for index, color in enumerate(colors):
    print(f"  Index {index}: {color}")
print()

# Starting enumerate from different number
print("Colors with index starting from 1:")
for index, color in enumerate(colors, 1):
    print(f"  Color {index}: {color}")
print()

# 4. WHILE LOOP - Repeat while condition is True
# Basic while loop
count = 1
print("Counting to 5:")
while count <= 5:
    print(f"  Count: {count}")
    count += 1  # Don't forget to update the variable!
print()