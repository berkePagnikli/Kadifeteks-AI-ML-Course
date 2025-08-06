# =====================================
# PART 1: MATH MODULE
# =====================================
import math

print("\n1.1 Basic Math Functions:")
print(f"Square root of 16: {math.sqrt(16)}")
print(f"Power (2^3): {math.pow(2, 3)}")
print(f"Absolute value of -5: {math.fabs(-5)}")
print(f"Ceiling of 4.3: {math.ceil(4.3)}")
print(f"Floor of 4.7: {math.floor(4.7)}")

print("\n1.2 Trigonometric Functions:")
print(f"Sin(π/2): {math.sin(math.pi/2)}")
print(f"Cos(0): {math.cos(0)}")
print(f"Tan(π/4): {math.tan(math.pi/4)}")

print("\n1.3 Logarithmic Functions:")
print(f"Natural log of e: {math.log(math.e)}")
print(f"Log base 10 of 100: {math.log10(100)}")
print(f"Log base 2 of 8: {math.log2(8)}")

print("\n1.4 Constants:")
print(f"π (pi): {math.pi}")
print(f"e: {math.e}")
print(f"τ (tau): {math.tau}")

# =====================================
# PART 2: RANDOM MODULE
# =====================================

import random

# Set seed for reproducible results
random.seed(42)

print("\n2.1 Basic Random Functions:")
print(f"Random float (0-1): {random.random()}")
print(f"Random integer (1-10): {random.randint(1, 10)}")
print(f"Random choice from list: {random.choice(['apple', 'banana', 'orange'])}")

print("\n2.2 Random Sequences:")
numbers = [1, 2, 3, 4, 5]
random.shuffle(numbers)
print(f"Shuffled list: {numbers}")

sample_list = random.sample(range(1, 21), 5)
print(f"Random sample of 5 from 1-20: {sample_list}")

print("\n2.3 Random with Different Distributions:")
print(f"Random uniform (1-10): {random.uniform(1, 10)}")
print(f"Random normal distribution: {random.gauss(0, 1)}")

# =====================================
# PART 3: DATETIME MODULE
# =====================================
import datetime

print("\n3.1 Current Date and Time:")
now = datetime.datetime.now()
today = datetime.date.today()
print(f"Current datetime: {now}")
print(f"Current date: {today}")
print(f"Current time: {now.time()}")

print("\n3.2 Creating Specific Dates:")
specific_date = datetime.date(2024, 12, 25)
specific_datetime = datetime.datetime(2024, 12, 25, 15, 30, 0)
print(f"Christmas 2024: {specific_date}")
print(f"Christmas afternoon: {specific_datetime}")

print("\n3.3 Date Formatting:")
print(f"Formatted date: {now.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Readable format: {now.strftime('%B %d, %Y at %I:%M %p')}")

print("\n3.4 Date Arithmetic:")
tomorrow = today + datetime.timedelta(days=1)
last_week = today - datetime.timedelta(weeks=1)
print(f"Tomorrow: {tomorrow}")
print(f"Last week: {last_week}")

# Calculate age
birth_date = datetime.date(1990, 5, 15)
age = today - birth_date
print(f"Days since birth (May 15, 1990): {age.days}")

# =====================================
# PART 4: OS MODULE
# =====================================
import os

print("\n4.1 Directory Operations:")
current_dir = os.getcwd()
print(f"Current directory: {current_dir}")

# List files in current directory
files = os.listdir('.')
print(f"Files in current directory: {files[:5]}...")  # Show first 5

print("\n4.2 Path Operations:")
file_path = "example.txt"
print(f"Absolute path: {os.path.abspath(file_path)}")
print(f"File exists: {os.path.exists(file_path)}")
print(f"Is directory: {os.path.isdir('.')}")
print(f"Is file: {os.path.isfile(__file__)}")

print("\n4.3 Environment Variables:")
# Get common environment variables
user = os.environ.get('USERNAME', 'Unknown')  # Windows
if user == 'Unknown':
    user = os.environ.get('USER', 'Unknown')  # Unix/Linux
print(f"Current user: {user}")

# =====================================
# PART 5: JSON MODULE
# =====================================

import json

print("\n5.1 Python to JSON:")
data = {
    "name": "Alice",
    "age": 30,
    "city": "New York",
    "hobbies": ["reading", "coding", "hiking"]
}

json_string = json.dumps(data, indent=2)
print("Python dict to JSON:")
print(json_string)

print("\n5.2 JSON to Python:")
json_data = '{"product": "laptop", "price": 999, "in_stock": true}'
parsed_data = json.loads(json_data)
print(f"Parsed JSON: {parsed_data}")
print(f"Product: {parsed_data['product']}")
print(f"Price: ${parsed_data['price']}")

# =====================================
# PART 6: STRING MODULE (not that important)
# =====================================

import string

print("\n6.1 String Constants:")
print(f"ASCII letters: {string.ascii_letters}")
print(f"Digits: {string.digits}")
print(f"Punctuation: {string.punctuation}")

print("\n6.2 Password Generation Example:")
import random
password_chars = string.ascii_letters + string.digits + "!@#$%"
password = ''.join(random.choice(password_chars) for _ in range(8))
print(f"Random password: {password}")

print("\n6.3 String Templates:")
template = string.Template("Hello, $name! Welcome to $place.")
message = template.substitute(name="Alice", place="Python Course")
print(f"Template message: {message}")

# =====================================
# PART 7: COLLECTIONS MODULE
# =====================================

from collections import Counter, defaultdict, namedtuple

print("\n7.1 Counter - Count Elements:")
text = "hello world"
letter_count = Counter(text)
print(f"Letter count in '{text}': {letter_count}")

colors = ['red', 'blue', 'red', 'green', 'blue', 'red']
color_count = Counter(colors)
print(f"Most common color: {color_count.most_common(1)}")

print("\n7.2 defaultdict - Default Values:")
dd = defaultdict(list)
dd['fruits'].append('apple')
dd['fruits'].append('banana')
dd['vegetables'].append('carrot')
print(f"defaultdict: {dict(dd)}")

print("\n7.3 namedtuple - Named Tuples:")
Point = namedtuple('Point', ['x', 'y'])
p1 = Point(3, 4)
print(f"Point: {p1}")
print(f"X coordinate: {p1.x}")
print(f"Y coordinate: {p1.y}")

# =====================================
# PART 8: ITERTOOLS MODULE (not that important)
# =====================================
import itertools

print("\n8.1 Infinite Iterators:")
# Take first 5 from infinite counter
counter = itertools.count(1, 2)  # Start at 1, step by 2
first_five_odd = [next(counter) for _ in range(5)]
print(f"First 5 odd numbers: {first_five_odd}")

print("\n8.2 Combinatorial Iterators:")
colors = ['red', 'green', 'blue']
combinations = list(itertools.combinations(colors, 2))
print(f"Combinations of 2 colors: {combinations}")

permutations = list(itertools.permutations(['A', 'B', 'C'], 2))
print(f"Permutations of 2 letters: {permutations}")

# =====================================
# PART 9: TIME MODULE
# =====================================

import time

print("\n9.1 Time Functions:")
current_time = time.time()
print(f"Current timestamp: {current_time}")

formatted_time = time.strftime("%Y-%m-%d %H:%M:%S")
print(f"Formatted time: {formatted_time}")

print("\n9.2 Sleep Function:")
print("Waiting 1 second...")
time.sleep(1)
print("Done waiting!")

print("\n9.3 Performance Timing:")
start_time = time.time()
# Simulate some work
sum([x**2 for x in range(10000)])
end_time = time.time()
print(f"Operation took: {end_time - start_time:.4f} seconds")