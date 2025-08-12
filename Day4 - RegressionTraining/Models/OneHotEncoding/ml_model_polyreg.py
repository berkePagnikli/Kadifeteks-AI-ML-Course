import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the preprocessed dataset from CSV file
df = pd.read_csv('manipulated_data.csv')

# Remove irrelevant or identifier columns that don't contribute to prediction
df = df.drop([
    'İş Emri No',
    'Tarih',
    'Sipariş No',
    'Kumaş Kodu',   
    'Firma Ülkesi',  
    'Desen Adı', 
    'Varyant No',
    'Kalite Adı',  
    'Tezgah Kodu',
    'Firma Adı', 
    'Proses Kodu',
    'Çözgü Adı'  
], axis=1)

# Define the target variable (what we want to predict)
target = 'İhtiyaç Kg'  # Required fabric weight in kilograms

# Identify categorical features (string/object type columns) that need encoding
# Only select categorical features, excluding the target variable
categorical_features = [col for col in df.columns if col != target and df[col].dtype == 'object']

# Separate categorical features and target variable
X = df[categorical_features]  # Only categorical features for one-hot encoding
y = df[target]                # Target variable

# Initialize One-Hot Encoder with specific parameters
encoder = OneHotEncoder(
    sparse_output=False,      # Return dense array instead of sparse matrix for easier handling
    handle_unknown='ignore'   # Ignore unknown categories during transform (assign zeros to new categories)
)

# Apply one-hot encoding to categorical features
# This converts each categorical value to a binary vector representation
# For example: if we have colors ['Red', 'Blue', 'Green'], each becomes [1,0,0], [0,1,0], [0,0,1]
X_encoded = encoder.fit_transform(X)

# Create polynomial features from the one-hot encoded binary features
# This will create interaction terms between different categorical variables
poly = PolynomialFeatures(degree=2)  # Create features up to degree 2 (interactions between categories)

# Transform the one-hot encoded features to include polynomial terms
# This creates interaction features like: Category1_A × Category2_X, etc.
X_poly = poly.fit_transform(X_encoded)

# Split the polynomial-transformed data into training (80%) and testing (20%) sets
# random_state=42 ensures reproducible results across runs
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Initialize Linear Regression model
# Even though we use linear regression, polynomial features allow modeling of non-linear relationships
model = LinearRegression()

# Train the model on the polynomial-transformed data
# Model learns coefficients for original binary features plus their interactions
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics to assess model performance
mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error - measures prediction accuracy
r2 = r2_score(y_test, y_pred)             # R-squared - measures how well the model explains variance

# Display model performance metrics
print(f"Test MSE: {mse:.2f}")  # Lower MSE indicates better accuracy
print(f"Test R2: {r2:.2f}")    # Higher R2 (closer to 1) indicates better fit