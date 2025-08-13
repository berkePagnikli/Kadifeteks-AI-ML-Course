import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
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
target = 'İhtiyaç Kg'

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
# For example: 'Red' → [1,0,0], 'Blue' → [0,1,0], 'Green' → [0,0,1]
X_encoded = encoder.fit_transform(X)

# Split the encoded data into training (80%) and testing (20%) sets
# random_state=42 ensures reproducible results across runs
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Initialize Decision Tree Regressor with specific hyperparameters
model = DecisionTreeRegressor(
    max_depth=15,         # Maximum depth of the tree to control overfitting
    min_samples_split=8,  # Minimum samples required to split an internal node
    min_samples_leaf=4,   # Minimum samples required to be at a leaf node
    random_state=42       # Random seed for reproducibility
)

# Train the model on the one-hot encoded training data
# The decision tree will learn to split based on binary features
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics to assess model performance
mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error - measures prediction accuracy
r2 = r2_score(y_test, y_pred)             # R-squared - measures how well the model explains variance

# Display model performance metrics
print(f"Test MSE: {mse:.2f}")  # Lower MSE indicates better accuracy
print(f"Test R2: {r2:.2f}")    # Higher R2 (closer to 1) indicates better fit