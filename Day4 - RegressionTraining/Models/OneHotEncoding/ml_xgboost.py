import pandas as pd
import numpy as np  
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import OneHotEncoder 
from sklearn.metrics import mean_squared_error, r2_score 
import xgboost as xgb 

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
# For example: if we have colors ['Red', 'Blue', 'Green'], each becomes [1,0,0], [0,1,0], [0,0,1]
X_encoded = encoder.fit_transform(X)

# Split the encoded data into training (80%) and testing (20%) sets
# random_state=42 ensures reproducible results across runs
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Initialize XGBoost Regressor with specific hyperparameters
# Note: XGBoost works well with one-hot encoded features, though it can handle categorical features natively
model = xgb.XGBRegressor(
    n_estimators=100,     # Number of boosting rounds (trees to build)
    max_depth=6,          # Maximum depth of each tree to control complexity
    learning_rate=0.1,    # Step size shrinkage to prevent overfitting
    random_state=42       # Random seed for reproducibility
)

# Train the model on the one-hot encoded training data
# XGBoost will build an ensemble of decision trees using the binary features
model.fit(X_train, y_train)

# Make predictions on the test set
# XGBoost combines predictions from all trees to make final predictions
y_pred = model.predict(X_test)

# Calculate evaluation metrics to assess model performance
mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
r2 = r2_score(y_test, y_pred)             # R-squared - measures how well the model explains variance

# Display model performance metrics
print(f"Test MSE: {mse:.2f}")  # Lower MSE indicates better accuracy
print(f"Test R2: {r2:.2f}")    # Higher R2 (closer to 1) indicates better fit
