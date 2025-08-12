import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score 
import pickle

# Load the preprocessed dataset from CSV file
df = pd.read_csv('manipulated_data.csv')

def get_season(date_str):
    """
    Extract season from date string in YYYY-MM-DD format
    This function converts date strings into categorical season features
    which can provide temporal context for fabric demand prediction
    """
    try:
        month = int(date_str.split('-')[1])
        # Map months to seasons based on meteorological seasons
        if month in [12, 1, 2]:  # December, January, February
            return 'Winter'
        elif month in [3, 4, 5]:  # March, April, May
            return 'Spring'
        elif month in [6, 7, 8]:  # June, July, August
            return 'Summer'
        else:  # September, October, November
            return 'Fall'
    except:
        return 'Unknown'

# Create a new feature 'Season' by applying the season extraction function to the 'Tarih' column
df['Season'] = df['Tarih'].apply(get_season)

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
categorical_features = [col for col in df.columns if col != target and df[col].dtype == 'object']

# Separate features (X) and target variable (y)
X = df.drop(target, axis=1)  # All columns except the target
y = df[target]               # Only the target column

# Initialize dictionary to store label encoders for each categorical feature
# We need to save these encoders to apply the same transformation during inference
label_encoders = {}
X_encoded = X.copy()  # Create a copy of features to avoid modifying original data

# Apply label encoding to each categorical feature
# Label encoding converts categorical values to numeric integers (e.g., 'Red'→0, 'Blue'→1, 'Green'→2)
for col in categorical_features:
    le = LabelEncoder()  # Create a new label encoder for this column
    # Fit the encoder and transform the column values to integers
    X_encoded[col] = le.fit_transform(X[col].astype(str))
    # Store the encoder for future use during inference
    label_encoders[col] = le

# Create polynomial features to capture non-linear relationships
# PolynomialFeatures generates polynomial and interaction features
poly = PolynomialFeatures(
    degree=2,           # Create features up to degree 2 (x₁², x₁x₂, x₂², etc.)
    include_bias=False  # Don't include the bias column (constant term)
)

# Transform the encoded features to polynomial features
# This creates new features like x₁², x₂², x₁x₂ for degree=2
X_poly = poly.fit_transform(X_encoded)

# Split the polynomial-transformed data into training (80%) and testing (20%) sets
# random_state=42 ensures reproducible results across runs
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Initialize Linear Regression model
# Even though we're using linear regression, the polynomial features allow us to model non-linear relationships
model = LinearRegression()

# Train the model on the polynomial-transformed training data
# The model will learn coefficients for original features, squared features, and interaction terms
model.fit(X_train, y_train)

# Make predictions on the test set
# The model uses the polynomial features to make more complex predictions
y_pred = model.predict(X_test)

# Calculate evaluation metrics to assess model performance
mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error - measures prediction accuracy
r2 = r2_score(y_test, y_pred)             # R-squared - measures how well the model explains variance

# Display model performance metrics
print(f"Test MSE: {mse:.2f}")  # Lower MSE indicates better accuracy
print(f"Test R2: {r2:.2f}")    # Higher R2 (closer to 1) indicates better fit

# Save the trained model using pickle
with open('ModelWeights/polynomial_regressor.pkl', 'wb') as f:
    pickle.dump(model, f)

# Create metadata dictionary containing important information for model deployment
model_components = {
    'polynomial_features': poly,             # PolynomialFeatures transformer needed for new data
    'label_encoders': label_encoders,        # Encoders needed for preprocessing new data
    'categorical_features': categorical_features,  # List of categorical feature names
    'feature_names': list(X.columns),       # Original feature names before polynomial transformation
    'target_name': target,                   # Target variable name
    'model_type': 'PolynomialRegression',    # Model type identifier
    'polynomial_degree': 2                   # Degree of polynomial features used
}

# Save model components and metadata using pickle for future inference
with open('ModelWeights/polynomial_regressor_components.pkl', 'wb') as f:
    pickle.dump(model_components, f)