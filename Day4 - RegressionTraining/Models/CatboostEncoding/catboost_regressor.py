import pandas as pd  
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor
import pickle

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

# Define the target variable
target = 'İhtiyaç Kg'

# Separate features (X) and target variable (y)
X = df.drop(target, axis=1)  # All columns except the target
y = df[target]               # Only the target column

# Identify categorical features (non-numeric columns) in the feature set
# CatBoost can handle categorical features natively without manual encoding
categorical_features = [col for col in X.columns if X[col].dtype == 'object']

# Get the column indices of categorical features for CatBoost
# CatBoost requires categorical feature indices to know which columns to treat as categorical
categorical_feature_indices = [X.columns.get_loc(col) for col in categorical_features]

# Split the data into training (80%) and testing (20%) sets
# random_state=42 ensures reproducible results across runs
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize CatBoost Regressor with specific hyperparameters
model = CatBoostRegressor(
    iterations=100,                           # Number of boosting iterations (trees)
    depth=6,                                  # Maximum depth of each tree
    learning_rate=0.1,                        # Step size for gradient descent
    random_state=42,                          # Random seed for reproducibility
    cat_features=categorical_feature_indices, # Indices of categorical features
    verbose=False                             # Suppress training output
)

# Train the model on the training data
# CatBoost will automatically handle categorical features without preprocessing
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
r2 = r2_score(y_test, y_pred)             # R-squared - measures how well the model explains variance

# Display model performance metrics
print(f"Test MSE: {mse:.2f}")  # Lower MSE indicates better accuracy
print(f"Test R2: {r2:.2f}")    # Higher R2 (closer to 1) indicates better fit

# Save the trained model in CatBoost's native format (.cbm)
# This format preserves all CatBoost-specific features and optimizations
model.save_model('ModelWeights/catboost_regressor.cbm')

# Create metadata dictionary containing important information about the model
model_metadata = {
    'categorical_features': categorical_features,           # List of categorical feature names
    'categorical_feature_indices': categorical_feature_indices,  # Their column indices
    'feature_names': list(X.columns),                     # All feature names
    'target_name': target,                                 # Target variable name
    'model_type': 'CatBoostRegressor'                      # Model type identifier
}

# Save model metadata using pickle for future inference
with open('ModelWeights/catboost_regressor_metadata.pkl', 'wb') as f:
    pickle.dump(model_metadata, f)