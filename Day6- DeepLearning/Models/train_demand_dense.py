import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import r2_score
import tensorflow as tf
import matplotlib.pyplot as plt

# Read the processed dataset containing features and the demand target.
data = pd.read_csv('manipulated_data.csv')

# If a date field exists, engineer time-based features that help capture seasonality.
if 'Tarih' in data.columns:
    # Parse dates robustly; any invalid entries become NaT to prevent failures.
    data['Tarih'] = pd.to_datetime(data['Tarih'], errors='coerce')
    # Extract month/day components as nullable ints to handle missing dates gracefully.
    data['month'] = data['Tarih'].dt.month.astype('Int64')
    data['day'] = data['Tarih'].dt.day.astype('Int64')

    # Convert for numeric transforms below.
    month = data['month'].astype(int)

    day = data['day'].astype(int)
    # Cyclical encodings ensure the network understands periodicity (Dec close to Jan etc.).
    data['month_sin'] = np.sin(2 * np.pi * (month - 1) / 12)
    data['month_cos'] = np.cos(2 * np.pi * (month - 1) / 12)
    data['day_sin'] = np.sin(2 * np.pi * (day - 1) / 31)
    data['day_cos'] = np.cos(2 * np.pi * (day - 1) / 31)

    # Map month to season, adding a compact categorical seasonal signal.
    def month_to_season(m):
        if m in (12, 1, 2):
            return 'Winter'
        if m in (3, 4, 5):
            return 'Spring'
        if m in (6, 7, 8):
            return 'Summer'
        return 'Autumn'

    data['season'] = month.map(month_to_season)

    # Drop the original datetime column to keep only numeric/categorical features for modeling.
    data = data.drop(columns=['Tarih'])

# Remove a likely identifier column that doesn't carry predictive signal.
data = data.drop(columns=['İş Emri No'])

# Select the target to predict (demand in kilograms).
target_col = 'İhtiyaç Kg'

# Log-transform the target to reduce skew/outlier effects and improve regression stability.
data[target_col + '_log'] = np.log1p(data[target_col])
target_col_transformed = target_col + '_log'

# Build features by excluding both raw and transformed target columns.
X_df = data.drop(columns=[target_col, target_col_transformed]).copy()
# Extract transformed target values as float array.
y = data[target_col_transformed].values.astype(float)


# Encode any non-numeric features to integers so the dense network can consume them.
encoders = {}
for col in X_df.columns:
    if X_df[col].dtype == 'object' or str(X_df[col].dtype).startswith('category'):
        le = LabelEncoder()
        # Standardize dtype and handle missing values before label encoding.
        X_df[col] = X_df[col].astype(str).fillna('')
        X_df[col] = le.fit_transform(X_df[col])
        encoders[col] = le

# Convert to a 2D float array that scalers and Keras expect.
X = X_df.values.astype(float)

# Use RobustScaler to lessen outlier influence on scale normalization.
feature_scaler = RobustScaler()
X = feature_scaler.fit_transform(X)

# Constrain extreme values to a bounded range to stabilize optimization.
X = np.clip(X, -10, 10)
y = np.clip(y, 0, 10)

# Split the dataset into training and testing to evaluate generalization.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a straightforward fully-connected regression model with regularization and dropout.
# Architecture notes:
# - Dense(256) with L2(0.001): L2 adds 0.001 * sum(w^2) to loss, encouraging smaller weights (weight shrinkage)
#   to reduce overfitting. 256 units provide decent capacity for tabular regression.
# - LeakyReLU avoids dying ReLU issues by allowing a small negative slope (default alpha=0.3 in Keras layer)
#   so gradients flow for negative inputs, improving learning stability.
# - Dropout(0.2) randomly zeros 20% of activations to improve generalization.
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, input_dim=X_train.shape[1], kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dropout(0.2),

    # Output layer: linear activation for regression on log-transformed target.
    tf.keras.layers.Dense(1)
])

# Compile with Adam optimizer and MSE loss in the log space; MAE monitored for interpretability.
# Adam optimizer with lr=1e-3 is a strong default for dense nets on tabular data.
# Mean Squared Error on the log-transformed target penalizes larger errors more strongly.
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

# Early stopping halts training when validation stops improving; LR reduction helps escape plateaus.
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, monitor='val_loss'),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)
]

# Train the model; keep 20% of the training data aside for validation feedback.
history = model.fit(X_train, y_train, 
                   epochs=150, 
                   batch_size=64,
                   validation_split=0.2, 
                   callbacks=callbacks,
                   verbose=1)

# Evaluate test loss (in log space) for an unbiased performance measure.
test_results = model.evaluate(X_test, y_test, verbose=0)
test_loss = test_results[0] if isinstance(test_results, list) else test_results


# Generate predictions on test set (still in log space), then invert to original units for reporting.
y_pred_log = model.predict(X_test, verbose=0)
y_pred_original = np.expm1(y_pred_log.flatten())
y_test_original = np.expm1(y_test)

# Compute error metrics in original target units (kilograms) to make them meaningful to stakeholders.
mse_original = np.mean((y_test_original - y_pred_original)**2)
mae_original = np.mean(np.abs(y_test_original - y_pred_original))

print(f'\nModel Performance:')
print(f'Log-scale Test MSE: {test_loss:.4f}')
print(f'Original-scale Test MSE: {mse_original:.2f}')
print(f'Original-scale Test MAE: {mae_original:.2f}')
print(f'Original-scale Test R2: {r2_score(y_test_original, y_pred_original):.4f}')

print(f'\nSample Predictions vs Actual (original scale):')
for i in range(min(10, len(y_test_original))):
    print(f'Predicted: {y_pred_original[i]:.2f}, Actual: {y_test_original[i]:.2f}')

plt.figure(figsize=(6,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()
