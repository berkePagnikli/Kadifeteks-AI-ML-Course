import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import r2_score
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the dataset used for training and evaluation.
data = pd.read_csv('manipulated_data.csv')

# If a date column exists, engineer relevant cyclical and seasonal features.
if 'Tarih' in data.columns:
    # Parse dates while tolerating invalid entries.
    data['Tarih'] = pd.to_datetime(data['Tarih'], errors='coerce')
    # Extract month and day to capture intra-year and intra-month patterns.
    data['month'] = data['Tarih'].dt.month.astype('Int64')
    data['day'] = data['Tarih'].dt.day.astype('Int64')

    # Prepare numeric arrays for the trigonometric transforms.
    month = data['month'].astype(int)

    day = data['day'].astype(int)
    # Cyclical encoding for periodic variables so the model understands wrap-around.
    data['month_sin'] = np.sin(2 * np.pi * (month - 1) / 12)
    data['month_cos'] = np.cos(2 * np.pi * (month - 1) / 12)
    data['day_sin'] = np.sin(2 * np.pi * (day - 1) / 31)
    data['day_cos'] = np.cos(2 * np.pi * (day - 1) / 31)

    # Seasonal bucket feature to encode coarse time-of-year effects.
    def month_to_season(m):
        if m in (12, 1, 2):
            return 'Winter'
        if m in (3, 4, 5):
            return 'Spring'
        if m in (6, 7, 8):
            return 'Summer'
        return 'Autumn'

    data['season'] = month.map(month_to_season)

    # Remove raw datetime; use engineered features instead.
    data = data.drop(columns=['Tarih'])

# Drop identifier-like column as it carries no predictive signal for regression.
data = data.drop(columns=['İş Emri No'])

# Define the target variable name.
target_col = 'İhtiyaç Kg'

# Apply log transform to stabilize variance and mitigate skew.
data[target_col + '_log'] = np.log1p(data[target_col])
target_col_transformed = target_col + '_log'

# Create feature matrix by excluding the target columns.
X_df = data.drop(columns=[target_col, target_col_transformed]).copy()
# Extract transformed target as float array for Keras.
y = data[target_col_transformed].values.astype(float)

# Encode categorical columns so sequence model receives numeric inputs only.
encoders = {}
for col in X_df.columns:
    if X_df[col].dtype == 'object' or str(X_df[col].dtype).startswith('category'):
        le = LabelEncoder()
        X_df[col] = X_df[col].astype(str).fillna('')
        X_df[col] = le.fit_transform(X_df[col])
        encoders[col] = le

# Convert to float matrix and scale with a method robust to outliers.
X = X_df.values.astype(float)

feature_scaler = RobustScaler()
X = feature_scaler.fit_transform(X)

# Clip extreme values to bounded ranges to improve training stability.
X = np.clip(X, -10, 10)
y = np.clip(y, 0, 10)

# Create fixed-length sequences out of tabular rows so the LSTM can capture short-term dynamics.
def create_sequences(X, y, seq_length=5):
    """Create sequences for LSTM from tabular data"""
    # If dataset is shorter than the window, tile it to create a minimal sequence input.
    if len(X) < seq_length:
        X_seq = np.repeat(X[np.newaxis, :, :], seq_length, axis=1)
        return X_seq, y
    
    X_seq = []
    y_seq = []
    
    for i in range(seq_length, len(X)):
        # Use a sliding window of past 'seq_length' rows to predict the current row's target.
        X_seq.append(X[i-seq_length:i])
        y_seq.append(y[i])
    
    return np.array(X_seq), np.array(y_seq)

# Hold out a test split for unbiased evaluation.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Select a short sequence length, but avoid excessively long windows for small datasets.
seq_length = min(5, len(X_train) // 10)

# Materialize sequence tensors for train and test partitions.
X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_length)

# Build a two-layer LSTM followed by dense layers for regression on the log target.
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2,
                        kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2,
                        kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dense(64, activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])

# RMSprop often works well on recurrent nets; MSE on log scale aligns with the transformed target.
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
    loss='mean_squared_error',
    metrics=['mae']
)
# Notes:
# - LSTM units (128 -> 64): first returns full sequences to feed the second layer; second returns last state.
# - dropout (inputs) vs recurrent_dropout (state): both at 0.2 provide regularization; recurrent dropout
#   reduces overfitting of temporal dependencies but can slow training.
# - L2(0.001) applies weight decay strength 1e-3; balances bias-variance.
# - RMSprop is often a solid choice for RNNs due to adaptive learning per-parameter with momentum-like decay.
# - MSE on log targets matches the target transform; MAE tracked for interpretability.

# Early stopping and LR scheduling to improve generalization and avoid wasting epochs.
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, monitor='val_loss'),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10),
]

# Train the sequence model with validation feedback.
history = model.fit(X_train_seq, y_train_seq, 
                   epochs=200, 
                   batch_size=64,
                   validation_split=0.2, 
                   callbacks=callbacks,
                   verbose=1)

# Evaluate performance on the test sequences.
test_results = model.evaluate(X_test_seq, y_test_seq, verbose=0)
test_loss = test_results[0] if isinstance(test_results, list) else test_results

# Predict in log space then invert to original target units to compute human-readable errors.
y_pred_log = model.predict(X_test_seq, verbose=0)
y_pred_original = np.expm1(y_pred_log.flatten())
y_test_original = np.expm1(y_test_seq)

mse_original = np.mean((y_test_original - y_pred_original)**2)
mae_original = np.mean(np.abs(y_test_original - y_pred_original))

print(f'\nLSTM Model Performance:')
print(f'Log-scale Test MSE: {test_loss:.4f}')
print(f'Original-scale Test MSE: {mse_original:.2f}')
print(f'Original-scale Test MAE: {mae_original:.2f}')
print(f'Original-scale Test R2: {r2_score(y_test_original, y_pred_original):.4f}')


print(f'\nSample Predictions vs Actual (original scale):')
for i in range(min(10, len(y_test_original))):
    print(f'Predicted: {y_pred_original[i]:.2f}, Actual: {y_test_original[i]:.2f}')

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.title('LSTM Model - Loss')
plt.show()