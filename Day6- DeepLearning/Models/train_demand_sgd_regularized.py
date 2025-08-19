import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import r2_score
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the dataset that contains features and the demand target.
data = pd.read_csv('manipulated_data.csv')

# Create additional time-derived features if the date column exists.
if 'Tarih' in data.columns:
    data['Tarih'] = pd.to_datetime(data['Tarih'], errors='coerce')
    data['month'] = data['Tarih'].dt.month.astype('Int64')
    data['day'] = data['Tarih'].dt.day.astype('Int64')

    month = data['month'].astype(int)

    day = data['day'].astype(int)
    data['month_sin'] = np.sin(2 * np.pi * (month - 1) / 12)
    data['month_cos'] = np.cos(2 * np.pi * (month - 1) / 12)
    data['day_sin'] = np.sin(2 * np.pi * (day - 1) / 31)
    data['day_cos'] = np.cos(2 * np.pi * (day - 1) / 31)

    def month_to_season(m):
        if m in (12, 1, 2):
            return 'Winter'
        if m in (3, 4, 5):
            return 'Spring'
        if m in (6, 7, 8):
            return 'Summer'
        return 'Autumn'

    data['season'] = month.map(month_to_season)
    data = data.drop(columns=['Tarih'])

# Drop an ID-like field that wouldn't help prediction.
data = data.drop(columns=['İş Emri No'])

# Choose the target (demand) and make a log-transformed version for modeling stability.
target_col = 'İhtiyaç Kg'

data[target_col + '_log'] = np.log1p(data[target_col])
target_col_transformed = target_col + '_log'

# Build the feature matrix from all remaining columns except the target variants.
X_df = data.drop(columns=[target_col, target_col_transformed]).copy()
y = data[target_col_transformed].values.astype(float)

# Encode categorical columns into integers for neural network consumption.
encoders = {}
for col in X_df.columns:
    if X_df[col].dtype == 'object' or str(X_df[col].dtype).startswith('category'):
        le = LabelEncoder()
        X_df[col] = X_df[col].astype(str).fillna('')
        X_df[col] = le.fit_transform(X_df[col])
        encoders[col] = le

# Convert to a numeric array and scale robustly.
X = X_df.values.astype(float)

feature_scaler = RobustScaler()
X = feature_scaler.fit_transform(X)

# Clip large magnitude values to stabilize gradient-based training.
X = np.clip(X, -10, 10)
y = np.clip(y, 0, 10)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define an MLP with strong L1/L2 regularization plus batch-norm and dropout to reduce overfitting.
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(512, input_dim=X_train.shape[1], 
                         kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01),
                         bias_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.ReLU(),  # ReLU is simple and fast; paired with L1/L2 it can generalize well on tabular data.
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),  # 30% dropout is relatively strong; combats overfitting with large capacity.
    
    tf.keras.layers.Dense(256, 
                         kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.005, l2=0.005)),
    tf.keras.layers.ReLU(),  # Keep ReLU for consistency and speed; smaller layer reduces model capacity.
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Dense(128, 
                         kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.ReLU(),  # L2-only here; encourages smooth weight distributions in this narrower layer.
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.01))  # Linear output for regression.
])

# Use momentum SGD with Nesterov acceleration to explore a different optimizer regime than Adam.
sgd_optimizer = tf.keras.optimizers.SGD(
    learning_rate=0.01,
    momentum=0.9,
    nesterov=True  # Nesterov momentum looks ahead, often converging faster/more smoothly than classical momentum.
)

# Compile with Huber loss for robustness to outliers, and track MAE/MSE as metrics.
model.compile(
    optimizer=sgd_optimizer, 
    loss=tf.keras.losses.Huber(delta=1.0),  # Huber blends MAE and MSE; delta=1.0 sets the switch point.
    metrics=['mae', 'mse']
)

# Early stopping and LR scheduling to curb overfitting and improve convergence.
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, monitor='val_loss'),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10),
]

history = model.fit(X_train, y_train, 
                   epochs=200, 
                   batch_size=32,
                   validation_split=0.2, 
                   callbacks=callbacks,
                   verbose=1)

# Evaluate generalization on the test split in the transformed (log) space.
test_results = model.evaluate(X_test, y_test, verbose=0)
test_loss = test_results[0] if isinstance(test_results, list) else test_results

# Predict in log space and convert to original units for intuitive error reporting.
y_pred_log = model.predict(X_test, verbose=0)
y_pred_original = np.expm1(y_pred_log.flatten())
y_test_original = np.expm1(y_test)

mse_original = np.mean((y_test_original - y_pred_original)**2)
mae_original = np.mean(np.abs(y_test_original - y_pred_original))

print(f'\nSGD Model Performance:')
print(f'Log-scale Test Loss (Huber): {test_loss:.4f}')
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
plt.ylabel('Huber Loss')
plt.legend()
plt.title('SGD Model - Loss')
plt.show()
