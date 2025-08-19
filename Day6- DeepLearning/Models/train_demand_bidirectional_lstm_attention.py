import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import r2_score
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the preprocessed dataset from disk so we can train a model on it.
data = pd.read_csv('manipulated_data.csv')

# If there is a date column named 'Tarih', derive useful time-based features from it.
if 'Tarih' in data.columns:
    # Convert the text date into a proper datetime; coerce invalid strings to NaT instead of crashing.
    data['Tarih'] = pd.to_datetime(data['Tarih'], errors='coerce')
    # Extract month and day numbers as nullable integer type to safely handle missing dates.
    data['month'] = data['Tarih'].dt.month.astype('Int64')
    data['day'] = data['Tarih'].dt.day.astype('Int64')

    # Convert month/day to plain Python ints for use in numeric transforms below.
    month = data['month'].astype(int)

    day = data['day'].astype(int)
    # Encode month cyclically with sine/cosine so the model learns that Dec (12) is close to Jan (1).
    data['month_sin'] = np.sin(2 * np.pi * (month - 1) / 12)
    data['month_cos'] = np.cos(2 * np.pi * (month - 1) / 12)
    # Encode day-of-month cyclically for the same reason (1 is close to 31 on a cycle).
    data['day_sin'] = np.sin(2 * np.pi * (day - 1) / 31)
    data['day_cos'] = np.cos(2 * np.pi * (day - 1) / 31)

    # Helper to bucket months into seasons; this gives a lower-cardinality categorical time feature.
    def month_to_season(m):
        if m in (12, 1, 2):
            return 'Winter'
        if m in (3, 4, 5):
            return 'Spring'
        if m in (6, 7, 8):
            return 'Summer'
        return 'Autumn'

    # Map numeric month to seasonal category.
    data['season'] = month.map(month_to_season)
    # Remove the raw datetime column so we don't accidentally feed non-numeric values to the model.
    data = data.drop(columns=['Tarih'])

# Drop an identifier-like column that doesn't help prediction and may leak nothing useful.
data = data.drop(columns=['İş Emri No'])

# Define the target column we want to predict (demand in kilograms).
target_col = 'İhtiyaç Kg'

# Apply log1p to the target to stabilize variance and reduce effect of large outliers.
data[target_col + '_log'] = np.log1p(data[target_col])
# Keep the name of the transformed target for clarity and to avoid typos.
target_col_transformed = target_col + '_log'

# Build the feature frame by dropping both the raw and transformed target; we'll predict the transformed one.
X_df = data.drop(columns=[target_col, target_col_transformed]).copy()
# Extract the transformed target values as a 1D float array for Keras.
y = data[target_col_transformed].values.astype(float)


# Keep a dictionary of label encoders so we could reuse them later if needed (e.g., inference).
encoders = {}
for col in X_df.columns:
    # Convert object/category columns to numeric codes so neural networks can process them.
    if X_df[col].dtype == 'object' or str(X_df[col].dtype).startswith('category'):
        le = LabelEncoder()
        # Ensure consistent string dtype and replace missing values before encoding.
        X_df[col] = X_df[col].astype(str).fillna('')
        X_df[col] = le.fit_transform(X_df[col])
        encoders[col] = le

# Convert the features to a dense float matrix for scaling and modeling.
X = X_df.values.astype(float)

# Scale features robustly (less sensitive to outliers than standard scaling) for better network training.
feature_scaler = RobustScaler()
X = feature_scaler.fit_transform(X)

# Clip extreme values after scaling to limit the impact of outliers and stabilize training.
X = np.clip(X, -10, 10)
# Clip the log target to a reasonable range; avoids exploding losses on rare extreme values.
y = np.clip(y, 0, 10)

# Turn tabular rows into short overlapping sequences so sequence models (LSTM) can learn temporal/local patterns.
def create_enhanced_sequences(X, y, seq_length=8):
    """Create enhanced sequences with overlapping windows"""
    # If we have fewer rows than the requested sequence length, repeat to form a minimal sequence tensor.
    if len(X) < seq_length:
        X_seq = np.repeat(X[np.newaxis, :, :], seq_length, axis=1)
        return X_seq, y
    
    X_seq = []
    y_seq = []
    
    # Use a stride of half the window for overlap to increase samples and context without exploding dataset size.
    stride = max(1, seq_length // 2)
    for i in range(seq_length, len(X), stride):
        # Slice a window of 'seq_length' rows to form one sequence sample.
        X_seq.append(X[i-seq_length:i])
        # Use the target at the window end as the label for that sequence.
        y_seq.append(y[i])
    
    return np.array(X_seq), np.array(y_seq)

# Split into train and test sets to fairly evaluate generalization.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose a sequence length; use at most 8 and scale down if the train set is small.
# Note: if len(X_train) < 8, len(X_train)//8 becomes 0 and seq_length becomes 0, which is risky;
# consider enforcing a minimum like max(2, ...) in production to avoid degenerate windows.
seq_length = min(8, len(X_train) // 8)

# Build overlapping sequences for both train and test using the chosen window size.
X_train_seq, y_train_seq = create_enhanced_sequences(X_train, y_train, seq_length)
X_test_seq, y_test_seq = create_enhanced_sequences(X_test, y_test, seq_length)

# Define a simple attention mechanism to learn a weighted summary over time steps output by the LSTM.
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
    # Learnable projection from feature space (the last dim of inputs) to a single scalar per time step.
    # Shape details:
    # - input_shape: (batch, timesteps, features)
    # - W shape (features, 1) turns each time step's feature vector into a 1D score.
        self.W = self.add_weight(name='attention_weight',
                                shape=(input_shape[-1], 1),
                                initializer='random_normal',
                                trainable=True)
        super(AttentionLayer, self).build(input_shape)
        
    def call(self, inputs):
    # Project features to scores; tensordot with axes=1 multiplies each time step by W.
    # tanh keeps scores bounded (-1,1) which stabilizes the softmax downstream.
        score = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1))
    # Softmax over axis=1 (time) produces attention weights that sum to 1 across timesteps for each sample.
        attention_weights = tf.nn.softmax(score, axis=1)
        
    # Weight the inputs by attention and sum across time to get a context vector (batch, features).
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector

# Stack bidirectional LSTMs to capture patterns looking both forward and backward in the sequence.
# Model architecture details:
# - Bidirectional LSTM layers let the network consider both past and future within each window, which can
#   be beneficial for non-causal tabular sequences.
# - units=128/64 control the dimensionality of the hidden state; larger = more capacity but higher overfit risk.
# - return_sequences=True keeps the full sequence output for the next LSTM and attention.
# - dropout vs recurrent_dropout:
#   * dropout=0.2 randomly zeros input connections to each LSTM unit (regularizes inputs at each time).
#   * recurrent_dropout=0.1 zeros a fraction of recurrent connections (state-to-state), regularizing memory.
#   Note: recurrent_dropout can slow training and disable certain CuDNN fast paths, but improves robustness.
# - kernel_regularizer=l2(0.001) adds 0.001 * sum(weights^2) to the loss, shrinking weights (weight decay)
#   to reduce overfitting; 0.001 is a moderate penalty commonly effective with dense/LSTM layers.
model = tf.keras.models.Sequential([
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.1,
                           kernel_regularizer=tf.keras.regularizers.l2(0.001))
    ),
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.1,
                           kernel_regularizer=tf.keras.regularizers.l2(0.001))
    ),
    
    # Apply attention to focus on the most informative time steps from the LSTM outputs.
    AttentionLayer(),
    
    # Dense head maps the attention-pooled features to the final scalar. ReLU avoids vanishing gradients.
    # L2(0.001) continues to constrain weights; BatchNorm normalizes activations, accelerating training.
    tf.keras.layers.Dense(128, activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    # Dropout(0.3) zeros 30% of activations to improve generalization.
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Dense(64, activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.2),
    
    # Final Dense(1) uses a linear activation for regression on the log-transformed target.
    tf.keras.layers.Dense(1)
])

# Use AdamW optimizer (Adam with decoupled weight decay): weight_decay=0.01 applies true weight decay update
# separate from gradient-based L2. This helps generalization compared to plain L2 in the loss.
# MeanSquaredLogarithmicError (MSLE) computes MSE in log-space; since targets are log1p-transformed,
# this aligns the loss with the prediction scale and reduces sensitivity to large targets.
model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.01),
    loss=tf.keras.losses.MeanSquaredLogarithmicError(),
    metrics=['mae', 'mse']
)

# Early stopping prevents overfitting; ReduceLROnPlateau lowers LR if validation stagnates.
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=25, restore_best_weights=True, monitor='val_loss'),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=12, min_lr=1e-7),
]

# Train the model on the sequence data; keep part of training data for validation.
history = model.fit(X_train_seq, y_train_seq, 
                   epochs=180, 
                   batch_size=16,
                   validation_split=0.2, 
                   callbacks=callbacks,
                   verbose=1)

# Evaluate on held-out test sequences to measure generalization.
test_results = model.evaluate(X_test_seq, y_test_seq, verbose=0)
# Keras may return list [loss, metrics...] or a scalar; normalize to a single value for printing.
test_loss = test_results[0] if isinstance(test_results, list) else test_results

# Predict on test sequences; outputs are in log space because we trained on log1p(target).
y_pred_log = model.predict(X_test_seq, verbose=0)
# Convert predictions back to the original target scale via inverse of log1p.
y_pred_original = np.expm1(y_pred_log.flatten())
y_test_original = np.expm1(y_test_seq)

# Compute mean squared error and mean absolute error on the original target scale for interpretability.
mse_original = np.mean((y_test_original - y_pred_original)**2)
mae_original = np.mean(np.abs(y_test_original - y_pred_original))

print(f'\nBidirectional LSTM with Attention Model Performance:')
print(f'Original-scale Test MSE: {mse_original:.2f}')
print(f'Original-scale Test MAE: {mae_original:.2f}')
print(f'Original-scale Test R2: {r2_score(y_test_original, y_pred_original):.4f}')

# Show some predictions vs actual
print(f'\nSample Predictions vs Actual (original scale):')
for i in range(min(10, len(y_test_original))):
    print(f'Predicted: {y_pred_original[i]:.2f}, Actual: {y_test_original[i]:.2f}')

# Plot training history
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('MSLE Loss')
plt.legend()
plt.title('BiLSTM-Attention - Loss')
plt.show()
