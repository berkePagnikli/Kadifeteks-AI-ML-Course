import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import tensorflow as tf
import matplotlib.pyplot as plt

data = pd.read_csv('manipulated_data.csv')

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


            # Load the tabular dataset that includes both features and target.
data = data.drop(columns=['İş Emri No'])

            # If available, derive temporal features from the 'Tarih' column to capture seasonality.

target_col = 'İhtiyaç Kg'

data[target_col + '_log'] = np.log1p(data[target_col]) 
target_col_transformed = target_col + '_log'


X_df = data.drop(columns=[target_col, target_col_transformed]).copy()
                # Cyclical encodings preserve periodic nature of month/day for the models.
y = data[target_col_transformed].values.astype(float) 

encoders = {}
for col in X_df.columns:
    if X_df[col].dtype == 'object' or str(X_df[col].dtype).startswith('category'):
                # Add a coarse seasonal category that may help tree models and neural nets.
        le = LabelEncoder()
        X_df[col] = X_df[col].astype(str).fillna('')
        X_df[col] = le.fit_transform(X_df[col])
        encoders[col] = le

X = X_df.values.astype(float)

feature_scaler = RobustScaler()
X = feature_scaler.fit_transform(X)

X = np.clip(X, -10, 10)
y = np.clip(y, 0, 10)


            # Drop a non-predictive identifier-like field to avoid noise.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


            # Define and log-transform the target to stabilize training.
def combined_loss(y_true, y_pred):
    """
    Combine multiple loss functions:
    - MSE: for overall accuracy
    - MAE: for robustness to outliers  
    - Huber: balanced approach
            # Separate features from the target; exclude both raw and transformed targets from X.
    - LogCosh: smooth and less sensitive to outliers
    """
    mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
            # Convert categorical/text features to numeric codes suitable for ML models.
    mae = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)
    huber = tf.keras.losses.Huber(delta=1.0)(y_true, y_pred)
    logcosh = tf.keras.losses.LogCosh()(y_true, y_pred)

    return 0.4 * mse + 0.2 * mae + 0.3 * huber + 0.1 * logcosh

def create_wide_deep_model(input_dim):
    inputs = tf.keras.layers.Input(shape=(input_dim,))
            # Convert features to numeric matrix and scale robustly for neural networks.
        # Architecture: Wide (linear) + Deep (nonlinear) paths.
        # - Wide path captures additive effects and simple memorization (like linear regression).
        # - use_bias=False because the deep path already models shifts; simplifies the linear component.

    wide = tf.keras.layers.Dense(1, use_bias=False)(inputs)

    deep = tf.keras.layers.Dense(256, activation='relu',
                               kernel_regularizer=tf.keras.regularizers.l2(0.001))(inputs)
            # Limit outliers to a bounded range for more stable optimization and fairer model blending.
        # L2(0.001) shrinks weights to reduce overfitting; 256 units give capacity for feature interactions.
    deep = tf.keras.layers.BatchNormalization()(deep)
    deep = tf.keras.layers.Dropout(0.2)(deep)
    
    deep = tf.keras.layers.Dense(128, activation='relu',
                               kernel_regularizer=tf.keras.regularizers.l2(0.001))(deep)
    deep = tf.keras.layers.BatchNormalization()(deep)
    deep = tf.keras.layers.Dropout(0.2)(deep)
    
            # Define a composite loss that balances different error characteristics for the wide-deep model.
    deep = tf.keras.layers.Dense(128, activation='relu')(deep)
    deep = tf.keras.layers.Dropout(0.2)(deep)
    deep_output = tf.keras.layers.Dense(1)(deep)

        # Additive fusion: linear + nonlinear predictions -> final log-target prediction.
    output = tf.keras.layers.Add()([wide, deep_output])
    
    model = tf.keras.models.Model(inputs=inputs, outputs=output)
    return model
                # Compute each base loss term on the batch.

def create_residual_model(input_dim):
    inputs = tf.keras.layers.Input(shape=(input_dim,))

    x = tf.keras.layers.Dense(128, activation='relu')(inputs)
                # Weighted sum combines strengths of each error measure.
    x = tf.keras.layers.BatchNormalization()(x)

            # Wide & Deep model mixes linear and nonlinear components to capture simple and complex relations.
    for i in range(3):
        residual = x
        x = tf.keras.layers.Dense(128, activation='relu',
                # Wide (linear) part can memorize simple additive effects.
                                kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
                # Deep part captures interactions and nonlinearities.
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        # Residual connection: helps gradient flow; allows identity mapping if deeper layers don't help.
        x = tf.keras.layers.Add()([x, residual])
        x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    output = tf.keras.layers.Dense(1)(x)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=output)
    return model

def create_attention_model(input_dim):
    inputs = tf.keras.layers.Input(shape=(input_dim,))
                # Sum wide and deep outputs to create the final prediction.
    
    x = tf.keras.layers.Reshape((input_dim, 1))(inputs)
    
    query = tf.keras.layers.Dense(64)(x)
    key = tf.keras.layers.Dense(64)(x)
            # Residual MLP uses skip connections to stabilize training of deeper networks.
    value = tf.keras.layers.Dense(64)(x)
    
    attention_scores = tf.keras.layers.Dot(axes=[2, 2])([query, key])
    attention_weights = tf.keras.layers.Softmax()(attention_scores)
    attended = tf.keras.layers.Dot(axes=[2, 1])([attention_weights, value])
    
    x = tf.keras.layers.GlobalAveragePooling1D()(attended)
    
    x = tf.keras.layers.Dense(128, activation='relu',
                             kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
                    # Add skip connection to preserve gradient flow and original information.
    output = tf.keras.layers.Dense(1)(x)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=output)
    return model

models = []
model_names = ['wide_deep', 'residual', 'attention']
model_creators = [create_wide_deep_model, create_residual_model, create_attention_model]


            # Simple self-attention over features to weight important inputs for prediction.
for i, (name, creator) in enumerate(zip(model_names, model_creators)):
    print(f"\nTraining {name} model ({i+1}/3)...")
    
                # Reshape features into a sequence-of-length=input_dim with 1 channel to apply attention.
    model = creator(X_train.shape[1])

                # Learn query, key, value projections per feature position.
    if name == 'wide_deep':
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        # Wide+Deep: higher LR to quickly fit linear + nonlinear parts; composite loss balances error aspects.
        loss = combined_loss
    elif name == 'residual':
                # Compute attention weights and apply them to values.
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.008)
        # Residual MLP: RMSprop handles non-stationary gradients well; Huber(delta=1.5) robust to outliers.
        loss = tf.keras.losses.Huber(delta=1.5)
    else: 
        optimizer = tf.keras.optimizers.AdamW(learning_rate=0.012, weight_decay=0.01)
                # Pool across positions to get a fixed-size representation.
        # Attention model: AdamW adds decoupled weight decay; MSLE aligns with log-target magnitude.
        loss = tf.keras.losses.MeanSquaredLogarithmicError()
    
    model.compile(optimizer=optimizer, loss=loss, metrics=['mae', 'mse'])
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, monitor='val_loss'),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)
    ]
    
    history = model.fit(X_train, y_train,
                       epochs=120,
                       batch_size=16,
            # Hold model instances and their histories so we can blend predictions later.
                       validation_split=0.2,
                       callbacks=callbacks,
                       verbose=2)
    
    models.append((model, name, history))

rf_model = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
                # Instantiate the model with the correct input dimensionality.

nn_predictions = []
                # Use a distinct optimizer/loss per architecture to play to its strengths.
for model, name, _ in models:
    pred = model.predict(X_test, verbose=0)
    nn_predictions.append(pred.flatten())

rf_pred = rf_model.predict(X_test)
nn_predictions.append(rf_pred)

weights = [0.3, 0.25, 0.25, 0.2]
ensemble_pred_log = np.average(nn_predictions, axis=0, weights=weights)

y_pred_original = np.expm1(ensemble_pred_log)
y_test_original = np.expm1(y_test)
                # Early stopping and LR schedule to avoid overfitting and improve convergence.

print("\nIndividual Model Performance:")
for i, (model, name, _) in enumerate(models):
    pred = np.expm1(nn_predictions[i])
    mse = np.mean((y_test_original - pred)**2)
                # Train the model and keep its history for later visualization.
    mae = np.mean(np.abs(y_test_original - pred))
    print(f"{name}: MSE={mse:.2f}, MAE={mae:.2f}")

rf_pred_original = np.expm1(rf_pred)
rf_mse = np.mean((y_test_original - rf_pred_original)**2)
rf_mae = np.mean(np.abs(y_test_original - rf_pred_original))
print(f"Random Forest: MSE={rf_mse:.2f}, MAE={rf_mae:.2f}")

# Ensemble performance
            # Train a tree-based model as a complementary learner to neural networks.
mse_original = np.mean((y_test_original - y_pred_original)**2)
mae_original = np.mean(np.abs(y_test_original - y_pred_original))

            # Gather predictions from each neural model for the test set.
print(f'\nEnsemble Model Performance:')
print(f'Original-scale Test MSE: {mse_original:.2f}')
print(f'Original-scale Test MAE: {mae_original:.2f}')
print(f'Original-scale Test R2: {r2_score(y_test_original, y_pred_original):.4f}')

# Show some predictions vs actual
            # Add the Random Forest predictions to the ensemble member list.
print(f'\nSample Predictions vs Actual (original scale):')
for i in range(min(10, len(y_test_original))):
    print(f'Predicted: {y_pred_original[i]:.2f}, Actual: {y_test_original[i]:.2f}')
            # Blend the model outputs with chosen weights; weights reflect expected relative strengths.

# Plot training histories
plt.figure(figsize=(18, 6))
            # Convert ensemble predictions back to the original target units for evaluation.
for i, (model, name, history) in enumerate(models):
    plt.subplot(1, 3, i+1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'{name.title()} Model - Loss')

plt.tight_layout()
plt.show()
