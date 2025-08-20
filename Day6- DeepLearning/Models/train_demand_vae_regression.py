import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import r2_score
import tensorflow as tf
import matplotlib.pyplot as plt

# Read the dataset from disk.
data = pd.read_csv('manipulated_data.csv')

# If a date column exists, engineer cyclical and seasonal features to enrich the inputs.
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
    # Drop the raw datetime; model will consume numeric/encoded features only.
    data = data.drop(columns=['Tarih'])

# Remove an ID-like column that is not useful for prediction.
data = data.drop(columns=['İş Emri No'])

# Define the target variable to predict and create a log-transformed version.
target_col = 'İhtiyaç Kg'

data[target_col + '_log'] = np.log1p(data[target_col])
target_col_transformed = target_col + '_log'


# Build the feature DataFrame and extract the transformed target.
X_df = data.drop(columns=[target_col, target_col_transformed]).copy()
y = data[target_col_transformed].values.astype(float)

# Label-encode categorical features so they become numeric inputs for the VAE and regressor.
encoders = {}
for col in X_df.columns:
    if X_df[col].dtype == 'object' or str(X_df[col].dtype).startswith('category'):
        le = LabelEncoder()
        X_df[col] = X_df[col].astype(str).fillna('')
        X_df[col] = le.fit_transform(X_df[col])
        encoders[col] = le

# Convert features to floats and scale robustly to reduce the effect of outliers.
X = X_df.values.astype(float)

feature_scaler = RobustScaler()
X = feature_scaler.fit_transform(X)

# Clip to a bounded range to make the VAE training more stable.
X = np.clip(X, -10, 10)
y = np.clip(y, 0, 10)

# Train/test split for unbiased evaluation of both the VAE and the downstream regressor.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Custom reparameterization layer to sample latent vectors z ~ N(z_mean, exp(z_log_var)).
class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        # Determine batch size and latent dimension dynamically from the inputs.
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        # Sample epsilon ~ N(0, I) and transform into the latent distribution.
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Build the VAE encoder that maps inputs to latent parameters and sampled latent code.
def create_encoder(input_dim, latent_dim):
    """Create VAE encoder"""
    inputs = tf.keras.layers.Input(shape=(input_dim,))
    
    # Encoder layers
    x = tf.keras.layers.Dense(256, activation="relu")(inputs)  # ReLU for encoder; simple and effective.
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Latent space parameters
    z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    
    encoder = tf.keras.models.Model(inputs, [z_mean, z_log_var, z], name="encoder")
    return encoder

# Build the VAE decoder that reconstructs inputs from latent vectors.
def create_decoder(latent_dim, output_dim):
    """Create VAE decoder"""
    latent_inputs = tf.keras.layers.Input(shape=(latent_dim,))
    
    x = tf.keras.layers.Dense(64, activation="relu")(latent_inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    outputs = tf.keras.layers.Dense(output_dim, name="decoder_output")(x)
    
    decoder = tf.keras.models.Model(latent_inputs, outputs, name="decoder")
    return decoder

# Define a subclassed VAE model to implement custom train/test steps and log losses.
class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        # Custom training step to compute VAE loss = recon_loss + beta * KL.
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            # Reconstruction loss (MSE) encourages the decoder to faithfully reproduce inputs from z.
            reconstruction_loss = tf.reduce_mean(tf.keras.losses.mse(data, reconstruction))
            
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

            beta = 0.5  # Beta-VAE weighting; >0 increases disentanglement pressure by upweighting KL.
            total_loss = reconstruction_loss + beta * kl_loss
        
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def test_step(self, data):
        # Validation step mirrors training without gradient updates.
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        
        reconstruction_loss = tf.reduce_mean(tf.keras.losses.mse(data, reconstruction))
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        beta = 0.5
        total_loss = reconstruction_loss + beta * kl_loss
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

# Choose input and latent dimensions; latent limited to keep representation compact.
input_dim = X_train.shape[1]
latent_dim = min(16, input_dim // 2)  # Keep latent small to compress information and avoid trivial identity.

encoder = create_encoder(input_dim, latent_dim)
decoder = create_decoder(latent_dim, input_dim)
vae = VAE(encoder, decoder)

# Optimize with Adam; the VAE learns a compressed representation of the inputs.
vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

vae_history = vae.fit(
    X_train, 
    epochs=100,
    batch_size=32,
    validation_data=(X_test, None),
    verbose=1
)

# Extract the mean of the latent distribution as a deterministic embedding for each row.
z_mean_train, _, _ = vae.encoder(X_train)
z_mean_test, _, _ = vae.encoder(X_test)

# Concatenate the learned latent features to the original features to create an enhanced input space.
X_train_enhanced = np.concatenate([X_train, z_mean_train.numpy()], axis=1)
X_test_enhanced = np.concatenate([X_test, z_mean_test.numpy()], axis=1)

# Define a regression model that consumes the enhanced features.
def create_regression_model(input_dim):
    inputs = tf.keras.layers.Input(shape=(input_dim,))

    x = tf.keras.layers.Dense(512, activation="swish")(inputs)
    # Swish activation: swish(x) = x * sigmoid(x). Compared to ReLU, Swish is smooth and non-monotonic,
    # often improving performance in deep networks by allowing small negative outputs and smoother gradients.
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Residual block to deepen the network while keeping gradients stable.
    residual = x
    x = tf.keras.layers.Dense(512, activation="swish", 
                             kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    # L2(0.001) encourages smaller weights (weight decay) to curb overfitting with high-capacity layers.
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(512, activation="swish")(x)
    x = tf.keras.layers.Add()([x, residual])
    x = tf.keras.layers.BatchNormalization()(x)
  
    # Split the input back into original vs latent parts to allow specialized processing paths.
    original_features = inputs[:, :input_dim-latent_dim]
    latent_features = inputs[:, input_dim-latent_dim:]
    
    orig_path = tf.keras.layers.Dense(256, activation="swish")(original_features)
    orig_path = tf.keras.layers.Dropout(0.2)(orig_path)
    
    latent_path = tf.keras.layers.Dense(128, activation="swish")(latent_features)
    latent_path = tf.keras.layers.Dropout(0.1)(latent_path)

    # Combine the shared residual stream with specialized paths.
    combined = tf.keras.layers.Concatenate()([x, orig_path, latent_path])
    
    x = tf.keras.layers.Dense(256, activation="swish",
                             kernel_regularizer=tf.keras.regularizers.l2(0.001))(combined)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    x = tf.keras.layers.Dense(128, activation="swish")(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    
    outputs = tf.keras.layers.Dense(1)(x)
    
    model = tf.keras.models.Model(inputs, outputs)
    return model

regression_model = create_regression_model(X_train_enhanced.shape[1])

# Define a custom loss that combines MSE and MAE, scaled by prediction dispersion to adapt to uncertainty.
def combined_regression_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    # Adaptive weighting: if predictions vary a lot (high std), reduce MAE impact; if too flat, increase it.
    pred_std = tf.keras.backend.std(y_pred)
    weight = tf.clip_by_value(1.0 / (pred_std + 1e-8), 0.1, 2.0)
    return 0.6 * mse + 0.4 * mae * weight

regression_model.compile(
    # AdamW combines Adam's adaptivity with decoupled weight decay (0.01) to improve generalization.
    optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.01),
    loss=combined_regression_loss,
    metrics=['mae', 'mse']
)

regression_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, monitor='val_loss'),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10),
]

regression_history = regression_model.fit(
    X_train_enhanced, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    callbacks=regression_callbacks,
    verbose=1
)


# Evaluate on the enhanced test features to measure real-world performance.
test_results = regression_model.evaluate(X_test_enhanced, y_test, verbose=0)
test_loss = test_results[0] if isinstance(test_results, list) else test_results

# Convert predictions back to the original units (kg) for intuitive interpretation.
y_pred_log = regression_model.predict(X_test_enhanced, verbose=0)
y_pred_original = np.expm1(y_pred_log.flatten())
y_test_original = np.expm1(y_test)

mse_original = np.mean((y_test_original - y_pred_original)**2)
mae_original = np.mean(np.abs(y_test_original - y_pred_original))

print(f'\nVAE-Enhanced Regression Model Performance:')
print(f'Log-scale Test Loss: {test_loss:.4f}')
print(f'Original-scale Test MSE: {mse_original:.2f}')
print(f'Original-scale Test MAE: {mae_original:.2f}')
print(f'Original-scale Test R2: {r2_score(y_test_original, y_pred_original):.4f}')

print(f'\nSample Predictions vs Actual (original scale):')
for i in range(min(10, len(y_test_original))):
    print(f'Predicted: {y_pred_original[i]:.2f}, Actual: {y_test_original[i]:.2f}')

plt.figure(figsize=(20, 10))
plt.subplot(2, 3, 1)
plt.plot(vae_history.history['loss'], label='Total Loss')
plt.plot(vae_history.history['val_loss'], label='Val Total Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('VAE - Total Loss')
plt.show()