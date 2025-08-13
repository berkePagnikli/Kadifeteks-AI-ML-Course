import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('DataSets\Concrete_Data_Winsorized.csv')

# Retain relevant features
relevant_features = ['Cement', 'SP', 'Age', 'Water']

# Features to create interactions with relevant features
interaction_features = {
    'BFS': ['Cement', 'FAgg', 'CAgg', 'FA'],
    'FA': ['Cement', 'SP', 'Water'],
    'CAgg': ['SP'],
    'FAgg': ['Water', 'SP', 'Cement']
}

# Create interaction terms
for feature, interactions in interaction_features.items():
    for interaction in interactions:
        interaction_term = data[feature] * data[interaction]
        data[f'{feature}_x_{interaction}'] = interaction_term

# Keep only the relevant features and the created interaction terms
selected_features = relevant_features + [f'{feature}_x_{interaction}' for feature, interactions in interaction_features.items() for interaction in interactions]
X = data[selected_features].values
y = data['CS'].values

# Apply log transformation to the target variable
y = np.log(y)

# Standardize the input features
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std

# Data augmentation function by adding Gaussian noise
def augment_data(X, y, num_augmentations=2, noise_level=0.02):
    augmented_X = []
    augmented_y = []
    for _ in range(num_augmentations):
        noise = np.random.normal(0, noise_level, X.shape)
        X_augmented = X + noise
        augmented_X.append(X_augmented)
        augmented_y.append(y)
    return np.vstack(augmented_X), np.hstack(augmented_y)

# Augment the data
X_augmented, y_augmented = augment_data(X, y)

# Combine original and augmented data
X_combined = np.vstack((X, X_augmented))
y_combined = np.hstack((y, y_augmented))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, input_dim=X_train.shape[1], kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9), loss='mean_squared_error')

# Set up callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)
]

# Train the model
history = model.fit(X_train, y_train, epochs=150, batch_size=16, validation_split=0.3, callbacks=callbacks)

# Evaluate the model
loss = model.evaluate(X_test, y_test)

# Save the model
np.save('ModelWeight/X_mean.npy', X_mean)
np.save('ModelWeight/X_std.npy', X_std)
model.save('ModelWeight/regression_model.h5')

# Plot training history
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()