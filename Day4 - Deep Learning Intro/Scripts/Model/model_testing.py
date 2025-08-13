import pandas as pd
import numpy as np
import tensorflow as tf

# Load the dataset
new_data = pd.read_csv('DataSets\Real_Life_Data.csv')

# Retain relevant features
relevant_features = ['Cement', 'SP', 'Age', 'Water']

# Features to create interactions with relevant features
interaction_features = {
    'BFS': ['Cement', 'FAgg', 'CAgg', 'FA'],
    'FA': ['Cement', 'SP', 'Water'],
    'CAgg': ['SP'],
    'FAgg': ['Water', 'SP', 'Cement']
}

# Create interaction terms in the dataset
for feature, interactions in interaction_features.items():
    for interaction in interactions:
        interaction_term = new_data[feature] * new_data[interaction]
        new_data[f'{feature}_x_{interaction}'] = interaction_term

# Keep only the relevant features and the created interaction terms
selected_features = relevant_features + [f'{feature}_x_{interaction}' for feature, interactions in interaction_features.items() for interaction in interactions]
X_new = new_data[selected_features].values

# Standardize the input features using the mean and std from the training data
X_mean = np.load('ModelWeight/X_mean.npy')
X_std = np.load('ModelWeight/X_std.npy')
X_new = (X_new - X_mean) / X_std

# Load the trained model
model = tf.keras.models.load_model('ModelWeight/regression_model.h5')

# Make predictions
predictions = model.predict(X_new)

# Convert predictions back to the original scale by exponentiating
predictions = np.exp(predictions)

# Save predictions to a CSV file
predictions_df = pd.DataFrame(predictions, columns=['Predicted_CS'])
predictions_df.to_csv('DataSets\Predictions.csv', index=False)
