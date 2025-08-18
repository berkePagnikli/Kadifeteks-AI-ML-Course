import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class CancerInference:
    def __init__(self):
        """
        Initialize the inference class for cancer classification
        """
        self.model = None
        self.scaler = None
        self.features_to_remove = [
            'fractal_dimension_mean',
            'texture_se',
            'smoothness_se',
            'symmetry_se',
            'fractal_dimension_se'
        ]
        
        # Available models and their corresponding files
        self.available_models = {
            'logistic_regression': {
                'model_file': 'ModelWeights/logistic_regression_model.pkl',
                'scaler_file': 'ModelWeights/scaler.pkl',
                'name': 'Logistic Regression'
            },
            'decision_tree': {
                'model_file': 'ModelWeights/decision_tree_model.pkl',
                'scaler_file': 'ModelWeights/scaler_dt.pkl',
                'name': 'Decision Tree'
            },
            'random_forest': {
                'model_file': 'ModelWeights/random_forest_model.pkl',
                'scaler_file': 'ModelWeights/scaler_rf.pkl',
                'name': 'Random Forest'
            },
            'xgboost': {
                'model_file': 'ModelWeights/xgboost_model.pkl',
                'scaler_file': 'ModelWeights/scaler_xgb.pkl',
                'name': 'XGBoost'
            },
            'catboost': {
                'model_file': 'ModelWeights/catboost_model.pkl',
                'scaler_file': 'ModelWeights/scaler_cb.pkl',
                'name': 'CatBoost'
            }
        }
    
    def load_model(self, model_type='xgboost'):
        """Load a trained model and its scaler """
        
        model_info = self.available_models[model_type]
        
        try:
            # Load the model
            self.model = joblib.load(model_info['model_file'])
            
            # Load the scaler
            self.scaler = joblib.load(model_info['scaler_file'])
            return True
            
        except FileNotFoundError as e:
            print(f"Error loading model files: {e}")
            return False
    
    def preprocess_data(self, df):
        """Preprocess the data same way as during training"""

        # Check if target column exists
        has_target = df.columns[0] == "diagnosis(1=m, 0=b)"
        
        if has_target:
            # Separate features and target
            y = df.iloc[:, 0]  # First column is diagnosis
            X = df.iloc[:, 1:]  # All other columns are features
        else:
            # No target column, all columns are features
            y = None
            X = df.copy()
        
        # Remove specified features (same as during training)
        X = X.drop(columns=self.features_to_remove, errors='ignore')
        
        # Scale the features using the loaded scaler
        X_scaled = self.scaler.transform(X)
        
        return X_scaled, y
    
    def make_predictions(self, X_scaled):
        """Make predictions using the loaded model"""

        if self.model is None:
            raise ValueError("No model loaded! Call load_model() first.")
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        # Get prediction probabilities
        try:
            probabilities = self.model.predict_proba(X_scaled)
        except:
            # Some models might not have predict_proba
            probabilities = None
        
        return predictions, probabilities
    
    def display_results(self, y_true, y_pred, probabilities=None, sample_limit=None):
        """Display actual vs predicted results.

        If sample_limit is None, display all samples. Otherwise, display up to sample_limit.
        """

        print("\n" + "="*80)
        print("INFERENCE RESULTS")
        print("="*80)
        
        # Overall metrics
        if y_true is not None:
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            
            print(f"\nOVERALL PERFORMANCE:")
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
        
        # Sample-by-sample comparison
        print("-" * 80)
        print(f"{'Index':<6} {'Actual':<8} {'Predicted':<10} {'Match':<6} {'Confidence':<12} {'Status':<10}")
        print("-" * 80)
        
        # Choose which indices to display to avoid redundant adjacent rows
        if sample_limit is None or sample_limit >= len(y_pred):
            display_indices = range(len(y_pred))
        else:
            # Evenly spaced sampling across the dataset length
            display_indices = np.linspace(0, len(y_pred) - 1, num=sample_limit, dtype=int)

        for idx in display_indices:
            # Get actual label
            actual = y_true[idx] if y_true is not None else "Unknown"
            predicted = y_pred[idx]
            
            # Check if prediction matches
            if y_true is not None:
                match = "+" if actual == predicted else "X"
                match_status = "Correct" if actual == predicted else "Wrong"
            else:
                match = "N/A"
                match_status = "N/A"
            
            # Get confidence score
            if probabilities is not None:
                confidence = f"{max(probabilities[idx]):.4f}"
            else:
                confidence = "N/A"
            
            # Convert labels to readable format
            actual_label = "Benign" if actual == 0 else "Malignant" if actual == 1 else actual
            pred_label = "Benign" if predicted == 0 else "Malignant"
            
            print(f"{idx+1:<6} {actual_label:<8} {pred_label:<10} {match:<6} {confidence:<12} {match_status:<10}")
        
        # Summary statistics
        print(f"\nSUMMARY:")
        print(f"Total samples: {len(y_pred)}")
        if y_true is not None:
            print(f"Correct predictions: {sum(y_true == y_pred)}")
            print(f"Wrong predictions: {sum(y_true != y_pred)}")
        print(f"Benign predictions: {sum(y_pred == 0)}")
        print(f"Malignant predictions: {sum(y_pred == 1)}")
    
    def run_inference(self, test_file='test.csv', model_type='xgboost', display_limit=None):
        """Run complete inference pipeline"""

        print("="*80)
        print(f"CANCER CLASSIFICATION INFERENCE")
        print("="*80)
        
        # Step 1: Load model
        if not self.load_model(model_type):
            return None
        
        # Step 2: Load test data
        try:
            test_df = pd.read_csv(test_file)
        except FileNotFoundError:
            return None
        
        # Step 3: Preprocess data
        X_scaled, y_true = self.preprocess_data(test_df)
        
        # Step 4: Make predictions
        y_pred, probabilities = self.make_predictions(X_scaled)
        
        # Step 5: Display results
        self.display_results(y_true, y_pred, probabilities, display_limit)
        
        return {
            'y_true': y_true,
            'y_pred': y_pred,
            'probabilities': probabilities
        }

def main():
    """Main function to run inference"""

    inference = CancerInference()
    
    print("\n" + "="*50)
    
    # Run inference with different models
    models_to_test = ['xgboost', 'logistic_regression', 'catboost', 'decision_tree', 'random_forest']
    
    for model_type in models_to_test:
        print(f"\n{'='*20} TESTING {model_type.upper()} {'='*20}")
        results = inference.run_inference(
            test_file='test.csv',
            model_type=model_type,
            display_limit=None  # Show all samples for each model
        )
        
        if results is not None:
            print(f"{model_type} inference completed successfully!")
        else:
            print(f"{model_type} inference failed!")
        
        print("\n" + "="*80)

if __name__ == "__main__":
    main()
