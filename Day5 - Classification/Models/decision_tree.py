import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib

class CancerDecisionTree:
    def __init__(self, data_path):
        """Initialize the decision tree class"""

        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.scaler = None
        
        # Features to remove as specified
        self.features_to_remove = [
            'fractal_dimension_mean',
            'texture_se',
            'smoothness_se',
            'symmetry_se',
            'fractal_dimension_se'
        ]
        
    def load_and_preprocess_data(self):
        """Load the dataset and perform preprocessing"""

        self.df = pd.read_csv(self.data_path)
    
        # Separate features and target
        self.y = self.df.iloc[:, 0]  # First column is diagnosis
        self.X = self.df.iloc[:, 1:]  # All other columns are features
        
        # Remove specified features
        self.X = self.X.drop(columns=self.features_to_remove, errors='ignore')
        
        return self.X, self.y
    
    def augment_minority_class(self, noise_std=0.01, random_state=42):
        """Augment the minority class (malignant) by adding noise to existing samples"""

        np.random.seed(random_state)
        
        # Get malignant samples
        malignant_mask = self.y == 1
        malignant_X = self.X[malignant_mask]
        
        # Calculate how many samples to add to balance the classes
        benign_count = sum(self.y == 0)
        malignant_count = sum(self.y == 1)
        samples_to_add = benign_count - malignant_count
        
        # Generate augmented samples
        augmented_samples = []
        augmented_labels = []
        
        for i in range(samples_to_add):
            # Randomly select a malignant sample
            idx = np.random.randint(0, len(malignant_X))
            original_sample = malignant_X.iloc[idx].values
            
            # Add small random noise
            noise = np.random.normal(0, noise_std, size=original_sample.shape)
            augmented_sample = original_sample + noise
            
            augmented_samples.append(augmented_sample)
            augmented_labels.append(1)  # Malignant label
        
        # Convert to DataFrame and Series
        augmented_X = pd.DataFrame(augmented_samples, columns=self.X.columns)
        augmented_y = pd.Series(augmented_labels)
        
        # Combine original and augmented data
        self.X = pd.concat([self.X, augmented_X], ignore_index=True)
        self.y = pd.concat([self.y, augmented_y], ignore_index=True)
        
    def split_and_scale_data(self, test_size=0.2, random_state=42):
        """Split the data into train/test sets and scale features"""
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, 
            stratify=self.y
        )
        
        # Scale the features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
    def train_model(self, random_state=42):
        """Train the decision tree model"""
        
        # Initialize and train the model
        self.model = DecisionTreeClassifier(
            random_state=random_state,
            max_depth=10,  # Prevent overfitting
            min_samples_split=20,  # Prevent overfitting
            min_samples_leaf=10,   # Prevent overfitting
            criterion='gini'       # Gini impurity
        )
        
        # Train the model 
        self.model.fit(self.X_train_scaled, self.y_train)
        
    def evaluate_model(self):
        """Evaluate the model and display metrics"""
        print(f"\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        # Make predictions
        y_train_pred = self.model.predict(self.X_train_scaled)
        y_test_pred = self.model.predict(self.X_test_scaled)
        
        # Calculate metrics for training set
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        train_precision = precision_score(self.y_train, y_train_pred)
        train_recall = recall_score(self.y_train, y_train_pred)
        train_f1 = f1_score(self.y_train, y_train_pred)
        
        # Calculate metrics for test set
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        test_precision = precision_score(self.y_test, y_test_pred)
        test_recall = recall_score(self.y_test, y_test_pred)
        test_f1 = f1_score(self.y_test, y_test_pred)
        
        # Display results
        print("\nTRAINING SET METRICS:")
        print(f"Accuracy:  {train_accuracy:.4f}")
        print(f"Precision: {train_precision:.4f}")
        print(f"Recall:    {train_recall:.4f}")
        print(f"F1-Score:  {train_f1:.4f}")
        
        print("\nTEST SET METRICS:")
        print(f"Accuracy:  {test_accuracy:.4f}")
        print(f"Precision: {test_precision:.4f}")
        print(f"Recall:    {test_recall:.4f}")
        print(f"F1-Score:  {test_f1:.4f}")
        
        # Detailed classification report
        print(f"\nDETAILED CLASSIFICATION REPORT (Test Set):")
        print(classification_report(self.y_test, y_test_pred, 
                                  target_names=['Benign', 'Malignant']))
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_test_pred)
        print(f"\nCONFUSION MATRIX (Test Set):")
        print(cm)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Benign', 'Malignant'],
                   yticklabels=['Benign', 'Malignant'])
        plt.title('Confusion Matrix - Decision Tree')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('Graphs/confusion_matrix_decision_tree.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1
        }
    
    def save_model(self, model_filename='ModelWeights/decision_tree_model.pkl', 
                   scaler_filename='ModelWeights/scaler_dt.pkl'):
        """Save the trained model and scaler"""
        
        # Save the model
        joblib.dump(self.model, model_filename)
        
        # Save the scaler
        joblib.dump(self.scaler, scaler_filename)
    
    def run_complete_pipeline(self):
        """Run the complete machine learning pipeline"""
        print("="*60)
        print("DECISION TREE PIPELINE FOR CANCER CLASSIFICATION")
        print("="*60)
        
        # Step 1: Load and preprocess data
        self.load_and_preprocess_data()
        
        # Step 2: Augment minority class
        self.augment_minority_class()
        
        # Step 3: Split and scale data
        self.split_and_scale_data()
        
        # Step 4: Train model
        self.train_model()
        
        # Step 5: Evaluate model
        metrics = self.evaluate_model()
        
        # Step 6: Save model
        self.save_model()
        
        print(f"\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return metrics

def main():
    """Main function to run the decision tree pipeline"""
    # Initialize the cancer classification system
    cancer_dt = CancerDecisionTree('cancer.csv')
    
    # Run the complete pipeline
    metrics = cancer_dt.run_complete_pipeline()
    
    print(f"\nFinal Test Set Performance:")
    print(f"Accuracy:  {metrics['test_accuracy']:.4f}")
    print(f"Precision: {metrics['test_precision']:.4f}")
    print(f"Recall:    {metrics['test_recall']:.4f}")
    print(f"F1-Score:  {metrics['test_f1']:.4f}")

if __name__ == "__main__":
    main()
