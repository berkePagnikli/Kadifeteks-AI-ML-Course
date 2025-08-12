import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def get_season(date_str):
    """Extract season from date string (YYYY-MM-DD format)"""
    try:
        month = int(date_str.split('-')[1])
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    except:
        return 'Unknown'

def preprocess_inference_data(df):
    """Preprocess inference data similar to training data"""
    df['Season'] = df['Tarih'].apply(get_season)

    df_processed = df.drop([
        'İş Emri No', 'Tarih', 'Sipariş No', 'Kumaş Kodu',
        'Firma Ülkesi', 'Desen Adı', 'Varyant No', 'Kalite Adı',
        'Tezgah Kodu', 'Firma Adı', 'Proses Kodu', 'Çözgü Adı'
    ], axis=1)
    
    return df_processed

def load_and_predict_catboost():
    """Load CatBoost model and make predictions"""
    try:
        model = CatBoostRegressor()
        model.load_model('ModelWeights/catboost_regressor.cbm')
        
        with open('ModelWeights/catboost_regressor_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        return model, metadata, 'CatBoostRegressor'
    except Exception as e:
        print(f"Error loading CatBoost model: {e}")
        return None, None, None

def load_and_predict_sklearn_model(model_name):
    """Load sklearn-based models and make predictions"""
    try:
        with open(f'ModelWeights/{model_name}.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open(f'ModelWeights/{model_name}_components.pkl', 'rb') as f:
            components = pickle.load(f)
        
        return model, components, components['model_type']
    except Exception as e:
        print(f"Error loading {model_name} model: {e}")
        return None, None, None

def encode_categorical_features(X, label_encoders, categorical_features):
    """Encode categorical features using saved label encoders"""
    X_encoded = X.copy()
    
    for col in categorical_features:
        if col in X_encoded.columns:
            le = label_encoders[col]
            try:
                X_encoded[col] = le.transform(X_encoded[col].astype(str))
            except ValueError:
                X_encoded[col] = X_encoded[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
    
    return X_encoded

def main():

    print("Loading inference data...")
    inference_df = pd.read_csv('inference.csv')

    y_actual = inference_df['İhtiyaç Kg'].values

    inference_processed = preprocess_inference_data(inference_df)
    
    target = 'İhtiyaç Kg'
    X_inference = inference_processed.drop(target, axis=1)
    
    print(f"Inference data shape: {X_inference.shape}")
    print(f"Number of samples: {len(y_actual)}")
    print("\n" + "="*60)
    
    models_info = [
        ('catboost_regressor', 'CatBoost'),
        ('linear_regressor', 'Linear Regression'),
        ('xgboost_regressor', 'XGBoost'),
        ('random_forest_regressor', 'Random Forest'),
        ('decision_tree_regressor', 'Decision Tree'),
        ('polynomial_regressor', 'Polynomial Regression')
    ]
    
    results = []
    
    for model_file, model_display_name in models_info:
        print(f"\n{model_display_name} Predictions:")
        print("-" * 40)
        
        try:
            if model_file == 'catboost_regressor':
                model, metadata, model_type = load_and_predict_catboost()
                if model is not None:
                    y_pred = model.predict(X_inference)
            else:
                model, components, model_type = load_and_predict_sklearn_model(model_file)
                if model is not None:
                    X_encoded = encode_categorical_features(
                        X_inference, 
                        components['label_encoders'], 
                        components['categorical_features']
                    )
                    
                    if model_file == 'polynomial_regressor':
                        poly = components['polynomial_features']
                        X_encoded = poly.transform(X_encoded)
                    
                    y_pred = model.predict(X_encoded)
            
            if model is not None:
                mse = mean_squared_error(y_actual, y_pred)
                r2 = r2_score(y_actual, y_pred)
                
                print(f"MSE: {mse:.2f}")
                print(f"R² Score: {r2:.4f}")
                print(f"RMSE: {np.sqrt(mse):.2f}")

                print(f"\nFirst 10 predictions:")
                print(f"{'Index':<5} {'Actual':<10} {'Predicted':<12} {'Diff':<10}")
                print("-" * 40)
                for i in range(min(10, len(y_actual))):
                    diff = abs(y_actual[i] - y_pred[i])
                    print(f"{i:<5} {y_actual[i]:<10.2f} {y_pred[i]:<12.2f} {diff:<10.2f}")
                
                results.append({
                    'model': model_display_name,
                    'mse': mse,
                    'r2': r2,
                    'rmse': np.sqrt(mse)
                })
                
        except Exception as e:
            print(f"Error with {model_display_name}: {e}")

    print("\n" + "="*60)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*60)
    print(f"{'Model':<20} {'MSE':<12} {'R² Score':<12} {'RMSE':<12}")
    print("-" * 60)
    
    for result in sorted(results, key=lambda x: x['r2'], reverse=True):
        print(f"{result['model']:<20} {result['mse']:<12.2f} {result['r2']:<12.4f} {result['rmse']:<12.2f}")

if __name__ == "__main__":
    main()