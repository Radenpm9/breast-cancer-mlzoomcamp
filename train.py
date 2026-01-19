import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline

# --- 1. SETTINGS & HYPERPARAMETERS ---
# We use the 'best' parameters found during your Jupyter experimentation
INPUT_FILE = 'breast cancer wisconsin dataset.csv'
OUTPUT_FILE = 'cancer_model_v1.pkl'

RF_PARAMS = {
    'n_estimators': 200, 
    'max_depth': 10, 
    'min_samples_split': 2,
    'random_state': 42
}

XGB_PARAMS = {
    'n_estimators': 100, 
    'learning_rate': 0.1, 
    'max_depth': 3,
    'eval_metric': 'logloss',
    'random_state': 42
}

# --- 2. DATA PREPROCESSING ---
def load_and_clean_data(file_path):
    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path)
    
    # Remove unnecessary columns found in EDA
    if 'Unnamed: 32' in df.columns:
        df.drop('Unnamed: 32', axis=1, inplace=True)
    
    # Feature Selection: Dropping highly correlated features to simplify the model
    drop_list = [
        'radius_mean', 'perimeter_mean', 'compactness_mean', 'concave points_mean', 
        'radius_se', 'perimeter_se', 'radius_worst', 'perimeter_worst', 
        'compactness_worst', 'concave points_worst', 'compactness_se', 
        'concave points_se', 'texture_worst', 'area_worst', 'id'
    ]
    
    X = df.drop(['diagnosis'] + [col for col in drop_list if col in df.columns], axis=1)
    y = df['diagnosis']
    
    # Encode target: B -> 0, M -> 1
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    return X, y_encoded, le

# --- 3. MODEL TRAINING ---
def train_ensemble(X, y):
    print("Initializing models with tuned hyperparameters...")
    
    best_rf = RandomForestClassifier(**RF_PARAMS)
    best_xgb = XGBClassifier(**XGB_PARAMS)
    
    # Building the Stacking Classifier with internal scaling
    stacked_model = StackingClassifier(
        estimators=[
            ('rf', make_pipeline(StandardScaler(), best_rf)),
            ('xgb', make_pipeline(StandardScaler(), best_xgb))
        ],
        final_estimator=LogisticRegression(),
        cv=5
    )
    
    print("Training the Stacking Ensemble (fitting)...")
    stacked_model.fit(X, y)
    return stacked_model

# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    # A. Prepare Data
    X, y, le = load_and_clean_data(INPUT_FILE)
    
    # B. Split for a final validation check
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # C. Train
    final_model = train_ensemble(X_train, y_train)
    
    # D. Quick Accuracy Check
    accuracy = final_model.score(X_test, y_test)
    print(f"\n✅ Training Complete!")
    print(f"Final Validation Accuracy: {accuracy:.4f}")
    
    # E. Exporting for Deployment
    # We bundle the model and the labels so the API knows what '0' and '1' mean
    deployment_package = {
        "model": final_model,
        "labels": le.classes_
    }
    
    with open(OUTPUT_FILE, 'wb') as f_out:
        pickle.dump(deployment_package, f_out)
    
    print(f"📦 Model exported to: {OUTPUT_FILE}")