import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

# 1. SETTINGS
MODEL_FILE = 'cancer_model_v1.pkl'
DATA_FILE = 'breast cancer wisconsin dataset.csv'

def run_test():
    print(f"--- Starting Model Test for {MODEL_FILE} ---")

    # 2. LOAD THE DEPLOYMENT PACKAGE
    try:
        with open(MODEL_FILE, 'rb') as f_in:
            package = pickle.load(f_in)
        
        model = package['model']
        labels = package['labels']
        print("✅ Model and Labels loaded successfully.")
    except FileNotFoundError:
        print(f"❌ Error: {MODEL_FILE} not found. Run train.py first!")
        return

    # 3. PREPARE TEST DATA
    # We use the same cleaning logic as your notebook
    df = pd.read_csv(DATA_FILE)
    
    # Drop columns to match the features the model was trained on
    drop_list = [
        'radius_mean', 'perimeter_mean', 'compactness_mean', 'concave points_mean', 
        'radius_se', 'perimeter_se', 'radius_worst', 'perimeter_worst', 
        'compactness_worst', 'concave points_worst', 'compactness_se', 
        'concave points_se', 'texture_worst', 'area_worst', 'id', 'Unnamed: 32'
    ]
    
    # Separate features and target
    X = df.drop(['diagnosis'] + [col for col in drop_list if col in df.columns], axis=1)
    y_actual = df['diagnosis']

    # 4. RUN PREDICTIONS
    print(f"Running predictions on {len(X)} samples...")
    y_pred_indices = model.predict(X)
    
    # Convert numeric predictions (0, 1) back to 'B' and 'M'
    y_pred_labels = [labels[i] for i in y_pred_indices]

    # 5. EVALUATE
    acc = accuracy_score(y_actual, y_pred_labels)
    print(f"\n--- Test Results ---")
    print(f"Overall Accuracy: {acc:.4f}")
    print("\nDetailed Report:")
    print(classification_report(y_actual, y_pred_labels))

    # Test one specific sample (the first one)
    sample_result = y_pred_labels[0]
    print(f"Sample 0 Test: Actual={y_actual[0]}, Predicted={sample_result}")

if __name__ == "__main__":
    run_test()