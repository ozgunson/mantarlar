import pandas as pd
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, fbeta_score
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(filepath):
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath, sep=';')
    except:
        df = pd.read_csv(filepath, sep=',')

    print(f"Data loaded. Shape: {df.shape}")
    
    # Preprocessing
    le = LabelEncoder()
    
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('unknown')
            df[col] = le.fit_transform(df[col].astype(str))
        else:
            df[col] = df[col].fillna(df[col].mean())
            
    X = df.drop('class', axis=1)
    y = df['class']
    
    return X, y

def train_and_evaluate_param_variations(X_train, X_test, y_train, y_test, output_dir):
    """
    Trains models with parameter variations.
    """
    
    # Define hyperparameter variations for requested algorithms
    models_config = [
        # Logistic Regression Variations
        ("Logistic_Reg_C0.1", LogisticRegression(C=0.1, max_iter=1000)),
        ("Logistic_Reg_C1.0", LogisticRegression(C=1.0, max_iter=1000)),
        ("Logistic_Reg_C10.0", LogisticRegression(C=10.0, max_iter=1000)),
        
        # Extra Trees Variations
        ("Extra_Trees_n50", ExtraTreesClassifier(n_estimators=50, random_state=42)),
        ("Extra_Trees_n100", ExtraTreesClassifier(n_estimators=100, random_state=42)),
        ("Extra_Trees_n200", ExtraTreesClassifier(n_estimators=200, random_state=42)),
        
        # Random Forest Variations
        ("Random_Forest_n50", RandomForestClassifier(n_estimators=50, random_state=42)),
        ("Random_Forest_n100", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("Random_Forest_n200", RandomForestClassifier(n_estimators=200, random_state=42))
    ]
    
    comparison_results = []
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"\nStarting Parameter Sensitivity Analysis ({len(models_config)} configurations)...")
    print(f"Results will be saved to: {output_dir}\n")
    
    for name, model in models_config:
        print(f"Training {name}...")
        start_time = time.time()
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        f2 = fbeta_score(y_test, y_pred, beta=2, average='weighted')
        
        # Save Report
        report_filename = os.path.join(output_dir, f"{name}_report.txt")
        with open(report_filename, 'w') as f:
            f.write(f"Model Variation: {name}\n")
            f.write(f"Training Time: {elapsed_time:.4f} seconds\n")
            f.write("-" * 30 + "\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"F2 Score: {f2:.4f}\n")
            f.write("-" * 30 + "\n")
            f.write("Classification Report:\n")
            f.write(classification_report(y_test, y_pred))
            
        print(f"  -> Accuracy: {acc:.4f}")
        
        comparison_results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "F2 Score": f2,
            "Training Time (s)": elapsed_time
        })
        
    # Save comparison CSV
    comparison_df = pd.DataFrame(comparison_results)
    comparison_file = os.path.join(output_dir, "model_comparison.csv")
    comparison_df.to_csv(comparison_file, index=False)
    print(f"\nComparison summary saved to: {comparison_file}")
    
    return comparison_df

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(current_dir, "../MushroomDataset/secondary_data.csv")
    OUTPUT_DIR = os.path.join(current_dir, "deney4")
    
    X, y = load_and_preprocess_data(DATA_PATH)
    
    # Fixed Split for Sensitivity Analysis: 80% Train, 20% Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results_df = train_and_evaluate_param_variations(X_train, X_test, y_train, y_test, OUTPUT_DIR)
    
    print("\n--- Parameter Sensitivity Results ---")
    print(results_df.sort_values(by="Accuracy", ascending=False).to_string(index=False))
