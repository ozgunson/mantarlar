import pandas as pd
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, fbeta_score
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_and_preprocess_data(filepath):
    """
    Loads the dataset and preprocesses it for machine learning.
    """
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath, sep=';')
    except:
        df = pd.read_csv(filepath, sep=',')

    print(f"Data loaded. Shape: {df.shape}")
    
    # Preprocessing
    le = LabelEncoder()
    mappings = {}
    
    # Handle missing values if any (simple imputation for now, or drop)
    # For this dataset, let's fill missing object columns with 'unknown' and numeric with mean
    # to ensure we use all data as requested.
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('unknown')
            df[col] = le.fit_transform(df[col].astype(str))
            mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))
        else:
            df[col] = df[col].fillna(df[col].mean())
            
    X = df.drop('class', axis=1)
    y = df['class']
    
    return X, y, mappings

def train_and_evaluate_models(X_train, X_test, y_train, y_test, output_dir):
    """
    Trains and evaluates multiple ML models, saving results to files.
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Linear SVM": LinearSVC(random_state=42, dual=False), # Faster than SVC for large datasets
        "KNN": KNeighborsClassifier(),
        "Gaussian Naive Bayes": GaussianNB(),
        "MLP Neural Network": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42)
    }
    
    comparison_results = []
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"\nStarting training and evaluation of {len(models)} models...")
    print(f"Results will be saved to: {output_dir}\n")
    
    for name, model in models.items():
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
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        f2 = fbeta_score(y_test, y_pred, beta=2, average='weighted')
        
        # Save detailed report
        report_filename = os.path.join(output_dir, f"{name.replace(' ', '_')}_report.txt")
        with open(report_filename, 'w') as f:
            f.write(f"Model: {name}\n")
            f.write(f"Training Time: {elapsed_time:.4f} seconds\n")
            f.write("-" * 30 + "\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"F2 Score: {f2:.4f}\n")
            f.write("-" * 30 + "\n")
            f.write("Confusion Matrix:\n")
            f.write(str(confusion_matrix(y_test, y_pred)))
            f.write("\n\n")
            f.write("Classification Report:\n")
            f.write(classification_report(y_test, y_pred))
            
        print(f"  -> Accuracy: {acc:.4f} (Saved to {report_filename})")
        
        comparison_results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
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

def run_experiment(X, y, test_size, output_dir_name):
    """
    Runs a single experiment with a specific test size and output directory.
    """
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_dir_name)
    
    print(f"\n{'='*40}")
    print(f"Running Experiment: {output_dir_name}")
    print(f"Test Size: {test_size} (Train: {1-test_size:.1f})")
    print(f"Output Directory: {output_path}")
    print(f"{'='*40}\n")
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Train and Evaluate
    results_df = train_and_evaluate_models(X_train, X_test, y_train, y_test, output_path)
    
    print(f"\n--- Model Comparison for {output_dir_name} ---")
    print(results_df.sort_values(by="Accuracy", ascending=False).to_string(index=False))

if __name__ == "__main__":
    # Construct path relative to this script's location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(current_dir, "../MushroomDataset/secondary_data.csv")
    
    # Load and Preprocess once
    X, y, mappings = load_and_preprocess_data(DATA_PATH)
    
    # Define experiments
    experiments = [
        {"test_size": 0.2, "output_dir": "results"},      # Original: 80% Train, 20% Test
        {"test_size": 0.4, "output_dir": "deney2"},       # Deney 2: 60% Train, 40% Test
        {"test_size": 0.3, "output_dir": "deney3"}        # Deney 3: 70% Train, 30% Test
    ]
    
    # Run all experiments
    for exp in experiments:
        run_experiment(X, y, exp["test_size"], exp["output_dir"])
