import pandas as pd
import numpy as np
import os
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, fbeta_score
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_and_preprocess_data(filepath):
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath, sep=';')
    except:
        df = pd.read_csv(filepath, sep=',')

    # Handle missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('unknown')
        else:
            df[col] = df[col].fillna(df[col].mean())

    X = df.drop('class', axis=1)
    y = df['class']
    
    le_y = LabelEncoder()
    y = le_y.fit_transform(y)
    
    cat_cols = X.select_dtypes(include=['object']).columns
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])
    
    return X, y

def create_cnn_model(input_shape, filters=32, kernel_size=3):
    model = keras.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Reshape((input_shape, 1)),
        layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_bidirectional_lstm_model(input_shape, units=64, dropout=0.0):
    model = keras.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Reshape((input_shape, 1)),
        layers.Bidirectional(layers.LSTM(units, return_sequences=False, dropout=dropout)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_resnet_model(input_shape, dense_units=64):
    inputs = layers.Input(shape=(input_shape,))
    
    x = layers.Dense(dense_units, activation='relu')(inputs)
    
    # Residual Block 1
    res = x
    x = layers.Dense(dense_units, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(dense_units, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, res])
    
    # Residual Block 2
    res = x
    x = layers.Dense(dense_units, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(dense_units, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, res])
    
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate_dl_variations(X_train, X_test, y_train, y_test, output_dir):
    input_shape = X_train.shape[1]
    comparison_results = []
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Define Variations for 1D-CNN, Bi-LSTM, ResNet
    variations = [
        # 1D-CNN Variations (Filters)
        {"name": "CNN_Filters_16", "model_fn": create_cnn_model, "kwargs": {"filters": 16}, "fit_kwargs": {"batch_size": 64}},
        {"name": "CNN_Filters_32", "model_fn": create_cnn_model, "kwargs": {"filters": 32}, "fit_kwargs": {"batch_size": 64}},
        {"name": "CNN_Filters_64", "model_fn": create_cnn_model, "kwargs": {"filters": 64}, "fit_kwargs": {"batch_size": 64}},
        
        # Bi-LSTM Variations (Units)
        {"name": "BiLSTM_Units_32", "model_fn": create_bidirectional_lstm_model, "kwargs": {"units": 32}, "fit_kwargs": {"batch_size": 64}},
        {"name": "BiLSTM_Units_64", "model_fn": create_bidirectional_lstm_model, "kwargs": {"units": 64}, "fit_kwargs": {"batch_size": 64}},
        {"name": "BiLSTM_Units_128", "model_fn": create_bidirectional_lstm_model, "kwargs": {"units": 128}, "fit_kwargs": {"batch_size": 64}},
        
        # ResNet Variations (Dense Units)
        {"name": "ResNet_Dense_32", "model_fn": create_resnet_model, "kwargs": {"dense_units": 32}, "fit_kwargs": {"batch_size": 64}},
        {"name": "ResNet_Dense_64", "model_fn": create_resnet_model, "kwargs": {"dense_units": 64}, "fit_kwargs": {"batch_size": 64}},
        {"name": "ResNet_Dense_128", "model_fn": create_resnet_model, "kwargs": {"dense_units": 128}, "fit_kwargs": {"batch_size": 64}}
    ]
    
    print(f"\nStarting Refined DL Parameter Sensitivity Analysis ({len(variations)} variations)...")
    
    for var in variations:
        name = var["name"]
        print(f"Training {name}...")
        
        # Create Model
        model = var["model_fn"](input_shape, **var["kwargs"])
        
        start_time = time.time()
        
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        
        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=20, 
            callbacks=[early_stopping],
            verbose=0,
            **var["fit_kwargs"]
        )
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Evaluate
        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        f2 = fbeta_score(y_test, y_pred, beta=2, average='weighted')
        
        # Save Report
        report_filename = os.path.join(output_dir, f"{name}_report.txt")
        with open(report_filename, 'w') as f:
            f.write(f"Model Variation: {name}\n")
            f.write(f"Params: {var['kwargs']} | Fit: {var['fit_kwargs']}\n")
            f.write("-" * 30 + "\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"F2 Score: {f2:.4f}\n")
            f.write(f"Epochs Run: {len(history.history['loss'])}\n")
            
        print(f"  -> Accuracy: {acc:.4f}")
        
        # Save Model
        model.save(os.path.join(output_dir, f"{name}.keras"))
        
        comparison_results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "F2 Score": f2,
            "Training Time (s)": elapsed_time
        })
        
    # Save CSV
    comparison_df = pd.DataFrame(comparison_results)
    comparison_file = os.path.join(output_dir, "dl_model_comparison.csv")
    comparison_df.to_csv(comparison_file, index=False)
    print(f"\nDL Comparison summary saved to: {comparison_file}")
    
    return comparison_df

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(current_dir, "../MushroomDataset/secondary_data.csv")
    OUTPUT_DIR = os.path.join(current_dir, "deney4/deep_learning_results")
    
    # Clean up previous runs if needed (optional)
    
    X, y = load_and_preprocess_data(DATA_PATH)
    
    # Fixed Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results_df = train_and_evaluate_dl_variations(X_train, X_test, y_train, y_test, OUTPUT_DIR)
    
    print("\n--- Refined DL Parameter Sensitivity Results ---")
    print(results_df.sort_values(by="Accuracy", ascending=False).to_string(index=False))
