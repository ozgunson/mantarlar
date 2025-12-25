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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF info logs

def load_and_preprocess_data(filepath):
    """
    Loads the dataset and preprocesses it for Deep Learning.
    Includes Label Encoding for categoricals and Scaling for numericals.
    """
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath, sep=';')
    except:
        df = pd.read_csv(filepath, sep=',')

    print(f"Data loaded. Shape: {df.shape}")
    
    # Handle missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('unknown')
        else:
            df[col] = df[col].fillna(df[col].mean())

    # Separate target
    X = df.drop('class', axis=1)
    y = df['class']
    
    # Encode Target
    le_y = LabelEncoder()
    y = le_y.fit_transform(y) # e=0, p=1 usually
    
    # Preprocessing Features
    le_dict = {}
    cat_cols = X.select_dtypes(include=['object']).columns
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le
        
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])
    
    return X, y, le_y, scaler

def create_baseline_model(input_shape):
    model = keras.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_deep_model(input_shape):
    model = keras.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_wide_model(input_shape):
    model = keras.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(256, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_cnn_model(input_shape):
    model = keras.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Reshape((input_shape, 1)),
        layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_lstm_model(input_shape):
    model = keras.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Reshape((input_shape, 1)),
        layers.LSTM(64, return_sequences=False),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_gru_model(input_shape):
    model = keras.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Reshape((input_shape, 1)),
        layers.GRU(64, return_sequences=False),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_bidirectional_lstm_model(input_shape):
    model = keras.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Reshape((input_shape, 1)),
        layers.Bidirectional(layers.LSTM(64, return_sequences=False)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_resnet_model(input_shape):
    inputs = layers.Input(shape=(input_shape,))
    
    # First dense layer
    x = layers.Dense(64, activation='relu')(inputs)
    
    # Residual Block 1
    res = x
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, res])
    
    # Residual Block 2
    res = x
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, res])
    
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate_dl_models(X_train, X_test, y_train, y_test, output_dir):
    input_shape = X_train.shape[1]
    
    models = {
        "DNN_Baseline": create_baseline_model(input_shape),
        "DNN_Deep_Regularized": create_deep_model(input_shape),
        "DNN_Wide": create_wide_model(input_shape),
        "CNN_1D": create_cnn_model(input_shape),
        "LSTM": create_lstm_model(input_shape),
        "GRU": create_gru_model(input_shape),
        "Bidirectional_LSTM": create_bidirectional_lstm_model(input_shape),
        "ResNet": create_resnet_model(input_shape)
    }
    
    comparison_results = []
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"\nStarting Deep Learning training of {len(models)} models...")
    print(f"Results will be saved to: {output_dir}\n")
    
    for name, model in models.items():
        print(f"Training {name}...")
        start_time = time.time()
        
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=5, 
            restore_best_weights=True
        )
        
        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=30,
            batch_size=64,
            callbacks=[early_stopping],
            verbose=0
        )
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        f2 = fbeta_score(y_test, y_pred, beta=2, average='weighted')
        
        report_filename = os.path.join(output_dir, f"{name}_report.txt")
        with open(report_filename, 'w') as f:
            f.write(f"Model: {name}\n")
            f.write(f"Training Time: {elapsed_time:.4f} seconds\n")
            f.write(f"Epochs Run: {len(history.history['loss'])}\n")
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
        
        model_save_path = os.path.join(output_dir, f"{name}.keras")
        model.save(model_save_path)
        
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
        
    comparison_df = pd.DataFrame(comparison_results)
    comparison_file = os.path.join(output_dir, "dl_model_comparison.csv")
    comparison_df.to_csv(comparison_file, index=False)
    print(f"\nDL Comparison summary saved to: {comparison_file}")
    
    return comparison_df

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(current_dir, "../MushroomDataset/secondary_data.csv")
    OUTPUT_DIR = os.path.join(current_dir, "deep_learning_results")
    
    X, y, le_y, scaler = load_and_preprocess_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results_df = train_and_evaluate_dl_models(X_train, X_test, y_train, y_test, OUTPUT_DIR)
    
    print("\n--- Final DL Model Comparison ---")
    print(results_df.sort_values(by="Accuracy", ascending=False).to_string(index=False))
