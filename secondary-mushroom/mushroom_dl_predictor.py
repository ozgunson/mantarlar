import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

class MushroomDLPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        
    def load_model(self):
        """Loads the saved Keras model."""
        if os.path.exists(self.model_path):
            print(f"Loading model from {self.model_path}...")
            self.model = keras.models.load_model(self.model_path)
            print("Model loaded successfully.")
        else:
            print(f"Error: Model file not found at {self.model_path}")
            
    def predict(self, features):
        """
        Predicts class for given features.
        Features should be a preprocessed numpy array or list matching input shape.
        """
        if self.model is None:
            self.load_model()
            
        if self.model is None:
            return None
            
        # Ensure features are in correct shape (1, n_features)
        features = np.array(features).reshape(1, -1)
        
        prediction_prob = self.model.predict(features, verbose=0)
        predicted_class = (prediction_prob > 0.5).astype(int)[0][0]
        
        return predicted_class, prediction_prob[0][0]

if __name__ == "__main__":
    # Example usage
    current_dir = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(current_dir, "deep_learning_results/DNN_Baseline.keras")
    
    # Mock feature vector (random for demonstration)
    # In real usage, this must be preprocessed exactly like training data
    # 20 features is typical for this dataset after encoding? 
    # Actually after label encoding it's 20 features.
    mock_features = np.random.rand(20) 
    
    predictor = MushroomDLPredictor(MODEL_PATH)
    pred_class, prob = predictor.predict(mock_features)
    
    if pred_class is not None:
        print(f"Predicted Class: {pred_class} (Probability: {prob:.4f})")
