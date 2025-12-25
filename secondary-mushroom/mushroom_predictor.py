import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

class MushroomPredictor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = None
        self.le_dict = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_and_preprocess(self):
        """Loads data and prepares it for training."""
        print(f"Loading data from {self.data_path}...")
        try:
            df = pd.read_csv(self.data_path, sep=';')
        except:
            df = pd.read_csv(self.data_path, sep=',')
            
        # Encode categorical variables
        le = LabelEncoder()
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = le.fit_transform(df[col].astype(str))
                self.le_dict[col] = le
                
        X = df.drop('class', axis=1)
        y = df['class']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("Data loaded and preprocessed.")
        
    def train(self):
        """Trains a Random Forest model."""
        if self.X_train is None:
            self.load_and_preprocess()
            
        print("Training Random Forest model...")
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(self.X_train, self.y_train)
        print("Training complete.")
        
    def evaluate(self):
        """Evaluates the trained model."""
        if self.model is None:
            print("Model not trained yet.")
            return
            
        y_pred = self.model.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        print(f"Model Accuracy: {acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        
    def predict(self, features):
        """
        Predicts the class for a given set of features.
        Expected features: dict or list matching the column order.
        """
        if self.model is None:
            print("Model not trained yet.")
            return None
            
        # This is a simplified prediction that assumes input is already encoded or matches format
        # In a real app, we would need to apply the saved LabelEncoders to the input features
        # For demonstration, we'll assume the input is compatible or raw values that need encoding
        
        # TODO: Implement robust feature encoding for single prediction
        # For now, we will just pass the features directly if they are numerical/encoded
        
        prediction = self.model.predict([features])
        return prediction[0]

if __name__ == "__main__":
    # Example usage
    current_dir = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(current_dir, "../MushroomDataset/secondary_data.csv")
    predictor = MushroomPredictor(DATA_PATH)
    predictor.train()
    predictor.evaluate()
