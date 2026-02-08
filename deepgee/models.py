"""
Deep learning models for Earth observation
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score
import joblib

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available. Install with: pip install tensorflow")


class LandCoverClassifier:
    """
    Deep learning classifier for land cover classification.
    """
    
    def __init__(
        self,
        n_classes: int,
        architecture: str = 'dense',
        input_shape: Optional[Tuple[int]] = None
    ):
        """
        Initialize land cover classifier.
        
        Parameters:
        -----------
        n_classes : int
            Number of land cover classes
        architecture : str
            Model architecture: 'dense', 'cnn1d', 'simple'
        input_shape : tuple, optional
            Input shape (n_features,) for dense, (n_features, 1) for cnn1d
        
        Examples:
        ---------
        >>> classifier = LandCoverClassifier(n_classes=9, architecture='dense')
        >>> classifier.build_model(input_shape=(14,))
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required. Install with: pip install tensorflow")
        
        self.n_classes = n_classes
        self.architecture = architecture
        self.input_shape = input_shape
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
        self.class_names = None
    
    def build_model(self, input_shape: Optional[Tuple[int]] = None) -> keras.Model:
        """
        Build the neural network model.
        
        Parameters:
        -----------
        input_shape : tuple, optional
            Input shape
        
        Returns:
        --------
        keras.Model : Built model
        """
        if input_shape is not None:
            self.input_shape = input_shape
        
        if self.input_shape is None:
            raise ValueError("input_shape must be provided")
        
        if self.architecture == 'dense':
            self.model = self._build_dense_model()
        elif self.architecture == 'cnn1d':
            self.model = self._build_cnn1d_model()
        elif self.architecture == 'simple':
            self.model = self._build_simple_model()
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")
        
        return self.model
    
    def _build_dense_model(self) -> keras.Model:
        """Build dense neural network."""
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=self.input_shape),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            
            keras.layers.Dense(64, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            
            keras.layers.Dense(32, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            
            keras.layers.Dense(self.n_classes, activation='softmax')
        ], name='DenseClassifier')
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _build_cnn1d_model(self) -> keras.Model:
        """Build 1D CNN model."""
        model = keras.Sequential([
            keras.layers.Conv1D(64, 3, activation='relu', input_shape=self.input_shape),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling1D(2),
            keras.layers.Dropout(0.3),
            
            keras.layers.Conv1D(32, 3, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.GlobalMaxPooling1D(),
            keras.layers.Dropout(0.3),
            
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(self.n_classes, activation='softmax')
        ], name='CNN1DClassifier')
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _build_simple_model(self) -> keras.Model:
        """Build simple neural network."""
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=self.input_shape),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(self.n_classes, activation='softmax')
        ], name='SimpleClassifier')
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42,
        normalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training.
        
        Parameters:
        -----------
        X : np.ndarray
            Features
        y : np.ndarray
            Labels
        test_size : float
            Test set proportion
        random_state : int
            Random seed
        normalize : bool
            Normalize features
        
        Returns:
        --------
        tuple : X_train, X_test, y_train, y_test
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Normalize
        if normalize:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
        
        # Reshape for CNN if needed
        if self.architecture == 'cnn1d':
            X_train = X_train.reshape(-1, X_train.shape[1], 1)
            X_test = X_test.reshape(-1, X_test.shape[1], 1)
        
        return X_train, X_test, y_train, y_test
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        validation_split: float = 0.2,
        epochs: int = 100,
        batch_size: int = 64,
        callbacks: Optional[List] = None,
        verbose: int = 1
    ) -> keras.callbacks.History:
        """
        Train the model.
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels
        validation_split : float
            Validation split proportion
        epochs : int
            Number of epochs
        batch_size : int
            Batch size
        callbacks : list, optional
            Keras callbacks
        verbose : int
            Verbosity level
        
        Returns:
        --------
        History : Training history
        
        Examples:
        ---------
        >>> classifier.train(X_train, y_train, epochs=50)
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        if callbacks is None:
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7,
                    verbose=1
                )
            ]
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return self.history
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        class_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Evaluate the model.
        
        Parameters:
        -----------
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test labels
        class_names : list, optional
            Class names for report
        
        Returns:
        --------
        dict : Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        # Predictions
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = y_pred_proba.argmax(axis=1)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        results = {
            'accuracy': accuracy,
            'kappa': kappa,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        if class_names is not None:
            self.class_names = class_names
            results['classification_report'] = classification_report(
                y_test, y_pred, target_names=class_names
            )
        
        return results
    
    def predict(
        self,
        X: np.ndarray,
        normalize: bool = True,
        batch_size: int = 10000
    ) -> np.ndarray:
        """
        Make predictions on new data.
        
        Parameters:
        -----------
        X : np.ndarray
            Features to predict
        normalize : bool
            Normalize features using fitted scaler
        batch_size : int
            Batch size for prediction
        
        Returns:
        --------
        np.ndarray : Predicted class labels
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        # Normalize
        if normalize:
            X = self.scaler.transform(X)
        
        # Reshape for CNN if needed
        if self.architecture == 'cnn1d':
            X = X.reshape(-1, X.shape[1], 1)
        
        # Predict in batches
        predictions = []
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size]
            pred = self.model.predict(batch, verbose=0)
            predictions.append(pred)
        
        predictions = np.concatenate(predictions, axis=0)
        return predictions.argmax(axis=1)
    
    def save(self, model_path: str, scaler_path: str) -> None:
        """
        Save model and scaler.
        
        Parameters:
        -----------
        model_path : str
            Path to save model (.h5 or .keras)
        scaler_path : str
            Path to save scaler (.pkl)
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"✓ Model saved to: {model_path}")
        print(f"✓ Scaler saved to: {scaler_path}")
    
    def load(self, model_path: str, scaler_path: str) -> None:
        """
        Load model and scaler.
        
        Parameters:
        -----------
        model_path : str
            Path to model file
        scaler_path : str
            Path to scaler file
        """
        self.model = keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        print(f"✓ Model loaded from: {model_path}")
        print(f"✓ Scaler loaded from: {scaler_path}")


class ChangeDetector:
    """
    Change detection using deep learning.
    """
    
    def __init__(self, method: str = 'difference'):
        """
        Initialize change detector.
        
        Parameters:
        -----------
        method : str
            Detection method: 'difference', 'ratio', 'pca', 'neural'
        """
        self.method = method
        self.model = None
    
    def detect_changes(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Detect changes between two images.
        
        Parameters:
        -----------
        image1 : np.ndarray
            First image (time 1)
        image2 : np.ndarray
            Second image (time 2)
        threshold : float, optional
            Change threshold
        
        Returns:
        --------
        np.ndarray : Binary change map
        """
        if self.method == 'difference':
            change = np.abs(image2 - image1)
        elif self.method == 'ratio':
            change = image2 / (image1 + 1e-10)
        else:
            raise NotImplementedError(f"Method {self.method} not implemented")
        
        if threshold is not None:
            change_binary = (change > threshold).astype(np.uint8)
            return change_binary
        else:
            return change
    
    def calculate_change_statistics(
        self,
        change_map: np.ndarray,
        pixel_area: float = 900.0
    ) -> Dict:
        """
        Calculate change statistics.
        
        Parameters:
        -----------
        change_map : np.ndarray
            Binary change map
        pixel_area : float
            Area per pixel in square meters
        
        Returns:
        --------
        dict : Change statistics
        """
        total_pixels = change_map.size
        changed_pixels = np.sum(change_map)
        unchanged_pixels = total_pixels - changed_pixels
        
        changed_area_m2 = changed_pixels * pixel_area
        changed_area_km2 = changed_area_m2 / 1e6
        
        change_percentage = (changed_pixels / total_pixels) * 100
        
        return {
            'total_pixels': int(total_pixels),
            'changed_pixels': int(changed_pixels),
            'unchanged_pixels': int(unchanged_pixels),
            'changed_area_m2': float(changed_area_m2),
            'changed_area_km2': float(changed_area_km2),
            'change_percentage': float(change_percentage)
        }
