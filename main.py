"""
YAQSHA: Quantum Financial Fraud Detector
A quantum-enhanced anomaly detection system for financial transactions
using PennyLane, Python, and Pandas
"""

import pandas as pd
import numpy as np
import pennylane as qml
from pennylane import numpy as qnp
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Any

# Set random seed for reproducibility
np.random.seed(42)

class YaqshaQuantumFraudDetector:
    """
    Quantum-enhanced fraud detection system using PennyLane
    """
    
    def __init__(
        self, 
        n_qubits: int = 4, 
        n_layers: int = 2,
        device_type: str = "default.qubit"
    ):
        """
        Initialize the quantum fraud detector
        
        Args:
            n_qubits: Number of qubits to use in the quantum circuit
            n_layers: Number of variational layers in the circuit
            device_type: Type of quantum device to use
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device_type = device_type
        self.device = qml.device(device_type, wires=n_qubits)
        self.weights = None
        
        # Define the quantum circuit as a QNode
        self.qnode = qml.QNode(self._quantum_circuit, self.device, interface="autograd")
        
    def _feature_encoding(self, x: np.ndarray) -> None:
        """
        Encode classical features into quantum states
        
        Args:
            x: Feature vector to encode
        """
        # Angle embedding of features
        qml.AngleEmbedding(features=x, wires=range(self.n_qubits))
    
    def _variational_layer(self, weights: np.ndarray, layer_idx: int) -> None:
        """
        Create a variational layer with parameterized gates
        
        Args:
            weights: Variational parameters
            layer_idx: Layer index
        """
        # Apply rotation gates
        for qubit in range(self.n_qubits):
            qml.RX(weights[layer_idx, qubit, 0], wires=qubit)
            qml.RY(weights[layer_idx, qubit, 1], wires=qubit)
            qml.RZ(weights[layer_idx, qubit, 2], wires=qubit)
        
        # Apply entangling gates - creating the quantum advantage
        for qubit in range(self.n_qubits - 1):
            qml.CNOT(wires=[qubit, qubit + 1])
        
        # Create a cycle
        qml.CNOT(wires=[self.n_qubits - 1, 0])
            
    def _quantum_circuit(self, x: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Define the quantum circuit architecture
        
        Args:
            x: Input feature vector
            weights: Variational parameters
            
        Returns:
            Measurement results from the quantum circuit
        """
        # Encode features into the quantum state
        self._feature_encoding(x)
        
        # Apply variational layers
        for layer in range(self.n_layers):
            self._variational_layer(weights, layer)
        
        # Measure Pauli-Z expectation for each qubit
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
    
    def _weight_shapes(self) -> Dict[str, tuple]:
        """
        Define the shape of the weights for the quantum circuit
        
        Returns:
            Dictionary specifying the weight shapes
        """
        return {
            "weights": (self.n_layers, self.n_qubits, 3)  # layers, qubits, rotation gates (X, Y, Z)
        }
    
    def _cost_function(self, weights: np.ndarray, X_batch: np.ndarray, y_batch: np.ndarray) -> float:
        """
        Binary cross-entropy loss function
        
        Args:
            weights: Circuit parameters
            X_batch: Batch of feature vectors
            y_batch: Batch of labels
            
        Returns:
            Loss value
        """
        predictions = np.array([self.predict_single(x, weights) for x in X_batch])
        
        # Binary cross-entropy loss
        epsilon = 1e-10  # Small constant to avoid log(0)
        loss = -np.mean(y_batch * np.log(predictions + epsilon) + 
                        (1 - y_batch) * np.log(1 - predictions + epsilon))
        return loss
    
    def predict_single(self, x: np.ndarray, weights: np.ndarray = None) -> float:
        """
        Make prediction for a single sample
        
        Args:
            x: Feature vector
            weights: Optional circuit parameters (uses self.weights if None)
            
        Returns:
            Probability of fraud
        """
        if weights is None:
            if self.weights is None:
                raise ValueError("Model not trained yet, weights are not set")
            weights = self.weights
        
        # Ensure x is the right dimension for our circuit
        x = self._prepare_input(x)
        
        # Get quantum measurements
        results = self.qnode(x, weights)
        
        # Convert to binary classification probability
        # Use the first qubit measurement as the main signal
        # Scale from [-1, 1] (PauliZ range) to [0, 1] (probability)
        return (results[0] + 1) / 2
    
    def _prepare_input(self, x: np.ndarray) -> np.ndarray:
        """
        Prepare input features for quantum processing
        
        Args:
            x: Input feature vector
            
        Returns:
            Processed feature vector suitable for quantum encoding
        """
        # Ensure x has the right dimension
        if len(x) > self.n_qubits:
            # If too many features, select the most important ones or use PCA
            # For simplicity, we'll just select the first n_qubits features
            x = x[:self.n_qubits]
        elif len(x) < self.n_qubits:
            # If too few features, pad with zeros
            x = np.pad(x, (0, self.n_qubits - len(x)))
        
        return x
    
    def fit(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        n_epochs: int = 100, 
        batch_size: int = 32, 
        learning_rate: float = 0.01
    ) -> List[float]:
        """
        Train the quantum circuit using gradient descent
        
        Args:
            X_train: Training features
            y_train: Training labels
            n_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimization
            
        Returns:
            List of loss values during training
        """
        # Initialize random weights
        init_shape = self._weight_shapes()
        weights = np.random.uniform(
            low=0, high=2*np.pi, 
            size=init_shape["weights"]
        )
        
        # Use Adam optimizer
        opt = qml.AdamOptimizer(stepsize=learning_rate)
        
        # Track loss history
        loss_history = []
        
        n_samples = len(X_train)
        
        print("Starting quantum model training...")
        for epoch in range(n_epochs):
            # Shuffle the data
            shuffle_idx = np.random.permutation(n_samples)
            X_shuffled = X_train[shuffle_idx]
            y_shuffled = y_train[shuffle_idx]
            
            # Mini-batch training
            for batch_start in range(0, n_samples, batch_size):
                batch_end = min(batch_start + batch_size, n_samples)
                X_batch = X_shuffled[batch_start:batch_end]
                y_batch = y_shuffled[batch_start:batch_end]
                
                # Apply gradient descent step
                weights, loss = opt.step_and_cost(
                    lambda w: self._cost_function(w, X_batch, y_batch), 
                    weights
                )
                
            # Calculate and store the loss for the entire dataset
            loss = self._cost_function(weights, X_train, y_train)
            loss_history.append(loss)
            
            if epoch % 10 == 0 or epoch == n_epochs - 1:
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss:.4f}")
        
        # Store the final weights
        self.weights = weights
        
        print("Training completed!")
        return loss_history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions for multiple samples
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions
        """
        if self.weights is None:
            raise ValueError("Model not trained yet")
        
        predictions = np.array([self.predict_single(x) for x in X])
        return predictions
    
    def predict_binary(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Make binary predictions
        
        Args:
            X: Feature matrix
            threshold: Classification threshold
            
        Returns:
            Binary predictions (0 or 1)
        """
        probabilities = self.predict(X)
        return (probabilities >= threshold).astype(int)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Evaluate the model performance
        
        Args:
            X_test: Test features
            y_test: Test labels
            threshold: Classification threshold
            
        Returns:
            Dictionary with performance metrics
        """
        # Get predictions
        y_pred_proba = self.predict(X_test)
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm,
            "predictions": y_pred,
            "probabilities": y_pred_proba
        }

    def plot_training_loss(self, loss_history: List[float]) -> None:
        """
        Plot the training loss curve
        
        Args:
            loss_history: List of loss values during training
        """
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()
        
    def plot_confusion_matrix(self, confusion_matrix: np.ndarray) -> None:
        """
        Plot a confusion matrix
        
        Args:
            confusion_matrix: Confusion matrix to plot
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            confusion_matrix, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Normal', 'Fraud'],
            yticklabels=['Normal', 'Fraud']
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()


class DataProcessor:
    """
    Process financial transaction data for quantum fraud detection
    """
    
    def __init__(self, target_col: str = 'Class'):
        """
        Initialize the data processor
        
        Args:
            target_col: Name of the target column (fraud indicator)
        """
        self.target_col = target_col
        self.scaler = StandardScaler()
        self.feature_scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))  # Scale for quantum encoding
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load financial transaction data
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            DataFrame with the loaded data
        """
        return pd.read_csv(file_path)
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess the data for fraud detection
        
        Args:
            df: Raw transaction data
            
        Returns:
            Tuple of (features_df, target_series)
        """
        print("Starting data preprocessing...")
        
        # Separate features and target
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Remove any constant columns
        const_cols = [col for col in X.columns if X[col].nunique() <= 1]
        X = X.drop(columns=const_cols)
        
        # Standardize the features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns
        )
        
        print(f"Preprocessing completed. Features shape: {X_scaled.shape}")
        
        return X_scaled, y
    
    def prepare_for_quantum(self, X: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for quantum encoding
        
        Args:
            X: Feature DataFrame
            
        Returns:
            NumPy array scaled for quantum encoding
        """
        # Scale features to be in range [-π, π] for angle encoding
        return self.feature_scaler.fit_transform(X)
    
    def train_test_validation_split(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        test_size: float = 0.2, 
        val_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train, test, and validation sets
        
        Args:
            X: Feature DataFrame
            y: Target Series
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: separate test set
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Second split: separate validation from training
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, 
            test_size=val_size/(1-test_size), 
            random_state=42, 
            stratify=y_train_val
        )
        
        # Prepare for quantum processing
        X_train_q = self.prepare_for_quantum(X_train)
        X_val_q = self.prepare_for_quantum(X_val)
        X_test_q = self.prepare_for_quantum(X_test)
        
        return X_train_q, X_val_q, X_test_q, y_train.values, y_val.values, y_test.values
    
    def dimensionality_reduction(self, X: pd.DataFrame, n_components: int = 4) -> pd.DataFrame:
        """
        Reduce dimensionality of the data to fit quantum circuit
        
        Args:
            X: Feature DataFrame
            n_components: Number of components to keep
            
        Returns:
            DataFrame with reduced dimensions
        """
        # For simplicity, we'll use PCA from scikit-learn
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X)
        
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.4f}")
        
        return pd.DataFrame(X_reduced, columns=[f'PC{i+1}' for i in range(n_components)])
    
    def feature_importance_selection(self, X: pd.DataFrame, y: pd.Series, n_features: int = 4) -> pd.DataFrame:
        """
        Select most important features using a classical ML model
        
        Args:
            X: Feature DataFrame
            y: Target Series
            n_features: Number of features to select
            
        Returns:
            DataFrame with selected features
        """
        from sklearn.ensemble import RandomForestClassifier
        
        # Train a random forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Get feature importances
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Select top features
        selected_features = X.columns[indices[:n_features]]
        
        print("Selected features:")
        for i, feature in enumerate(selected_features):
            print(f"{i+1}. {feature} (Importance: {importances[indices[i]]:.4f})")
        
        return X[selected_features]


def load_and_sample_credit_card_data(
    file_path: str, 
    fraud_sample_ratio: float = 1.0, 
    normal_sample_ratio: float = 0.1,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Load credit card fraud data and create a balanced sample
    
    Args:
        file_path: Path to the CSV file
        fraud_sample_ratio: Ratio of fraud transactions to keep
        normal_sample_ratio: Ratio of normal transactions to keep
        random_state: Random seed for sampling
        
    Returns:
        Balanced DataFrame
    """
    # Load the data
    df = pd.read_csv(file_path)
    
    # Split into fraud and normal transactions
    fraud_df = df[df['Class'] == 1]
    normal_df = df[df['Class'] == 0]
    
    # Sample from each class
    n_fraud = int(len(fraud_df) * fraud_sample_ratio)
    n_normal = int(len(normal_df) * normal_sample_ratio)
    
    fraud_sample = fraud_df.sample(n=n_fraud, random_state=random_state)
    normal_sample = normal_df.sample(n=n_normal, random_state=random_state)
    
    # Combine the samples
    balanced_df = pd.concat([fraud_sample, normal_sample])
    
    # Shuffle the data
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    print(f"Data loaded and sampled:")
    print(f"Total transactions: {len(balanced_df)}")
    print(f"Fraud transactions: {len(balanced_df[balanced_df['Class'] == 1])} ({len(balanced_df[balanced_df['Class'] == 1])/len(balanced_df):.2%})")
    print(f"Normal transactions: {len(balanced_df[balanced_df['Class'] == 0])} ({len(balanced_df[balanced_df['Class'] == 0])/len(balanced_df):.2%})")
    
    return balanced_df


def evaluate_classical_models(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """
    Evaluate classical ML models for comparison
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary with performance metrics for each model
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    
    # Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"{name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        results[name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm,
            "predictions": y_pred,
            "probabilities": y_pred_proba
        }
    
    return results


def run_quantum_fraud_detection(data_path: str = 'creditcard.csv') -> None:
    """
    Main function to run the quantum fraud detection pipeline
    
    Args:
        data_path: Path to the credit card transaction data
    """
    print("=== YAQSHA: Quantum Financial Fraud Detector ===")
    print("Starting fraud detection pipeline...\n")
    
    # Load and sample data
    df = load_and_sample_credit_card_data(data_path)
    
    # Initialize data processor
    processor = DataProcessor()
    
    # Preprocess data
    X, y = processor.preprocess_data(df)
    
    # For quantum processing, we need to reduce dimensions
    # Option 1: PCA
    X_reduced = processor.dimensionality_reduction(X, n_components=4)
    
    # Option 2: Feature selection
    # X_reduced = processor.feature_importance_selection(X, y, n_features=4)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = processor.train_test_validation_split(X_reduced, y)
    
    print("\n--- Classical Model Benchmarks ---")
    classical_results = evaluate_classical_models(X_train, y_train, X_test, y_test)
    
    print("\n--- Quantum Model Training ---")
    # Initialize quantum model
    qml_model = YaqshaQuantumFraudDetector(n_qubits=4, n_layers=2)
    
    # Train the model
    loss_history = qml_model.fit(X_train, y_train, n_epochs=50, batch_size=16, learning_rate=0.01)
    
    # Plot training loss
    qml_model.plot_training_loss(loss_history)
    
    # Evaluate the model
    print("\n--- Quantum Model Evaluation ---")
    quantum_results = qml_model.evaluate(X_test, y_test)
    
    print("Quantum Model Results:")
    print(f"Accuracy: {quantum_results['accuracy']:.4f}")
    print(f"Precision: {quantum_results['precision']:.4f}")
    print(f"Recall: {quantum_results['recall']:.4f}")
    print(f"F1 Score: {quantum_results['f1_score']:.4f}")
    
    # Plot confusion matrix
    qml_model.plot_confusion_matrix(quantum_results['confusion_matrix'])
    
    print("\n--- Model Comparison ---")
    models = ["Quantum"] + list(classical_results.keys())
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    
    # Add quantum results to all results
    all_results = {"Quantum": quantum_results, **classical_results}
    
    # Print comparison table
    print("{:<20} {:<10} {:<10} {:<10} {:<10}".format("Model", "Accuracy", "Precision", "Recall", "F1 Score"))
    print("-" * 65)
    
    for model in models:
        print("{:<20} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
            model,
            all_results[model]["accuracy"],
            all_results[model]["precision"],
            all_results[model]["recall"],
            all_results[model]["f1_score"]
        ))


if __name__ == "__main__":
    # Run the fraud detection pipeline
    run_quantum_fraud_detection()
