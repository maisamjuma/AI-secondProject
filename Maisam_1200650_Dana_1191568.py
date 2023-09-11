import numpy as np
import csv
import sys
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import warnings

# Constants
TEST_SIZE = 0.3  # Proportion of the dataset used for testing
K = 3  # Number of nearest neighbors to consider in k-NN

class NN:
    def __init__(self, trainingFeatures, trainingLabels):
        self.trainingFeatures = trainingFeatures
        self.trainingLabels = trainingLabels

    def predict(self, features, k):
        predictions = []
        for test_sample in features:
            distances = []
            for train_sample in self.trainingFeatures:
                dist = np.linalg.norm(test_sample - train_sample)
                distances.append(dist)
            indices = np.argsort(distances)[:k]
            k_nearest_labels = self.trainingLabels[indices]
            pred_label = np.argmax(np.bincount(k_nearest_labels))
            predictions.append(pred_label)
        return predictions

def load_data(filename):
    # Read the dataset from a CSV file
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        data = []
        for row in reader:
            data.append([float(val) for val in row])
    data = np.array(data)
    features = data[:, :-1]
    labels = data[:, -1]
    labels = np.where(labels == 0, 0, 1)  # Map labels to 0 (non-spam) or 1 (spam)
    return features, labels

def preprocess(features):
    # Normalize the features by subtracting the mean and dividing by the standard deviation
    means = np.mean(features, axis=0)
    stds = np.std(features, axis=0)
    normalized_features = (features - means) / stds
    return normalized_features

def train_mlp_model(features, labels):
    # Train the MLP model with warnings suppressed
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=Warning)
        model = MLPClassifier(hidden_layer_sizes=(10, 5), activation='logistic', random_state=42, max_iter=500)
        model.fit(features, labels)
    return model

def evaluate(labels, predictions):
    # Evaluate the performance by computing accuracy, precision, recall, and F1 score
    accuracy = np.mean(labels == predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    return accuracy, precision, recall, f1

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python template.py ./spambase.csv")

    # Load the data
    features, labels = load_data(sys.argv[1])
    features = preprocess(features)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=TEST_SIZE)

    # k-Nearest Neighbors (k-NN)
    model_nn = NN(X_train, y_train)
    predictions = model_nn.predict(X_test, K)
    accuracy, precision, recall, f1 = evaluate(y_test, predictions)
    confusion_matrix_nn = confusion_matrix(y_test, predictions)

    # Print k-NN results
    print("************************************************")
    print(" k-Nearest Neighbors (k-NN) Results ")
    print("1. Accuracy: ", accuracy)
    print("2. Precision: ", precision)
    print("3. Recall: ", recall)
    print("4. F1: ", f1)
    print("5. Confusion Matrix:\n", confusion_matrix_nn)
    print("************************************************")

    # Multi-Layer Perceptron (MLP)
    model = train_mlp_model(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy, precision, recall, f1 = evaluate(y_test, predictions)
    confusion_matrix_mlp = confusion_matrix(y_test, predictions)

    # Print MLP results
    
    print("**** Multi-Layer Perceptron (MLP) Results ****")
    print("1. Accuracy: ", accuracy)
    print("2. Precision: ", precision)
    print("3. Recall: ", recall)
    print("4. F1: ", f1)
    print("5. Confusion Matrix:\n", confusion_matrix_mlp)
    print("************************************************")

if __name__ == "__main__":
    main()
