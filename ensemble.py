import numpy as np
from knn import knn_impute_by_user
from item_response import irt, evaluate
from matrix_factorization import als
from utils import load_train_sparse, load_valid_csv, load_public_test_csv, load_train_csv, sparse_matrix_evaluate

def ensemble_predict(train_matrix, val_data, test_data, k, lr, iterations, als_k, als_eta, als_num_iterations):
    # KNN User-based prediction
    knn_matrix = knn_impute_by_user(train_matrix, val_data, k)

    # IRT prediction
    train_data = load_train_csv("./data")
    val_data = load_valid_csv("./data")
    theta, beta, _, _, _ = irt(train_data, val_data, lr, iterations)
    irt_val_pred = evaluate(val_data, theta, beta)
    irt_test_pred = evaluate(test_data, theta, beta)

    # Matrix Factorization prediction
    als_matrix, _, _ = als(train_data, als_k, als_eta, als_num_iterations, val_data)

    # Averaging predictions
    val_predictions = (knn_matrix + irt_val_pred + als_matrix) / 3
    test_predictions = (knn_matrix + irt_test_pred + als_matrix) / 3

    val_accuracy = sparse_matrix_evaluate(val_data, val_predictions)
    test_accuracy = sparse_matrix_evaluate(test_data, test_predictions)

    return val_accuracy, test_accuracy

def main():
    train_matrix = load_train_sparse("./data").toarray()
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    k = 11  # Optimal K for KNN User-based
    lr = 0.01  # Learning rate for IRT
    iterations = 50  # Number of iterations for IRT
    als_k = 50  # Optimal K for ALS
    als_eta = 0.01  # Learning rate for ALS
    als_num_iterations = 100  # Number of iterations for ALS

    val_accuracy, test_accuracy = ensemble_predict(train_matrix, val_data, test_data, k, lr, iterations, als_k, als_eta, als_num_iterations)

    print(f"Ensemble Validation Accuracy: {val_accuracy}")
    print(f"Ensemble Test Accuracy: {test_accuracy}")

if __name__ == "__main__":
    main()
