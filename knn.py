import numpy as np
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)


def knn_impute_by_user(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    matrix = matrix.T
    mat = nbrs.fit_transform(matrix)
    mat = mat.T
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Item-based KNN Validation Accuracy (k={}): {}".format(k, acc))
    return acc


def main():
    sparse_matrix = load_train_sparse("./data").toarray()
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    k_values = [1, 6, 11, 16, 21, 26]
    user_validation_accuracies = []
    item_validation_accuracies = []
    user_test_accuracies = []
    item_test_accuracies = []

    for k in k_values:
        user_acc = knn_impute_by_user(sparse_matrix, val_data, k)
        item_acc = knn_impute_by_item(sparse_matrix, val_data, k)
        user_validation_accuracies.append(user_acc)
        item_validation_accuracies.append(item_acc)

    for k in k_values:
        user_test_acc = knn_impute_by_user(sparse_matrix, test_data, k)
        item_test_acc = knn_impute_by_item(sparse_matrix, test_data, k)
        user_test_accuracies.append(user_test_acc)
        item_test_accuracies.append(item_test_acc)

    best_k_user = k_values[np.argmax(user_validation_accuracies)]
    best_k_item = k_values[np.argmax(item_validation_accuracies)]

    print("Best k for user-based KNN:", best_k_user)
    print("Best k for item-based KNN:", best_k_item)

    test_acc_user = knn_impute_by_user(sparse_matrix, test_data, best_k_user)
    test_acc_item = knn_impute_by_item(sparse_matrix, test_data, best_k_item)

    print("User-based KNN Test Accuracy with k={}: {}".format(best_k_user, test_acc_user))
    print("Item-based KNN Test Accuracy with k={}: {}".format(best_k_item, test_acc_item))

    plt.figure(figsize=(10, 5))
    plt.plot(k_values, user_validation_accuracies, label='User-based KNN')
    plt.plot(k_values, item_validation_accuracies, label='Item-based KNN')
    plt.xlabel('k')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy vs. k for KNN Imputation')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(k_values, user_test_accuracies, label='User-based KNN')
    plt.plot(k_values, item_test_accuracies, label='Item-based KNN')
    plt.xlabel('k')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy vs. k for Item-based KNN')
    plt.legend()
    plt.show()

    #####################################################################
    #####################################################################
    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
