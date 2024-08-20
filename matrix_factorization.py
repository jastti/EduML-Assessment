import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)


def svd_reconstruct(matrix, k):
    new_matrix = matrix.copy()
    mask = np.isnan(new_matrix)
    masked_matrix = np.ma.masked_array(new_matrix, mask)
    item_means = np.mean(masked_matrix, axis=0)
    new_matrix = masked_matrix.filled(item_means)

    item_means = np.mean(new_matrix, axis=0)
    mu = np.tile(item_means, (new_matrix.shape[0], 1))
    new_matrix = new_matrix - mu

    Q, s, Ut = np.linalg.svd(new_matrix, full_matrices=False)
    s = np.diag(s)

    s = s[0:k, 0:k]
    Q = Q[:, 0:k]
    Ut = Ut[0:k, :]
    s_root = sqrtm(s)

    reconst_matrix = np.dot(np.dot(Q, s_root), np.dot(s_root, Ut))
    reconst_matrix = reconst_matrix + mu
    return np.array(reconst_matrix)


def squared_error_loss(data, u, z):
    loss = 0
    for i, q in enumerate(data["question_id"]):
        loss += (data["is_correct"][i] - np.sum(u[data["user_id"][i]] * z[q])) ** 2.0
    return 0.5 * loss


def update_u_z(train_data, lr, u, z):
    i = np.random.choice(len(train_data["question_id"]), 1)[0]

    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]

    error = c - np.dot(u[n], z[q])

    u[n] += lr * error * z[q]
    z[q] += lr * error * u[n]

    return u, z


def als(train_data, k, lr, num_iteration, val_data=None):
    u = np.random.uniform(
        low=0, high=1 / np.sqrt(k), size=(len(set(train_data["user_id"])), k)
    )
    z = np.random.uniform(
        low=0, high=1 / np.sqrt(k), size=(len(set(train_data["question_id"])), k)
    )

    train_losses = []
    val_losses = []

    for i in range(num_iteration):
        u, z = update_u_z(train_data, lr, u, z)
        train_loss = squared_error_loss(train_data, u, z)
        train_losses.append(train_loss)

        if val_data is not None:
            val_loss = squared_error_loss(val_data, u, z)
            val_losses.append(val_loss)

    return np.dot(u, z.T), train_losses, val_losses


def plot_losses(train_losses, val_losses):
    iterations = range(len(train_losses))
    plt.figure()
    plt.plot(iterations, train_losses, label='Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Training Squared Error Loss')
    plt.title('Training Loss over Iterations')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(iterations, val_losses, label='Validation Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Validation Squared Error Loss')
    plt.title('Validation Loss over Iterations')
    plt.legend()
    plt.show()


def main():
    train_matrix = load_train_sparse("./data").toarray()
    train_data = load_train_csv("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    k_values = [1, 10, 20, 50, 100]
    best_k = k_values[0]
    best_val_mse = float('inf')

    for k in k_values:
        reconstructed_matrix = svd_reconstruct(train_matrix, k)
        val_mse = sparse_matrix_evaluate(val_data, reconstructed_matrix)
        print(f"SVD with k={k}, Validation MSE: {val_mse}")
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_k = k

    print(f"Best k for SVD: {best_k}, Best Validation MSE: {best_val_mse}")

    final_reconstructed_matrix = svd_reconstruct(train_matrix, best_k)
    test_mse = sparse_matrix_evaluate(test_data, final_reconstructed_matrix)
    print(f"SVD Test MSE with best k={best_k}: {test_mse}")

    k_values = [1, 10, 20, 50, 100]
    learning_rates = [0.001, 0.01, 0.1]
    num_iterations_values = [50, 100, 200]

    best_k = k_values[0]
    best_eta = learning_rates[0]
    best_num_iterations = num_iterations_values[0]
    best_val_loss = float('inf')

    for eta in learning_rates:
        for num_iterations in num_iterations_values:
            for k in k_values:
                reconstructed_matrix, train_losses, val_losses = als(train_data, k, eta, num_iterations, val_data)

                u, s, vt = np.linalg.svd(reconstructed_matrix, full_matrices=False)
                z = vt.T

                val_loss = squared_error_loss(val_data, u, z)
                print(f"ALS with k={k}, eta={eta}, num_iterations={num_iterations}, Validation Loss: {val_loss}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_k = k
                    best_eta = eta
                    best_num_iterations = num_iterations

    print(f"Best k for ALS: {best_k}, Best Validation Loss: {best_val_loss}")

    mat, train_losses, val_losses = als(train_data, best_k, best_eta, best_num_iterations, val_data)

    plot_losses(train_losses, val_losses)

    u, s, vt = np.linalg.svd(mat, full_matrices=False)
    z = vt.T
    test_als_loss = squared_error_loss(test_data, u, z)
    print(f"ALS Test Loss with best k={best_k}, best eta={best_eta}, best num_iterations={best_num_iterations}: {test_als_loss}")

    # Calculate and report final validation and test accuracy
    final_val_accuracy = sparse_matrix_evaluate(val_data, mat)
    final_test_accuracy = sparse_matrix_evaluate(test_data, mat)

    print(f"Final Validation Accuracy for ALS: {final_val_accuracy}")
    print(f"Final Test Accuracy for ALS: {final_test_accuracy}")


if __name__ == "__main__":
    main()
