from matplotlib import pyplot as plt

from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)
import numpy as np


def sigmoid(x):
    """Apply sigmoid function."""
    # x = np.clip(x, -500, 500)
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################         #
    #####################################################################
    log_lklihood = 0.0

    len_uid = len(data['user_id'])
    for i in range(len_uid):
        uid = data['user_id'][i]
        qid = data['question_id'][i]
        c = data['is_correct'][i]
        z = theta[uid] - beta[qid]  # theta_i - beta_j
        log_lklihood += c * z - np.log(1 + np.exp(z))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    #####################################################################
    theta_grad = np.zeros_like(theta)
    beta_grad = np.zeros_like(beta)

    len_uid = len(data['user_id'])
    for i in range(len_uid):
        uid = data['user_id'][i]
        qid = data['question_id'][i]
        c = data['is_correct'][i]
        z = theta[uid] - beta[qid]  # theta_i - beta_j
        p = sigmoid(z)  # p_ij = sigmoid(theta_i - beta_j)
        theta_grad[uid] += c - p
        beta_grad[qid] += p - c

    # update for theta
    theta += lr * theta_grad
    # update for beta
    beta += lr * beta_grad
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    num_u = max(data['user_id']) + 1  # prevent out of bounds errors
    num_q = max(data['question_id']) + 1

    theta = np.zeros(num_u)
    beta = np.zeros(num_q)

    val_acc = []
    train_neg_lld = []
    val_neg_lld = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        val_lld = neg_log_likelihood(val_data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)

        train_neg_lld.append(neg_lld)
        val_neg_lld.append(val_lld)
        val_acc.append(score)

        # print("NLLK: {} \t Score: {}".format(neg_lld, score))
        print(f"Iteration {i + 1} - NLLK: {neg_lld}, Validation NLLK: {val_lld}, Score: {score}")
        theta, beta = update_theta_beta(data, lr, theta, beta)

    return theta, beta, train_neg_lld, val_neg_lld, val_acc


def evaluate(data, theta, beta):
    """Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])

def plot_probability_curves(theta, beta, questions):
    """Plot probability curves for selected questions."""
    theta_range = np.linspace(-3, 3, 100)
    plt.figure(figsize=(10, 6))

    for j in questions:
        prob = sigmoid(theta_range - beta[j])
        plt.plot(theta_range, prob, label=f"Question {j}")

    plt.xlabel("Theta")
    plt.ylabel("P(Correct Answer)")
    plt.title("Probability of Correct Response as a Function of Theta")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    train_data = load_train_csv("./data")
    # You may optionally use the sparse matrix.
    # sparse_matrix = load_train_sparse("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    #####################################################################
    learning_rate = 0.01
    iterations = 50

    # train irt model
    # theta, beta, val_acc_lst = irt(train_data, val_data, learning_rate, iterations)

    theta, beta, val_acc_lst, train_neg_lld_lst, val_neg_lld_lst = irt(train_data, val_data, learning_rate, iterations)

    # log-likelihoods for each iteration
    # train_neg_lld_lst = []
    # val_neg_lld_lst = []
    # for i in range(iterations):
    #     train_neg_lld = neg_log_likelihood(train_data, theta, beta)
    #     val_neg_lld = neg_log_likelihood(val_data, theta, beta)
    #     train_neg_lld_lst.append(train_neg_lld)
    #     val_neg_lld_lst.append(val_neg_lld)

    # Plot training and validation log-likelihoods
    # plt.plot(range(iterations), [-x for x in train_neg_lld_lst], label='Training Log-Likelihood')
    # plt.plot(range(iterations), [-x for x in val_neg_lld_lst], label='Validation Log-Likelihood')

    plt.plot(range(iterations), [-x for x in train_neg_lld_lst], label='Training Log-Likelihood')
    plt.plot(range(iterations), [-x for x in val_neg_lld_lst], label='Validation Log-Likelihood')
    plt.xlabel('Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Log-Likelihood')
    plt.title('Training and Validation Log-Likelihoods')
    plt.legend()
    plt.show()

    # Evaluate on test data
    test_accuracy = evaluate(test_data, theta, beta)
    print(f"Final Test Accuracy: {test_accuracy}")

    pass

    selected_questions = [0, 1, 2]
    plot_probability_curves(theta, beta, selected_questions)


    # code, report the validation and test accuracy.                    #
    #####################################################################
    # pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    #
    #####################################################################
    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
