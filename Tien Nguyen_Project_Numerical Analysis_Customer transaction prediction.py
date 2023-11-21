import pandas as pd
import numpy as np
import matplotlib.pyplot as plt   # The default processing plot platform for python
import statsmodels.api as sm      # Scipy or sklearn or statmodels.api
import statsmodels.formula.api as smf # A way to do forward and backward selections
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from scipy.stats import chisquare
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
from concurrent.futures import ThreadPoolExecutor

#==============================================================================================
class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000, normalize=False):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.mean = None
        self.std = None
        self.normalize = normalize
        self.tol = 1e-3

    def _normalize(self, X):
        if self.mean is None or self.std is None:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
        return (X - self.mean) / (self.std + 1e-10)  # added epsilon to avoid division by zero

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # Backtracking for alpha using Armijo
    def line_search_armijo(self, X, y, dw, db, current_loss):
        alpha = 1.0  # Start with a step size of 1
        beta = 0.5  # Reduction factor for step size
        c = 1e-4  # Armijo condition constant

        # Calculate the initial gradient dot product
        grad_dot_p = np.dot(dw.T, dw) + db**2

        while True:
            # Propose new weights and bias
            new_weights = self.weights - alpha * dw
            new_bias = self.bias - alpha * db

            # Compute the new predictions and loss
            new_predictions = self.sigmoid(np.dot(X, new_weights) + new_bias)
            new_loss = -np.mean(y * np.log(new_predictions) + (1 - y) * np.log(1 - new_predictions))

            # Check the Armijo condition
            if new_loss <= current_loss - c * alpha * grad_dot_p:
                break  # The condition is satisfied

            alpha *= beta  # Reduce the step size

            if alpha < 1e-8:  # Prevent the step size from becoming too small
                break

        return alpha

    def fit(self, X, y, learning_rate):

        if self.normalize:
            X = self._normalize(X)

        # Get number of samples (rows) and number of features (columns) from matrix X
        num_samples, num_features = X.shape

        # Set weights and bias to 0
        self.weights = np.zeros(num_features)
        self.bias = 0

        loss_history = []  # List to store the loss at each epoch
        previous_loss = float('inf')

        # Gradient descent
        # Loop through all training epochs
        for epoch in range(self.epochs):
            print(f"\rEpoch {epoch + 1}- ", end="", flush=True)
            epoch_loss = 0  # Variable to store loss for each epoch
            linear_model = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_model)

            # Compute binary cross-entropy loss
            batch_loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
            epoch_loss += batch_loss * (len(X) / num_samples)  # Weighted average for the epoch

            # Compute gradients
            dw = (1 / num_samples) * np.dot(X.T, (predictions - y))
            db = (1 / num_samples) * np.sum(predictions - y)

            # Update parameters
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

            self.evaluate(X, y)
            loss_history.append(epoch_loss)  # Append the average loss for this epoch to the history

        return loss_history, list(range(1, self.epochs + 1))

    def fit_rmsprop(self, X, y, batch_size=30, beta=0.9, epsilon=1e-8, regularization=None, learning_rate=0.1):
        if self.normalize:
            X = self._normalize(X)

        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        s_w = np.zeros_like(self.weights)
        s_b = 0

        loss_history = []  # List to store the loss at each epoch
        previous_loss = float('inf')

        # RMSprop optimization
        for epoch in range(self.epochs):
            print(f"\rEpoch {epoch + 1}- ", end="", flush=True)
            epoch_loss = 0  # Variable to store loss for each epoch
            indices = np.random.permutation(num_samples)

            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, num_samples, batch_size):
                xi = X_shuffled[i:i + batch_size]
                yi = y_shuffled[i:i + batch_size]

                linear_model = np.dot(xi, self.weights) + self.bias
                predictions = self.sigmoid(linear_model)

                # Compute binary cross-entropy loss
                batch_loss = -np.mean(yi * np.log(predictions + epsilon) + (1 - yi) * np.log(1 - predictions + epsilon))
                epoch_loss += batch_loss * (len(xi) / num_samples)  # Weighted average for the epoch

                # Compute gradients
                dw = np.dot(xi.T, (predictions - yi)) / batch_size
                db = np.sum(predictions - yi) / batch_size

                # Update second moment estimate for weights
                s_w = beta * s_w + (1 - beta) * dw ** 2

                # Update second moment estimate for bias
                s_b = beta * s_b + (1 - beta) * db ** 2

                # Update parameters
                self.weights -= learning_rate * dw / (np.sqrt(s_w) + epsilon)
                self.bias -= learning_rate * db / (np.sqrt(s_b) + epsilon)

            # ... (evaluation and potential early stopping)

            self.evaluate(X, y)
            loss_history.append(epoch_loss)  # Append the average loss for this epoch to the history

        return loss_history, list(range(1, self.epochs + 1))

    def fit_adagrad(self, X, y, batch_size=30, epsilon=1e-8, regularization=None, learning_rate=0.1):

        if self.normalize:
            X = self._normalize(X)

        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        r_w = np.zeros_like(self.weights)
        r_b = 0

        loss_history = []  # List to store the loss at each epoch
        previous_loss = float('inf')

        # AdaGrad optimization
        for epoch in range(self.epochs):
            print(f"\rEpoch {epoch + 1}- ", end="", flush=True)
            epoch_loss = 0  # Variable to store loss for each epoch
            indices = np.random.permutation(num_samples)

            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, num_samples, batch_size):
                xi = X_shuffled[i:i + batch_size]
                yi = y_shuffled[i:i + batch_size]

                linear_model = np.dot(xi, self.weights) + self.bias
                predictions = self.sigmoid(linear_model)

                # Compute binary cross-entropy loss
                batch_loss = -np.mean(yi * np.log(predictions + epsilon) + (1 - yi) * np.log(1 - predictions + epsilon))
                epoch_loss += batch_loss * (len(xi) / num_samples)  # Weighted average for the epoch

                # Compute gradients
                dw = np.dot(xi.T, (predictions - yi)) / batch_size
                db = np.sum(predictions - yi) / batch_size

                # Accumulate squared gradient for weights and bias
                r_w += dw ** 2
                r_b += db ** 2

                # Update parameters
                self.weights -= learning_rate * dw / (np.sqrt(r_w) + epsilon)
                self.bias -= learning_rate * db / (np.sqrt(r_b) + epsilon)

            # ... (evaluation and potential early stopping)

            self.evaluate(X, y)
            loss_history.append(epoch_loss)  # Append the average loss for this epoch to the history

        return loss_history, list(range(1, self.epochs + 1))

    def fit_adam(self, X, y, batch_size=30, beta1=0.9, beta2=0.999, epsilon=1e-8, regularization=None, learning_rate=0.1):

        if self.normalize:
            X = self._normalize(X)

        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        m_w, v_w = np.zeros_like(self.weights), np.zeros_like(self.weights)
        m_b, v_b = 0, 0
        t = 0

        loss_history = []  # List to store the loss at each epoch
        previous_loss = float('inf')

        # Adam optimization
        for epoch in range(self.epochs):
            print(f"\rEpoch {epoch + 1}- ", end="", flush=True)
            epoch_loss = 0  # Variable to store loss for each epoch
            indices = np.random.permutation(num_samples)

            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, num_samples, batch_size):
                xi = X_shuffled[i:i + batch_size]
                yi = y_shuffled[i:i + batch_size]

                linear_model = np.dot(xi, self.weights) + self.bias
                predictions = self.sigmoid(linear_model)

                # Compute binary cross-entropy loss
                batch_loss = -np.mean(yi * np.log(predictions + epsilon) + (1 - yi) * np.log(1 - predictions + epsilon))
                epoch_loss += batch_loss * (len(xi) / num_samples)  # Weighted average for the epoch

                # Compute gradients
                dw = np.dot(xi.T, (predictions - yi)) / batch_size
                db = np.sum(predictions - yi) / batch_size


                # Regularization
                if regularization == 'l2':
                    dw += learning_rate * self.weights  # Adding L2 regularization term
                    db += learning_rate * self.bias  # Adding L2 regularization term
                elif regularization == 'l1':
                    dw += learning_rate * np.sign(self.weights)  # Adding L1 regularization term
                    db += learning_rate * np.sign(self.bias)  # Adding L1 regularization term
                elif regularization == 'both':
                    dw += learning_rate * np.sign(self.weights)  # Adding L1 regularization term
                    db += learning_rate * np.sign(self.bias)  # Adding L1 regularization term

                    dw += learning_rate * self.weights  # Adding L2 regularization term
                    db += learning_rate * self.bias  # Adding L2 regularization term

                # Update biased first and second moment estimates for weights
                m_w = beta1 * m_w + (1 - beta1) * dw
                v_w = beta2 * v_w + (1 - beta2) * dw ** 2

                # Update biased first and second moment estimates for bias
                m_b = beta1 * m_b + (1 - beta1) * db
                v_b = beta2 * v_b + (1 - beta2) * db ** 2

                # Compute bias-corrected moment estimates
                m_w_hat = m_w / (1 - beta1 ** (t + 1))
                v_w_hat = v_w / (1 - beta2 ** (t + 1))

                m_b_hat = m_b / (1 - beta1 ** (t + 1))
                v_b_hat = v_b / (1 - beta2 ** (t + 1))

                # Update parameters
                self.weights -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
                self.bias -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

                t += 1

            y_predict =  self.evaluate(X, y)
            loss_history.append(epoch_loss)  # Append the average loss for this epoch to the history

        return loss_history, list(range(1, self.epochs + 1))

    # Label prediction
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        class_predictions = [1 if i > 0.5 else 0 for i in y_pred]
        return class_predictions

    # Probability prediction
    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        return y_pred

    # Evaluation for training or testing dataset (requires knowing y_test)
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)  # Labels
        # Calculate accuracy by commparing predicted y and tested y, which creates a boolean of 1 and 0
        # The mean is the percentage of True
        accuracy = np.mean(y_pred == y_test)
        print(f'Model Accuracy: {accuracy * 100:.2f}%')

        return y_pred

    def plot_roc_curve_train(self, X, y_true):
        """
        Plot ROC curve.

        X: The input features
        y_true: True labels
        """

        X = self._normalize(X)

        # Assuming predict_proba is a method in your class that predicts the probability
        # of the positive class
        y_score = self.predict_proba(X)

        thresholds = np.linspace(1, 0, 100)
        tpr = []  # True Positive Rate
        fpr = []  # False Positive Rate

        for thresh in thresholds:
            y_pred = (y_score > thresh).astype(int)
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            tn = np.sum((y_pred == 0) & (y_true == 0))

            tpr.append(tp / (tp + fn))
            fpr.append(fp / (fp + tn))


        # # Accuracy
        # accuracy = (tp + tn) / (fp + fn)
        # print(f'Accuracy: {accuracy:.2f}')
        # # Precision
        # precision = tp / (tp + fp)
        # print(f'Precision: {precision:.2f}')
        # # Recall
        # recall = tp / (tp + fn)
        # print(f'Recall: {recall:.2f}')
        # # F1
        # f1 = 2 * (precision * recall) / (precision + recall)
        # print(f'F1: {f1:.2f}')

        # Calculating AUC using the trapezoidal rule
        auc = np.trapz(tpr, x=fpr)

        # Plotting the ROC curve
        plt.plot(fpr, tpr, label=f'ROC Curve for TRAIN set (AUC = {auc:.5f})')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.grid(True)

    def plot_roc_curve_test(self, X, y_true):
        """
        Plot ROC curve.

        X: The input features
        y_true: True labels
        """

        X = self._normalize(X)

        # Assuming predict_proba is a method in your class that predicts the probability
        # of the positive class
        y_score = self.predict_proba(X)

        thresholds = np.linspace(1, 0, 100)
        tpr = []  # True Positive Rate
        fpr = []  # False Positive Rate

        for thresh in thresholds:
            y_pred = (y_score > thresh).astype(int)
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            tn = np.sum((y_pred == 0) & (y_true == 0))

            tpr.append(tp / (tp + fn))
            fpr.append(fp / (fp + tn))

        # # Accuracy
        # accuracy = (tp + tn) / (fp + fn)
        # print(f'Accuracy: {accuracy:.2f}')
        # # Precision
        # precision = tp / (tp + fp)
        # print(f'Precision: {precision:.2f}')
        # # Recall
        # recall = tp / (tp + fn)
        # print(f'Recall: {recall:.2f}')
        # # F1
        # f1 = 2 * (precision * recall) / (precision + recall)
        # print(f'F1: {f1:.2f}')

        # Calculating AUC using the trapezoidal rule
        auc = np.trapz(tpr, x=fpr)

        # Plotting the ROC curve
        plt.plot(fpr, tpr, label=f'ROC Curve for TEST set (AUC = {auc:.5f})')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.grid(True)

        return fpr, tpr, auc

    def confusion_matrix(self, y_predict, y_test):
        print("Confusion Matrix:")

        y_predict = [1 if i == True else 0 for i in y_predict]
        y_test = [1 if i == True else 0 for i in y_test]

        print(len(y_predict))
        print(len(y_test))

def plot_loss_vs_epochs(loss_epoch_pairs, legends):
    plt.figure(figsize=(10, 5))  # Set the figure size

    # Iterate over the list of loss-epoch pairs and legends
    for (loss_history, epochs), legend in zip(loss_epoch_pairs, legends):
        plt.plot(epochs, loss_history, label=legend)

    # plt.title('Training Loss vs. Epochs')  # Set the title of the graph
    plt.xlabel('Epochs')  # Set the x-axis label
    plt.ylabel('Loss')  # Set the y-axis label
    plt.legend()  # Show legend
    plt.grid(True)  # Show grid
    plt.tight_layout()  # Adjust the layout

def random_sparsity_df(data, sparsity_level):
    # data should be a pandas DataFrame
    total_entries = data.size
    total_sparsity = int(total_entries * sparsity_level)
    rows, cols = data.shape
    for _ in range(total_sparsity):
        i = np.random.randint(0, rows)
        j = np.random.randint(0, cols)
        data.iat[i, j] = 0  # or np.nan if you prefer
    return data

def feature_wise_sparsity_df(data, sparsity_level, features=range(200)):
    num_rows = data.shape[0]
    num_values_to_zero = int(sparsity_level * num_rows)

    for feature in features:
        zero_indices = np.random.choice(num_rows, num_values_to_zero, replace=False)
        data.iloc[zero_indices, feature] = 0  # or np.nan if you prefer

    return data

def row_wise_sparsity_df(data, sparsity_level):
    num_rows, num_cols = data.shape
    num_values_to_zero = int(sparsity_level * num_cols)

    # Create a copy to avoid modifying the original DataFrame
    sparse_data = data.copy()

    for index in range(num_rows):
        zero_indices = np.random.choice(num_cols, num_values_to_zero, replace=False)
        sparse_data.iloc[index, zero_indices] = 0  # or np.nan if you prefer

    return sparse_data

# Define a function to replace outliers with NaN
def replace_outliers_with_nan(data, m=3):
    mean = np.mean(data)
    std = np.std(data)

    # Identify outliers
    outliers = (data < (mean - m * std)) | (data > (mean + m * std))

    # Replace outliers with NaN
    data[outliers] = np.nan
    return data

def plot_combined_roc_curve(model_fpr_tpr_auc_list):
    """
    Plots a combined ROC curve for multiple models.

    model_fpr_tpr_auc_list: List of tuples, each containing the FPR, TPR, and AUC for a model.
                            Each tuple is expected to be in the format (fpr, tpr, auc, model_name).
    """
    plt.figure(figsize=(10, 8))

    for fpr, tpr, auc, model_name in model_fpr_tpr_auc_list:
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.5f})')

    plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend()
    plt.grid(True)
    # plt.show()


#==============================================================================================
### Datasource: https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data?select=test.csv

### Data preparation
# Read the CSV file into a DataFrame
Data_original = pd.read_csv('train.csv')

## Randomly pick datapoints to reduce the size of datasets
# Check if the dataset has at least 5000 rows
if len(Data_original) >= 5000:
    # Randomly sample 20,000 rows from the DataFrame
    Data = Data_original.sample(n=5000, random_state=42)
else:
    print("The dataset contains fewer than 20,000 rows.")

Data = Data.drop(columns=['ID_code'])


# Add an intercept column
Data['intercept'] = np.ones(Data.shape[0])

# Prepare the data
X = Data.drop(columns=['target'])
X = X.fillna(method='ffill')
Y = Data['target']

#==============================================================================================
### MODELS
#   Train/Test split
X_train_sparse_00, X_test_sparse_00, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

X_train_sparse_00 = np.asarray(X_train_sparse_00)
X_test_sparse_00 = np.asarray(X_test_sparse_00)
Y_train = np.asarray(Y_train)
Y_test = np.asarray(Y_test)

#   Training models
epochs = 50

model_0_sparse_00 = LogisticRegression(epochs=epochs)
model_1_sparse_00 = LogisticRegression(epochs=epochs)
model_3_sparse_00 = LogisticRegression(epochs=epochs)
model_4_sparse_00 = LogisticRegression(epochs=epochs)

model_0_sparse_00_sensitivity = LogisticRegression(epochs=epochs)
model_1_sparse_00_sensitivity = LogisticRegression(epochs=epochs)
model_3_sparse_00_sensitivity = LogisticRegression(epochs=epochs)
model_4_sparse_00_sensitivity = LogisticRegression(epochs=epochs)

#==============================================================================================
## Hyperparameter sensitivity analysis
# GD
alpha_fit = []
legends_fit = []
for learning_rate in [0.002, 0.001, 0.0005, 0.0001]:
    a, b = model_0_sparse_00_sensitivity.fit(X_train_sparse_00, Y_train, learning_rate=learning_rate)
    alpha_fit.append((a, b))
    legends_fit.append(learning_rate)

while True:
    plot_loss_vs_epochs(alpha_fit, legends_fit)
    plt.title('Alpha sensitivity for gradient descent')
    # plt.ylim([0.2, 0.29])
    # plt.show()  # Display the plot
    break

# Adagrad
alpha_adagrad = []
legends_adagrad = []
for learning_rate in [0.005, 0.001, 0.0005, 0.0001, 0.00005]:
    a, b = model_3_sparse_00_sensitivity.fit_adagrad(X_train_sparse_00, Y_train, regularization=None, learning_rate=learning_rate, batch_size=5000)
    alpha_adagrad.append((a, b))
    legends_adagrad.append(learning_rate)

while True:
    plot_loss_vs_epochs(alpha_adagrad, legends_adagrad)
    plt.title('Alpha sensitivity for Adagrad')
    # plt.show()  # Display the plot
    break

# RMSProp
alpha_rmsprop = []
legends_rmsprop = []
for learning_rate in [0.002, 0.001, 0.0005, 0.0001, 0.00005]:
    a, b = model_4_sparse_00_sensitivity.fit_rmsprop(X_train_sparse_00, Y_train, regularization=None, learning_rate=learning_rate, batch_size=5000)
    alpha_rmsprop.append((a, b))
    legends_rmsprop.append(learning_rate)

while True:
    plot_loss_vs_epochs(alpha_rmsprop, legends_rmsprop)
    plt.title('Alpha sensitivity for RMSProp')
    # plt.show()  # Display the plot
    break

# Adam
alpha_adam = []
legends_adam = []
for learning_rate in [0.002, 0.001, 0.0005, 0.0001, 0.00005]:
    a, b = model_1_sparse_00_sensitivity.fit_adam(X_train_sparse_00, Y_train, regularization=None, learning_rate=learning_rate, batch_size=5000)
    alpha_adam.append((a, b))
    legends_adam.append(learning_rate)

while True:
    plot_loss_vs_epochs(alpha_adam, legends_adam)
    plt.title('Alpha sensitivity for Adam')
    # plt.show()  # Display the plot
    break

plt.show()
#==============================================================================================
### Traing loss and ROC
while True:
    # print("This is fit model")
    a0, b0 = model_0_sparse_00.fit(X_train_sparse_00, Y_train, learning_rate=0.0005)
    # print("This is adagrad model")
    a3, b3 = model_3_sparse_00.fit_adagrad(X_train_sparse_00, Y_train, regularization=None, learning_rate=0.0005, batch_size=5000)
    # print("This is RMSProp model")
    a4, b4 = model_4_sparse_00.fit_rmsprop(X_train_sparse_00, Y_train, regularization=None, learning_rate=0.0001, batch_size=5000)
    # print("This is adam model")
    a1, b1 = model_1_sparse_00.fit_adam(X_train_sparse_00, Y_train, regularization=None, learning_rate=0.0001, batch_size=5000)

    #   Plot loss vs epoch
    legends = ['Gradient Descent', 'Adam', 'Adagrad', 'RMSProp']
    plot_loss_vs_epochs([(a0, b0), (a1, b1), (a3, b3), (a4, b4)], legends)
    # plt.ylim([0.2, 0.7])
    plt.title(f'Training Loss vs. Epochs')  # Set the title of the graph
    plt.show()  # Display the plot

    #==============================================================================================
    # Combined ROC
    model_0_fpr_sparse_00, model_0_tpr_sparse_00, model_0_auc_sparse_00 = model_0_sparse_00.plot_roc_curve_test(X_test_sparse_00,Y_test)
    model_1_fpr_sparse_00, model_1_tpr_sparse_00, model_1_auc_sparse_00 = model_1_sparse_00.plot_roc_curve_test(X_test_sparse_00,Y_test)
    model_3_fpr_sparse_00, model_3_tpr_sparse_00, model_3_auc_sparse_00 = model_3_sparse_00.plot_roc_curve_test(X_test_sparse_00,Y_test)
    model_4_fpr_sparse_00, model_4_tpr_sparse_00, model_4_auc_sparse_00 = model_4_sparse_00.plot_roc_curve_test(X_test_sparse_00,Y_test)

    model_data_sparse_00 = [
        (model_0_fpr_sparse_00, model_0_tpr_sparse_00, model_0_auc_sparse_00, 'Gradient Descent'),
        (model_1_fpr_sparse_00, model_1_tpr_sparse_00, model_1_auc_sparse_00, 'Adam'),
        (model_3_fpr_sparse_00, model_3_tpr_sparse_00, model_3_auc_sparse_00, 'Adagrad'),
        (model_4_fpr_sparse_00, model_4_tpr_sparse_00, model_4_auc_sparse_00, 'RMSProp')
    ]

    plot_combined_roc_curve(model_data_sparse_00)
    plt.title(f'ROC Curves of test sets')
    plt.show()

    break

