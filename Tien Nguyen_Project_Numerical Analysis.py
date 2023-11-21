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
### Data processing
# Read the CSV file into a DataFrame
Data = pd.read_csv('HW3-data.csv')
Data = Data.drop(columns=['ID'])
# Data = Data[['Diagnosis','RadiusM', 'TextureM', 'PerimeterM', 'AreaM', 'SmoothnessM', 'CompactnessM', 'ConcavityM', 'ConcavePointsM', 'SymmetryM', 'FdimensionM']]
# Data = Data[['Diagnosis','RadiusSE', 'TextureSE', 'PerimeterSE', 'AreaSE', 'SmoothnessSE', 'CompactnessSE', 'ConcavitySE', 'ConcavePointsSE', 'SymmetrySE', 'FdimensionSE']]
# Data = Data[['Diagnosis','RadiusW', 'TextureW', 'PerimeterW', 'AreaW', 'SmoothnessW', 'CompactnessW', 'ConcavityW', 'ConcavePointsW', 'SymmetryW', 'FdimensionW']]

# Remove rows with 0s
# Data = Data[(Data != 0).all(axis=1)]

# Add an intercept column
Data['intercept'] = np.ones(Data.shape[0])

# Prepare the data
X = Data.drop(columns=['Diagnosis'])
X = X.fillna(method='ffill')
Y = Data['Diagnosis']

X_fit = X

X = np.asarray(X)
Y = np.asarray(Y)
Y = pd.get_dummies(Y, drop_first=True).iloc[:,0] # Drop the baseline, already included in the intercept

#==============================================================================================
class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.mean = None
        self.std = None

    def _normalize(self, X):
        if self.mean is None or self.std is None:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
        return (X - self.mean) / (self.std + 1e-10)  # added epsilon to avoid division by zero

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y, regularization=None, lambda_=0.1):
        # Get number of samples (rows) and number of features (columns) from matrix X
        num_samples, num_features = X.shape

        # Set weights and bias to 0
        self.weights = np.zeros(num_features)
        self.bias = 0

        loss_history = []  # List to store the loss at each epoch

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

            # Regularization
            if regularization == 'l2':
                dw += lambda_ * self.weights  # Adding L2 regularization term
                db += lambda_ * self.bias  # Adding L2 regularization term
            elif regularization == 'l1':
                dw += lambda_ * np.sign(self.weights)  # Adding L1 regularization term
                db += lambda_ * np.sign(self.bias)  # Adding L1 regularization term

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            self.evaluate(X, y)
            loss_history.append(epoch_loss)  # Append the average loss for this epoch to the history

        return loss_history, list(range(1, self.epochs + 1))

    def fit_rmsprop(self, X, y, batch_size=30, beta=0.9, epsilon=1e-8, regularization=None, lambda_=0.1):
        # Normalization
        X = self._normalize(X)

        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        s_w = np.zeros_like(self.weights)
        s_b = 0

        loss_history = []  # List to store the loss at each epoch

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
                self.weights -= self.learning_rate * dw / (np.sqrt(s_w) + epsilon)
                self.bias -= self.learning_rate * db / (np.sqrt(s_b) + epsilon)

            # ... (evaluation and potential early stopping)

            self.evaluate(X, y)
            loss_history.append(epoch_loss)  # Append the average loss for this epoch to the history

        return loss_history, list(range(1, self.epochs + 1))

    def fit_adagrad(self, X, y, batch_size=30, epsilon=1e-8, regularization=None, lambda_=0.1):
        # Normalization
        X = self._normalize(X)

        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        r_w = np.zeros_like(self.weights)
        r_b = 0

        loss_history = []  # List to store the loss at each epoch

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
                self.weights -= self.learning_rate * dw / (np.sqrt(r_w) + epsilon)
                self.bias -= self.learning_rate * db / (np.sqrt(r_b) + epsilon)

            # ... (evaluation and potential early stopping)

            self.evaluate(X, y)
            loss_history.append(epoch_loss)  # Append the average loss for this epoch to the history

        return loss_history, list(range(1, self.epochs + 1))

    def fit_adam(self, X, y, batch_size=30, beta1=0.9, beta2=0.999, epsilon=1e-8, regularization=None, lambda_=0.1):

        # Normalization
        X = self._normalize(X)

        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        m_w, v_w = np.zeros_like(self.weights), np.zeros_like(self.weights)
        m_b, v_b = 0, 0
        t = 0

        loss_history = []  # List to store the loss at each epoch

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
                    dw += lambda_ * self.weights  # Adding L2 regularization term
                    db += lambda_ * self.bias  # Adding L2 regularization term
                elif regularization == 'l1':
                    dw += lambda_ * np.sign(self.weights)  # Adding L1 regularization term
                    db += lambda_ * np.sign(self.bias)  # Adding L1 regularization term
                elif regularization == 'both':
                    dw += lambda_ * np.sign(self.weights)  # Adding L1 regularization term
                    db += lambda_ * np.sign(self.bias)  # Adding L1 regularization term

                    dw += lambda_ * self.weights  # Adding L2 regularization term
                    db += lambda_ * self.bias  # Adding L2 regularization term

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
                self.weights -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
                self.bias -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

                t += 1

            y_predict =  self.evaluate(X, y)
            loss_history.append(epoch_loss)  # Append the average loss for this epoch to the history

        return loss_history, list(range(1, self.epochs + 1))

    def fit_sgd(self, X, y, batch_size=30, regularization=None, lambda_=0.1):
        # Normalization
        X = self._normalize(X)

        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        loss_history = []  # List to store the loss at each epoch

        # Stochastic gradient descent
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
                batch_loss = -np.mean(yi * np.log(predictions) + (1 - yi) * np.log(1 - predictions))
                epoch_loss += batch_loss * (len(xi) / num_samples)  # Weighted average for the epoch

                # Compute gradients
                dw = np.dot(xi.T, (predictions - yi))
                db = np.sum(predictions - yi)

                # Regularization
                if regularization == 'l2':
                    dw += lambda_ * self.weights  # Adding L2 regularization term
                    db += lambda_ * self.bias  # Adding L2 regularization term
                elif regularization == 'l1':
                    dw += lambda_ * np.sign(self.weights)  # Adding L1 regularization term
                    db += lambda_ * np.sign(self.bias)  # Adding L1 regularization term

                # Update parameters
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

            self.evaluate(X, y)
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
        plt.title('Receiver Operating Characteristic (ROC) Curve')
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
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.grid(True)

    def confusion_matrix(self, y_predict, y_test):
        print("Confusion Matrix:")

        y_predict = [1 if i == True else 0 for i in y_predict]
        y_test = [1 if i == True else 0 for i in y_test]

        print(len(y_predict))
        print(len(y_test))


import matplotlib.pyplot as plt


def plot_loss_vs_epochs(loss_epoch_pairs, legends):
    plt.figure(figsize=(10, 5))  # Set the figure size

    # Iterate over the list of loss-epoch pairs and legends
    for (loss_history, epochs), legend in zip(loss_epoch_pairs, legends):
        plt.plot(epochs, loss_history, label=legend)

    plt.title('Training Loss vs. Epochs')  # Set the title of the graph
    plt.xlabel('Epochs')  # Set the x-axis label
    plt.ylabel('Loss')  # Set the y-axis label
    plt.legend()  # Show legend
    plt.grid(True)  # Show grid
    plt.tight_layout()  # Adjust the layout


#==============================================================================================
### Normality check
## Histogram
# Define the number of rows and columns for subplots
while False:
    n_rows = 6
    n_cols = 5

    # Create a figure and a grid of subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        # Plot histogram on each subplot
        if i < len(X_fit.columns):
            ax.hist(X_fit.iloc[:, i], bins=20, alpha=0.7, color='blue')
            ax.set_title(X_fit.columns[i], fontsize=12)  # Adjusted font size
        else:
            ax.axis('off')  # Hide unused subplots

    # Adjust layout to prevent overlapping
    plt.tight_layout(pad=2.0, w_pad=1.0, h_pad=6.0)
    plt.show()

    break

## QQ-plot
# Define the number of rows and columns for subplots
while False:
    n_rows = 6
    n_cols = 5

    # Create a figure and a grid of subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        # Plot Q-Q plot on each subplot
        if i < len(X_fit.columns):
            stats.probplot(X_fit.iloc[:, i], dist="norm", plot=ax)
            ax.set_title(X_fit.columns[i], fontsize=12)  # Adjusted font size
            ax.set_xticklabels([])  # Hide x-axis tick labels
            ax.set_yticklabels([])  # Hide y-axis tick labels
            ax.set_xlabel('')  # Clear x-axis label
            ax.set_ylabel('')  # Clear y-axis label
        else:
            ax.axis('off')  # Hide unused subplots

    # Adjust layout to prevent overlapping
    plt.tight_layout(pad=2.0, w_pad=1.0, h_pad=6.0)
    plt.show()
    break

## Outlers removal
# Define a function to replace outliers with NaN
def replace_outliers_with_nan(data, m=3):
    mean = np.mean(data)
    std = np.std(data)

    # Identify outliers
    outliers = (data < (mean - m * std)) | (data > (mean + m * std))

    # Replace outliers with NaN
    data[outliers] = np.nan
    return data


# Apply the function to each column in the DataFrame
df_cleaned = X_fit.apply(lambda col: replace_outliers_with_nan(col, m=3))
# Calculate the mean of each column, ignoring NaN values
means = df_cleaned.mean()
# Fill NaN values with the mean of each column
df_cleaned = df_cleaned.fillna(means)

## Histogram
# Define the number of rows and columns for subplots
while False:
    n_rows = 6
    n_cols = 5

    # Create a figure and a grid of subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        # Plot histogram on each subplot
        if i < len(df_cleaned.columns):
            ax.hist(df_cleaned.iloc[:, i], bins=20, alpha=0.7, color='blue')
            ax.set_title(df_cleaned.columns[i], fontsize=12)  # Adjusted font size
        else:
            ax.axis('off')  # Hide unused subplots

    # Adjust layout to prevent overlapping
    plt.tight_layout(pad=2.0, w_pad=1.0, h_pad=6.0)
    plt.show()

    break

## QQ-plot
# Define the number of rows and columns for subplots
while False:
    n_rows = 6
    n_cols = 5

    # Create a figure and a grid of subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        # Plot Q-Q plot on each subplot
        if i < len(df_cleaned.columns):
            stats.probplot(df_cleaned.iloc[:, i], dist="norm", plot=ax)
            ax.set_title(df_cleaned.columns[i], fontsize=12)  # Adjusted font size
            ax.set_xticklabels([])  # Hide x-axis tick labels
            ax.set_yticklabels([])  # Hide y-axis tick labels
            ax.set_xlabel('')  # Clear x-axis label
            ax.set_ylabel('')  # Clear y-axis label
        else:
            ax.axis('off')  # Hide unused subplots

    # Adjust layout to prevent overlapping
    plt.tight_layout(pad=2.0, w_pad=1.0, h_pad=6.0)
    plt.show()
    break

X_fit = df_cleaned
#==============================================================================================
### Check for multicollinearity among predictors.
# Correlation Matrix
while False:
    correlation_matrix = X_fit.corr()

    # Display the correlation matrix
    print(correlation_matrix)


    plt.figure(figsize=(20, 16))
    sn.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.show()

    break

# VIF
while True:
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_fit.columns

    # Calculating VIF for each feature
    vif_data["VIF"] = [variance_inflation_factor(X_fit.values, i) for i in range(len(X_fit.columns))]

    print(vif_data)

    break

#==============================================================================================
### PCA
while True:
    # Standardize the data
    scaler = StandardScaler()
    data_std = scaler.fit_transform(X_fit)

    # Perform PCA
    pca = PCA(n_components=30)  # Reduce to 2 components for visualization
    principalComponents = pca.fit_transform(data_std)

    # Create a DataFrame with the principal components
    principalDf = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12', 'PC13', 'PC14', 'PC15', 'PC16', 'PC17', 'PC18', 'PC19', 'PC20', 'PC21', 'PC22', 'PC23', 'PC24', 'PC25', 'PC26', 'PC27', 'PC28', 'PC29', 'PC30'])
    # Only take the top 6 parameters
    principalDf = principalDf[['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6']]

    X_PCA = np.asarray(principalDf)

    # Explained variance ratio
    explained_variance = pca.explained_variance_ratio_

    # Eigenvalues (explained variance)
    eigenvalues = pca.explained_variance_

    # Create a DataFrame with the eigenvalues
    eigenvalues_df = pd.DataFrame(eigenvalues, index=['PC' + str(i + 1) for i in range(len(eigenvalues))],
                                  columns=['Eigenvalue'])

    # Sort the DataFrame by eigenvalues in descending order
    eigenvalues_df.sort_values(by='Eigenvalue', ascending=False, inplace=True)

    # Print the eigenvalues and their ranking
    print(eigenvalues_df)

    break


#==============================================================================================
### MODELS
#   Train/Test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
X_train_PCA, X_test_PCA, Y_train_PCA, Y_test_PCA = train_test_split(X_PCA, Y, test_size=0.3, random_state=42)

X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
Y_train = np.asarray(Y_train)
Y_test = np.asarray(Y_test)

X_train_PCA = np.asarray(X_train_PCA)
X_test_PCA = np.asarray(X_test_PCA)
Y_train_PCA = np.asarray(Y_train_PCA)
Y_test_PCA = np.asarray(Y_test_PCA)

#   Training models
model_0 = LogisticRegression(epochs=40)
model_0_PCA = LogisticRegression(epochs=40)
model_1 = LogisticRegression(epochs=40)
model_2 = LogisticRegression(epochs=40)
model_3 = LogisticRegression(epochs=40)
model_4 = LogisticRegression(epochs=40)


print("This is fit model")
a0, b0 = model_0.fit(X_train, Y_train, regularization='l2', lambda_=0.01)
print("This is fit model with PCA")
model_0_PCA.fit(X_train_PCA , Y_train_PCA )
print("This is adam model")
a1, b1 = model_1.fit_adam(X_train, Y_train, regularization='l2', lambda_=0.01, batch_size=32)
print("This is sgd model")
a2, b2 = model_2.fit_sgd(X_train, Y_train, batch_size=32, regularization='l2', lambda_=0.01)
print("This is adagrad model")
a3, b3 = model_3.fit_adagrad(X_train, Y_train, regularization='l2', lambda_=0.01, batch_size=32)
print("This is RMSProp model")
a4, b4 = model_4.fit_rmsprop(X_train, Y_train, batch_size=32, regularization='l2', lambda_=0.01)

while True:
    legends = ['Gradient descent', 'Adam', 'SGD', 'Adagrad', 'RMSProp']
    plot_loss_vs_epochs([(a0, b0), (a1, b1), (a2, b2), (a3, b3), (a4, b4)], legends)
    plt.show()  # Display the plot

    break

# ### ROC
# model_0.plot_roc_curve_train(X_train,Y_train)
# model_0.plot_roc_curve_test(X_test,Y_test)
# plt.show()
#
# model_0_PCA.plot_roc_curve_train(X_train_PCA,Y_train_PCA)
# model_0_PCA.plot_roc_curve_test(X_test_PCA,Y_test_PCA)
# plt.show()
#
# model_1.plot_roc_curve_train(X_train,Y_train)
# model_1.plot_roc_curve_test(X_test,Y_test)
# plt.show()
#
# model_2.plot_roc_curve_train(X_train,Y_train)
# model_2.plot_roc_curve_test(X_test,Y_test)
# plt.show()
#
# model_3.plot_roc_curve_train(X_train,Y_train)
# model_3.plot_roc_curve_test(X_test,Y_test)
# plt.show()
#
# model_4.plot_roc_curve_train(X_train,Y_train)
# model_4.plot_roc_curve_test(X_test,Y_test)
# plt.show()

# # # Save to excel
# with pd.ExcelWriter('HW3_Analysis.xlsx') as writer:
#
# #  Save main database
#     vif_data.to_excel(writer, sheet_name='VIF')
#     eigenvalues_df.to_excel(writer, sheet_name='PCA')
