import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000, normalize=False):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.normalize = normalize
        self.mean = None
        self.std = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # Normalization
    def _normalize(self, X):
        if self.mean is None or self.std is None:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
        return (X - self.mean) / (self.std + 1e-10)  # added epsilon to avoid division by zero

    def fit(self, X, y, regularization=None, lambda_=0.1, batch_size=32, learning_rate=0.001):

        if self.normalize:
            X = self._normalize(X)

        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        loss_history = []  # List to store the loss at each epoch
        previous_loss = float('inf')

        # Gradient descent
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
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

            self.evaluate(X, y)

            loss_history.append(epoch_loss)  # Append the average loss for this epoch to the history

        return loss_history, list(range(1, self.epochs + 1))

    def fit_sgd(self, X, y, batch_size=32, regularization=None, lambda_=0.1):
        if self.normalize:
            X = self._normalize(X)

        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        # Stochastic gradient descent
        for epoch in range(self.epochs):
            print(f"\rEpoch {epoch + 1}- ", end="", flush=True)
            indices = np.random.permutation(num_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, num_samples, batch_size):
                xi = X_shuffled[i:i + batch_size]
                yi = y_shuffled[i:i + batch_size]

                linear_model = np.dot(xi, self.weights) + self.bias
                predictions = self.sigmoid(linear_model)

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

    def fit_adam(self, X, y, batch_size=32, beta1=0.7, beta2=0.999, epsilon=1e-8, regularization=None, lambda_=0.1, learning_rate=0.001):
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
                    dw += lambda_ * self.weights  # Adding L2 regularization term
                    db += lambda_ * self.bias  # Adding L2 regularization term
                elif regularization == 'l1':
                    dw += lambda_ * np.sign(self.weights)  # Adding L1 regularization term
                    db += lambda_ * np.sign(self.bias)  # Adding L1 regularization term

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

            self.evaluate(X, y)
            loss_history.append(epoch_loss)  # Append the average loss for this epoch to the history

        return loss_history, list(range(1, self.epochs + 1))

    def fit_rmsprop(self, X, y, batch_size=32, beta=0.9, epsilon=1e-8, regularization=None, learning_rate=0.001):
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

    def fit_adagrad(self, X, y, batch_size=32, epsilon=1e-8, regularization=None, learning_rate=0.001):
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

    def predict(self, X):
        if self.normalize:
            X = self._normalize(X)

        linear_model = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(linear_model)
        class_predictions = [1 if i > 0.5 else 0 for i in predictions]
        return class_predictions

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(linear_model)
        return predictions

    def evaluate(self, X_test, y_test):
        if self.normalize:
            X_test = self._normalize(X_test)
        predictions = self.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        print(f'Model Accuracy: {accuracy * 100:.2f}%')
        return accuracy

    def significant_vars(self, threshold):
        significant_indices = np.where(abs(self.weights) >= threshold)[0]
        print(f'Significant variables (index): {significant_indices}')
        return significant_indices

    def plot_roc_curve(self, X, y_true):
        """
        Plot ROC curve.

        X: The input features
        y_true: True labels
        """

        if self.normalize:
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

        # Calculating AUC using the trapezoidal rule
        auc = np.trapz(tpr, x=fpr)

        # Plotting the ROC curve
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.5f})')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.grid(True)
        # plt.show()

        return fpr, tpr, auc

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

# Misc function to plot 25 images and compare true vs predicted labels
def plot_images(images, true_labels, predicted_labels, nrows=5, ncols=5):
    """
    Plots images along with their true and predicted labels.

    images: Array of image data.
    true_labels: Array of true labels.
    predicted_labels: Array of predicted labels.
    nrows: Number of rows in the subplot grid.
    ncols: Number of columns in the subplot grid.
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 6))
    for i, ax in enumerate(axes.flatten()):
        img_2d = images[i].reshape((28, 28))
        ax.imshow(img_2d, cmap='gray')
        ax.set_title(f"True: {true_labels[i]}\nPred: {predicted_labels[i]}", fontsize=12)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Misc function to plot the confusion matrix
def confusion_matrix(true_labels, predicted_labels, classes):
    """
    Calculate and plot the confusion matrix.

    true_labels: Actual labels
    predicted_labels: Predicted labels
    classes: List of unique class labels
    """
    num_classes = len(classes)
    cm = np.zeros((num_classes, num_classes), dtype=int)

    for true, pred in zip(true_labels, predicted_labels):
        cm[true, pred] += 1

    # Plotting
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False,
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    # plt.title('Confusion Matrix')
    # plt.show()

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
###########################################################
# Load dataset
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape)

# Preprocess the data: flatten the images and scale to [0, 1]
X_train = X_train.reshape((X_train.shape[0], -1)).astype(np.float32) / 255.
X_test = X_test.reshape((X_test.shape[0], -1)).astype(np.float32) / 255.

# Define the number of samples you want to pick
num_samples = 5000

# Generate random indices
random_indices = np.random.choice(X_train.shape[0], num_samples, replace=False)

# Select the random samples
X_train = X_train[random_indices]
y_train = y_train[random_indices]


# Create a binary classification problem: "0" or "not 0"
y_train_0 = np.where(y_train == 0, 1, 0)
y_test_0 = np.where(y_test == 0, 1, 0)

# Create a binary classification problem: "1" or "not 1"
y_train_1 = np.where(y_train == 1, 1, 0)
y_test_1 = np.where(y_test == 1, 1, 0)

# Create a binary classification problem: "2" or "not 2"
y_train_2 = np.where(y_train == 2, 1, 0)
y_test_2 = np.where(y_test == 2, 1, 0)

# Create a binary classification problem: "3" or "not 3"
y_train_3 = np.where(y_train == 3, 1, 0)
y_test_3 = np.where(y_test == 3, 1, 0)

# Create a binary classification problem: "4" or "not 4"
y_train_4 = np.where(y_train == 4, 1, 0)
y_test_4 = np.where(y_test == 4, 1, 0)

# Create a binary classification problem: "5" or "not 5"
y_train_5 = np.where(y_train == 5, 1, 0)
y_test_5 = np.where(y_test == 5, 1, 0)

# Create a binary classification problem: "6" or "not 6"
y_train_6 = np.where(y_train == 6, 1, 0)
y_test_6 = np.where(y_test == 6, 1, 0)

# Create a binary classification problem: "7" or "not 7"
y_train_7 = np.where(y_train == 7, 1, 0)
y_test_7 = np.where(y_test == 7, 1, 0)

# Create a binary classification problem: "8" or "not 8"
y_train_8 = np.where(y_train == 8, 1, 0)
y_test_8 = np.where(y_test == 8, 1, 0)

# Create a binary classification problem: "9" or "not 9"
y_train_9 = np.where(y_train == 9, 1, 0)
y_test_9 = np.where(y_test == 9, 1, 0)

epochs = 50

#==============================================================================================
## Hyperparameter sensitivity analysis
model_5_fit_sensitivity = LogisticRegression(epochs=epochs,normalize=False)
model_5_adagrad_sensitivity = LogisticRegression(epochs=epochs,normalize=False)
model_5_rmsprop_sensitivity = LogisticRegression(epochs=epochs,normalize=False)
model_5_adam_sensitivity = LogisticRegression(epochs=epochs,normalize=False)
# GD
alpha_fit = []
legends_fit = []
for learning_rate in [0.1, 0.05, 0.01, 0.002, 0.001]:
    a, b = model_5_fit_sensitivity.fit(X_train, y_train_5, learning_rate=learning_rate)
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
for learning_rate in [0.5, 0.1, 0.01, 0.001, 0.0001, 0.00001]:
    a, b = model_5_adagrad_sensitivity.fit_adagrad(X_train, y_train_5, regularization=None, learning_rate=learning_rate, batch_size=5000)
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
for learning_rate in [0.5, 0.1, 0.01, 0.001, 0.0001, 0.00001]:
    a, b = model_5_rmsprop_sensitivity.fit_rmsprop(X_train, y_train_5, regularization=None, learning_rate=learning_rate, batch_size=5000)
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
for learning_rate in [0.5, 0.1, 0.01, 0.001, 0.0001, 0.00001]:
    a, b = model_5_adam_sensitivity.fit_adam(X_train, y_train_5, regularization=None, learning_rate=learning_rate, batch_size=5000)
    alpha_adam.append((a, b))
    legends_adam.append(learning_rate)

while True:
    plot_loss_vs_epochs(alpha_adam, legends_adam)
    plt.title('Alpha sensitivity for Adam')
    # plt.show()  # Display the plot
    break

plt.show()
###############################################################
### STEEPEST GRADIENT DESCENT
print("######### Model 0 ##########")
model_0_fit = LogisticRegression(epochs=epochs)
a0_fit, b0_fit = model_0_fit.fit(X_train, y_train_0, batch_size=5000, learning_rate=0.1)

print("######### Model 1 ##########")
model_1_fit = LogisticRegression(epochs=epochs)
a1_fit, b1_fit = model_1_fit.fit(X_train, y_train_1, batch_size=5000, learning_rate=0.1)

print("######### Model 2 ##########")
model_2_fit = LogisticRegression(epochs=epochs)
a2_fit, b2_fit = model_2_fit.fit(X_train, y_train_2, batch_size=5000, learning_rate=0.1)

print("######### Model 3 ##########")
model_3_fit = LogisticRegression(epochs=epochs)
a3_fit, b3_fit = model_3_fit.fit(X_train, y_train_3, batch_size=5000, learning_rate=0.1)

print("######### Model 4 ##########")
model_4_fit = LogisticRegression(epochs=epochs)
a4_fit, b4_fit = model_4_fit.fit(X_train, y_train_4, batch_size=5000, learning_rate=0.1)

print("######### Model 5 ##########")
model_5_fit = LogisticRegression(epochs=epochs)
a5_fit, b5_fit = model_5_fit.fit(X_train, y_train_5, batch_size=5000, learning_rate=0.1)

print("######### Model 6 ##########")
model_6_fit = LogisticRegression(epochs=epochs)
a6_fit, b6_fit = model_6_fit.fit(X_train, y_train_6, batch_size=5000, learning_rate=0.1)

print("######### Model 7 ##########")
model_7_fit = LogisticRegression(epochs=epochs)
a7_fit, b7_fit = model_7_fit.fit(X_train, y_train_7, batch_size=5000, learning_rate=0.1)

print("######### Model 8 ##########")
model_8_fit = LogisticRegression(epochs=epochs)
a8_fit, b8_fit = model_8_fit.fit(X_train, y_train_8, batch_size=5000, learning_rate=0.1)

print("######### Model 9 ##########")
model_9_fit = LogisticRegression(epochs=epochs)
a9_fit, b9_fit = model_9_fit.fit(X_train, y_train_9, batch_size=5000, learning_rate=0.1)


y_pred_0_fit = model_0_fit.predict_proba(X_test)
y_pred_1_fit = model_1_fit.predict_proba(X_test)
y_pred_2_fit = model_2_fit.predict_proba(X_test)
y_pred_3_fit = model_3_fit.predict_proba(X_test)
y_pred_4_fit = model_4_fit.predict_proba(X_test)
y_pred_5_fit = model_5_fit.predict_proba(X_test)
y_pred_6_fit = model_6_fit.predict_proba(X_test)
y_pred_7_fit = model_7_fit.predict_proba(X_test)
y_pred_8_fit = model_8_fit.predict_proba(X_test)
y_pred_9_fit = model_9_fit.predict_proba(X_test)

Y_pred_fit = np.array([y_pred_0_fit,y_pred_1_fit,y_pred_2_fit,y_pred_3_fit,y_pred_4_fit,y_pred_5_fit,y_pred_6_fit,y_pred_7_fit,y_pred_8_fit,y_pred_9_fit])
predicted_labels = np.argmax(Y_pred_fit,axis=0)

## confusion matrix
labels = [0,1,2,3,4,5,6,7,8,9]
confusion_matrix(y_test, predicted_labels, labels)
plt.title('Confusion Matrix_Steepest Gradient Descent')
# plt.show()


#################################################################################
### ADAM
print("######### Model 0 ##########")
model_0_adam = LogisticRegression(epochs=epochs,normalize=False)
a0_adam, b0_adam = model_0_adam.fit_adam(X_train, y_train_0, regularization=None, lambda_=.01, batch_size=5000, learning_rate=0.01)

print("######### Model 1 ##########")
model_1_adam = LogisticRegression(epochs=epochs,normalize=False)
a1_adam, b1_adam = model_1_adam.fit_adam(X_train, y_train_1, regularization=None, lambda_=.01, batch_size=5000, learning_rate=0.01)

print("######### Model 2 ##########")
model_2_adam = LogisticRegression(epochs=epochs,normalize=False)
a2_adam, b2_adam = model_2_adam.fit_adam(X_train, y_train_2, regularization=None, lambda_=.01, batch_size=5000, learning_rate=0.01)

print("######### Model 3 ##########")
model_3_adam = LogisticRegression(epochs=epochs,normalize=False)
a3_adam, b3_adam = model_3_adam.fit_adam(X_train, y_train_3, regularization=None, lambda_=.01, batch_size=5000, learning_rate=0.01)

print("######### Model 4 ##########")
model_4_adam = LogisticRegression(epochs=epochs,normalize=False)
a4_adam, b4_adam = model_4_adam.fit_adam(X_train, y_train_4, regularization=None, lambda_=.01, batch_size=5000, learning_rate=0.01)

print("######### Model 5 ##########")
model_5_adam = LogisticRegression(epochs=epochs,normalize=False)
a5_adam, b5_adam = model_5_adam.fit_adam(X_train, y_train_5, regularization=None, lambda_=.01, batch_size=5000, learning_rate=0.01)

print("######### Model 6 ##########")
model_6_adam = LogisticRegression(epochs=epochs,normalize=False)
a6_adam, b6_adam = model_6_adam.fit_adam(X_train, y_train_6, regularization=None, lambda_=.01, batch_size=5000, learning_rate=0.01)

print("######### Model 7 ##########")
model_7_adam = LogisticRegression(epochs=epochs,normalize=False)
a7_adam, b7_adam = model_7_adam.fit_adam(X_train, y_train_7, regularization=None, lambda_=.01, batch_size=5000, learning_rate=0.01)

print("######### Model 8 ##########")
model_8_adam = LogisticRegression(epochs=epochs,normalize=False)
a8_adam, b8_adam = model_8_adam.fit_adam(X_train, y_train_8, regularization=None, lambda_=.01, batch_size=5000, learning_rate=0.01)

print("######### Model 9 ##########")
model_9_adam = LogisticRegression(epochs=epochs,normalize=False)
a9_adam, b9_adam = model_9_adam.fit_adam(X_train, y_train_9, regularization=None, lambda_=.01, batch_size=5000, learning_rate=0.01)


y_pred_0_adam = model_0_adam.predict_proba(X_test)
y_pred_1_adam = model_1_adam.predict_proba(X_test)
y_pred_2_adam = model_2_adam.predict_proba(X_test)
y_pred_3_adam = model_3_adam.predict_proba(X_test)
y_pred_4_adam = model_4_adam.predict_proba(X_test)
y_pred_5_adam = model_5_adam.predict_proba(X_test)
y_pred_6_adam = model_6_adam.predict_proba(X_test)
y_pred_7_adam = model_7_adam.predict_proba(X_test)
y_pred_8_adam = model_8_adam.predict_proba(X_test)
y_pred_9_adam = model_9_adam.predict_proba(X_test)

Y_pred_adam = np.array([y_pred_0_adam,y_pred_1_adam,y_pred_2_adam,y_pred_3_adam,y_pred_4_adam,y_pred_5_adam,y_pred_6_adam,y_pred_7_adam,y_pred_8_adam,y_pred_9_adam])
predicted_labels = np.argmax(Y_pred_adam,axis=0)

## confusion matrix
labels = [0,1,2,3,4,5,6,7,8,9]
confusion_matrix(y_test, predicted_labels, labels)
plt.title('Confusion Matrix_Adam')
# plt.show()

#################################################################################
### ADAGRAD
print("######### Model 0 ##########")
model_0_adagrad = LogisticRegression(epochs=epochs,normalize=False)
a0_adagrad, b0_adagrad = model_0_adagrad.fit_adagrad(X_train, y_train_0, regularization=None, batch_size=5000, learning_rate=0.1)

print("######### Model 1 ##########")
model_1_adagrad = LogisticRegression(epochs=epochs,normalize=False)
a1_adagrad, b1_adagrad = model_1_adagrad.fit_adagrad(X_train, y_train_1, regularization=None, batch_size=5000, learning_rate=0.1)

print("######### Model 2 ##########")
model_2_adagrad = LogisticRegression(epochs=epochs,normalize=False)
a2_adagrad, b2_adagrad = model_2_adagrad.fit_adagrad(X_train, y_train_2, regularization=None, batch_size=5000, learning_rate=0.1)

print("######### Model 3 ##########")
model_3_adagrad = LogisticRegression(epochs=epochs,normalize=False)
a3_adagrad, b3_adagrad = model_3_adagrad.fit_adagrad(X_train, y_train_3, regularization=None, batch_size=5000, learning_rate=0.1)

print("######### Model 4 ##########")
model_4_adagrad = LogisticRegression(epochs=epochs,normalize=False)
a4_adagrad, b4_adagrad = model_4_adagrad.fit_adagrad(X_train, y_train_4, regularization=None, batch_size=5000, learning_rate=0.1)

print("######### Model 5 ##########")
model_5_adagrad = LogisticRegression(epochs=epochs,normalize=False)
a5_adagrad, b5_adagrad = model_5_adagrad.fit_adagrad(X_train, y_train_5, regularization=None, batch_size=5000, learning_rate=0.1)

print("######### Model 6 ##########")
model_6_adagrad = LogisticRegression(epochs=epochs,normalize=False)
a6_adagrad, b6_adagrad = model_6_adagrad.fit_adagrad(X_train, y_train_6, regularization=None, batch_size=5000, learning_rate=0.1)

print("######### Model 7 ##########")
model_7_adagrad = LogisticRegression(epochs=epochs,normalize=False)
a7_adagrad, b7_adagrad = model_7_adagrad.fit_adagrad(X_train, y_train_7, regularization=None, batch_size=5000, learning_rate=0.1)

print("######### Model 8 ##########")
model_8_adagrad = LogisticRegression(epochs=epochs,normalize=False)
a8_adagrad, b8_adagrad = model_8_adagrad.fit_adagrad(X_train, y_train_8, regularization=None, batch_size=5000, learning_rate=0.1)

print("######### Model 9 ##########")
model_9_adagrad = LogisticRegression(epochs=epochs,normalize=False)
a9_adagrad, b9_adagrad = model_9_adagrad.fit_adagrad(X_train, y_train_9, regularization=None, batch_size=5000, learning_rate=0.1)


y_pred_0_adagrad = model_0_adagrad.predict_proba(X_test)
y_pred_1_adagrad = model_1_adagrad.predict_proba(X_test)
y_pred_2_adagrad = model_2_adagrad.predict_proba(X_test)
y_pred_3_adagrad = model_3_adagrad.predict_proba(X_test)
y_pred_4_adagrad = model_4_adagrad.predict_proba(X_test)
y_pred_5_adagrad = model_5_adagrad.predict_proba(X_test)
y_pred_6_adagrad = model_6_adagrad.predict_proba(X_test)
y_pred_7_adagrad = model_7_adagrad.predict_proba(X_test)
y_pred_8_adagrad = model_8_adagrad.predict_proba(X_test)
y_pred_9_adagrad = model_9_adagrad.predict_proba(X_test)

Y_pred_adagrad = np.array([y_pred_0_adagrad,y_pred_1_adagrad,y_pred_2_adagrad,y_pred_3_adagrad,y_pred_4_adagrad,y_pred_5_adagrad,y_pred_6_adagrad,y_pred_7_adagrad,y_pred_8_adagrad,y_pred_9_adagrad])
predicted_labels = np.argmax(Y_pred_adagrad,axis=0)

## confusion matrix
labels = [0,1,2,3,4,5,6,7,8,9]
confusion_matrix(y_test, predicted_labels, labels)
plt.title('Confusion Matrix_Adagrad')
# plt.show()

#################################################################################
### RMSPROP
print("######### Model 0 ##########")
model_0_rmsprop = LogisticRegression(epochs=epochs,normalize=False)
a0_rmsprop, b0_rmsprop = model_0_rmsprop.fit_rmsprop(X_train, y_train_0, regularization=None, batch_size=5000, learning_rate=0.01)

print("######### Model 1 ##########")
model_1_rmsprop = LogisticRegression(epochs=epochs,normalize=False)
a1_rmsprop, b1_rmsprop = model_1_rmsprop.fit_rmsprop(X_train, y_train_1, regularization=None, batch_size=5000, learning_rate=0.01)

print("######### Model 2 ##########")
model_2_rmsprop = LogisticRegression(epochs=epochs,normalize=False)
a2_rmsprop, b2_rmsprop = model_2_rmsprop.fit_rmsprop(X_train, y_train_2, regularization=None, batch_size=5000, learning_rate=0.01)

print("######### Model 3 ##########")
model_3_rmsprop = LogisticRegression(epochs=epochs,normalize=False)
a3_rmsprop, b3_rmsprop = model_3_rmsprop.fit_rmsprop(X_train, y_train_3, regularization=None, batch_size=5000, learning_rate=0.01)

print("######### Model 4 ##########")
model_4_rmsprop = LogisticRegression(epochs=epochs,normalize=False)
a4_rmsprop, b4_rmsprop = model_4_rmsprop.fit_rmsprop(X_train, y_train_4, regularization=None, batch_size=5000, learning_rate=0.01)

print("######### Model 5 ##########")
model_5_rmsprop = LogisticRegression(epochs=epochs,normalize=False)
a5_rmsprop, b5_rmsprop = model_5_rmsprop.fit_rmsprop(X_train, y_train_5, regularization=None, batch_size=5000, learning_rate=0.01)

print("######### Model 6 ##########")
model_6_rmsprop = LogisticRegression(epochs=epochs,normalize=False)
a6_rmsprop, b6_rmsprop = model_6_rmsprop.fit_rmsprop(X_train, y_train_6, regularization=None, batch_size=5000, learning_rate=0.01)

print("######### Model 7 ##########")
model_7_rmsprop = LogisticRegression(epochs=epochs,normalize=False)
a7_rmsprop, b7_rmsprop = model_7_rmsprop.fit_rmsprop(X_train, y_train_7, regularization=None, batch_size=5000, learning_rate=0.01)

print("######### Model 8 ##########")
model_8_rmsprop = LogisticRegression(epochs=epochs,normalize=False)
a8_rmsprop, b8_rmsprop = model_8_rmsprop.fit_rmsprop(X_train, y_train_8, regularization=None, batch_size=5000, learning_rate=0.01)

print("######### Model 9 ##########")
model_9_rmsprop = LogisticRegression(epochs=epochs,normalize=False)
a9_rmsprop, b9_rmsprop= model_9_rmsprop.fit_rmsprop(X_train, y_train_9, regularization=None, batch_size=5000, learning_rate=0.01)


y_pred_0_rmsprop = model_0_rmsprop.predict_proba(X_test)
y_pred_1_rmsprop = model_1_rmsprop.predict_proba(X_test)
y_pred_2_rmsprop = model_2_rmsprop.predict_proba(X_test)
y_pred_3_rmsprop = model_3_rmsprop.predict_proba(X_test)
y_pred_4_rmsprop = model_4_rmsprop.predict_proba(X_test)
y_pred_5_rmsprop = model_5_rmsprop.predict_proba(X_test)
y_pred_6_rmsprop = model_6_rmsprop.predict_proba(X_test)
y_pred_7_rmsprop = model_7_rmsprop.predict_proba(X_test)
y_pred_8_rmsprop = model_8_rmsprop.predict_proba(X_test)
y_pred_9_rmsprop = model_9_rmsprop.predict_proba(X_test)

Y_pred_rmsprop = np.array([y_pred_0_rmsprop,y_pred_1_rmsprop,y_pred_2_rmsprop,y_pred_3_rmsprop,y_pred_4_rmsprop,y_pred_5_rmsprop,y_pred_6_rmsprop,y_pred_7_rmsprop,y_pred_8_rmsprop,y_pred_9_rmsprop])
predicted_labels = np.argmax(Y_pred_rmsprop,axis=0)

## confusion matrix
labels = [0,1,2,3,4,5,6,7,8,9]
confusion_matrix(y_test, predicted_labels, labels)
plt.title('Confusion Matrix_RMSProp')
# plt.show()

############################################################################################
#   Plot loss vs epoch for 5
legends = ['Gradient descent', 'Adam', 'Adagrad', 'RMSProp']
plot_loss_vs_epochs([(a5_fit, b5_fit), (a5_adam, b5_adam), (a5_adagrad, b5_adagrad), (a5_rmsprop, b5_rmsprop)], legends)
plt.title('Training Loss vs. Epochs for 5')
plt.show()  # Display the plot

############################################################################################
## ROC curve for 5
fpr_model_5_fit, tpr_model_5_fit, auc_model_5_fit = model_5_fit.plot_roc_curve(
    X_test, y_test_5)
fpr_model_5_adam, tpr_model_5_adam, auc_model_5_adam = model_5_adam.plot_roc_curve(
    X_test, y_test_5)
fpr_model_5_adagrad, tpr_model_5_adagrad, auc_model_5_adagrad = model_5_adagrad.plot_roc_curve(
    X_test, y_test_5)
fpr_model_5_rmsprop, tpr_model_5_rmsprop, auc_model_5_rmsprop = model_5_rmsprop.plot_roc_curve(
    X_test, y_test_5)


model_5_ROC = [
    (fpr_model_5_fit, tpr_model_5_fit, auc_model_5_fit, 'Gradient Descent'),
    (fpr_model_5_adam, tpr_model_5_adam, auc_model_5_adam, 'Adam'),
    (fpr_model_5_adagrad, tpr_model_5_adagrad, auc_model_5_adagrad, 'Adagrad'),
    (fpr_model_5_rmsprop, tpr_model_5_rmsprop, auc_model_5_rmsprop, 'RMSProp')
]

plot_combined_roc_curve(model_5_ROC)
plt.title(f'ROC Curves of test sets_5')
plt.show()


