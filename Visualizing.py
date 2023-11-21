import numpy as np
import matplotlib.pyplot as plt


# # Define the Rosenbrock function, global minima is at (1,1)
# def f(x, y, a=1, b=100):
#     return (a - x) ** 2 + b * (y - x ** 2) ** 2
#
# # Define the gradient of the Rosenbrock function
# def gradient_f(x, y, a=1, b=100):
#     try:
#         dfdx = -2 * (a - x) - 4 * b * x * (y - x ** 2)
#         dfdy = 2 * b * (y - x ** 2)
#     except OverflowError:
#         print("Overflow encountered. Stopping iteration.")
#         return np.array([0, 0])
#
#     if np.isinf(dfdx) or np.isinf(dfdy):
#         print("Overflow encountered. Stopping iteration.")
#         return np.array([0, 0])
#
#     return np.array([dfdx, dfdy])


# Aluffi-Pentini function, global minima is at (−1.046680576580755,0)
def f(x, y):
    return 0.25 * x ** 4 + - 0.5 * x ** 2 + 0.1 * x + 0.5 * y **2

# Define the gradient of the Rosenbrock function
def gradient_f(x, y, noise_scale=0.1):
    dfdx = 1 * x ** 3 + - 1 * x + 0.1
    dfdy = 1 * y

    # Add Gaussian noise to simulate stochasticity
    dfdx = dfdx + np.random.normal(scale=noise_scale)
    dfdy = dfdy + np.random.normal(scale=noise_scale)

    return np.array([dfdx, dfdy])

# Rastrigin function definition for two variables x and y
def f(x, y, A=10):
    return A*2 + (x**2 - A * np.cos(2 * np.pi * x)) + (y**2 - A * np.cos(2 * np.pi * y))

# Gradient of the Rastrigin function for two variables x and y
def gradient_f(x, y, A=10, noise_scale=0.5):
    grad_x = 2*x + 2*np.pi*A * np.sin(2 * np.pi * x)
    grad_y = 2*y + 2*np.pi*A * np.sin(2 * np.pi * y)

    # Add Gaussian noise to simulate stochasticity
    grad_x = grad_x + np.random.normal(scale=noise_scale)
    grad_y = grad_y + np.random.normal(scale=noise_scale)

    return np.array([grad_x, grad_y])


# Steepest descent method
def steepest_descent(x0, y0, alpha):
    x, y = x0, y0
    history = [(x, y)]
    prev_f_val = f(x, y)
    iterations = 0

    while True:
        grad = gradient_f(x, y)
        x -= alpha * grad[0]
        y -= alpha * grad[1]
        history.append((x, y))

        # Check for convergence
        curr_f_val = f(x, y)
        if abs(curr_f_val - prev_f_val) < tol:
            print(f"Convergence reached at: x = {x}, y = {y} after {iterations} iterations")
            break
        prev_f_val = curr_f_val
        iterations += 1

    return np.array(history)

# Conjugate Gradient method
def conjugate_gradient(x0, y0):
    x, y = x0, y0
    history = [(x, y)]
    grad = gradient_f(x, y)
    d = -grad
    prev_f_val = f(x, y)
    iterations = 0

    while True:
        denom = np.dot(d, gradient_f(x + d[0], y + d[1]))

        if denom == 0:
            print("Division by zero encountered. Stopping iteration.")
            break

        alpha = np.dot(grad, grad) / denom
        x += alpha * d[0]
        y += alpha * d[1]
        history.append((x, y))

        new_grad = gradient_f(x, y)
        beta = np.dot(new_grad, new_grad) / np.dot(grad, grad)
        d = -new_grad + beta * d
        grad = new_grad

        # Check for convergence
        curr_f_val = f(x, y)
        if abs(curr_f_val - prev_f_val) < tol:
            print(f"Convergence reached at: x = {x}, y = {y} after {iterations} iterations")
            break
        prev_f_val = curr_f_val
        iterations += 1

    return np.array(history)


# Adagrad method
def adagrad(x0, y0, alpha, epsilon):
    x, y = x0, y0
    history = [(x, y)]
    grad_squared = np.zeros(2)
    prev_f_val = f(x, y)
    iterations = 0

    while True:
        grad = gradient_f(x, y)
        grad_squared += grad ** 2
        adjusted_grad = grad / (np.sqrt(grad_squared) + epsilon)

        x -= alpha * adjusted_grad[0]
        y -= alpha * adjusted_grad[1]
        history.append((x, y))

        # Check for convergence
        curr_f_val = f(x, y)
        if abs(curr_f_val - prev_f_val) < tol:
            print(f"Convergence reached at: x = {x}, y = {y} after {iterations} iterations")
            break
        prev_f_val = curr_f_val
        iterations += 1

    return np.array(history)


# RMSprop method
def rmsprop(x0, y0, alpha, beta):
    x, y = x0, y0
    history = [(x, y)]
    moving_avg_sq = np.zeros(2)
    prev_f_val = f(x, y)
    iterations = 0

    while True:
        grad = gradient_f(x, y)
        moving_avg_sq = beta * moving_avg_sq + (1 - beta) * (grad ** 2)
        adjusted_grad = grad / (np.sqrt(moving_avg_sq) + epsilon)

        x -= alpha * adjusted_grad[0]
        y -= alpha * adjusted_grad[1]
        history.append((x, y))

        # Check for convergence
        curr_f_val = f(x, y)
        if abs(curr_f_val - prev_f_val) < tol:
            print(f"Convergence reached at: x = {x}, y = {y} after {iterations} iterations")
            break
        prev_f_val = curr_f_val
        iterations += 1

    return np.array(history)

# Adam method
def adam(x0, y0, alpha, beta1, beta2, epsilon):
    x, y = x0, y0
    history = [(x, y)]
    m = np.zeros(2)
    v = np.zeros(2)
    t = 0
    prev_f_val = f(x, y)
    iterations = 0

    while True:
        t += 1
        grad = gradient_f(x, y)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        x -= alpha * m_hat[0] / (np.sqrt(v_hat[0]) + epsilon)
        y -= alpha * m_hat[1] / (np.sqrt(v_hat[1]) + epsilon)
        history.append((x, y))

        # Check for convergence
        curr_f_val = f(x, y)
        if abs(curr_f_val - prev_f_val) < tol:
            print(f"Convergence reached at: x = {x}, y = {y} after {iterations} iterations")
            break
        prev_f_val = curr_f_val
        iterations += 1

    return np.array(history)


# Adam method with belief
# def modified_adam_with_belief(x0, y0, alpha, beta1, beta2, epsilon, gradient_threshold):
#     x, y = x0, y0
#     history = [(x, y)]
#     m = np.zeros(2)
#     v = np.zeros(2)
#     t = 0
#     sigma = 0.5
#     prev_f_val = f(x, y)
#     prev_2_f_val = 0
#     iterations = 0
#
#     while True:
#         t += 1
#         grad = gradient_f(x, y)
#         m = beta1 * m + (1 - beta1) * grad
#         v = beta2 * v + (1 - beta2) * grad ** 2
#         m_hat = m / (1 - beta1 ** t)
#         v_hat = v / (1 - beta2 ** t)
#
#         x -= alpha * m_hat[0] / (np.sqrt(v_hat[0]) + epsilon)
#         y -= alpha * m_hat[1] / (np.sqrt(v_hat[1]) + epsilon)
#         history.append((x, y))
#
#         # Check for convergence
#         curr_f_val = f(x, y)
#         if abs(curr_f_val - prev_f_val) < tol:
#             # sigma = sigma + (curr_f_val - prev_f_val)/(prev_f_val - prev_2_f_val)
#
#
#
#         prev_2_f_val = prev_2_f_val
#         prev_f_val = curr_f_val
#         iterations += 1
#
#     return np.array(history)



# Initial point
x0, y0 = 1.25, 1.0


# Learning rate
alpha = 0.1

# Adam parameters
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

# Tolerance for stopping criterion
tol = 1e-4

# Choose optimization method: 'sd' for Steepest Descent, 'cg' for Conjugate Gradient, 'adam' for Adam
method = 'adam_belief'

if method == 'sd':
    history = steepest_descent(x0, y0, alpha)
elif method == 'cg':
    history = conjugate_gradient(x0, y0)
elif method == 'adam':
    history = adam(x0, y0, alpha, beta1, beta2, epsilon)
elif method == 'adagrad':
    history = adagrad(x0, y0, alpha, epsilon)
elif method == 'rms':
    history = rmsprop(x0, y0, alpha, beta1)
elif method == 'adam_belief':
    history = modified_adam_with_belief(x0, y0, alpha, beta1, beta2, epsilon, gradient_threshold=1e-4)
else:
    print("Invalid method. Choose 'sd', 'cg','adagrad', 'rms' or 'adam'.")



# Create meshgrid for contour plot
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# Plotting
plt.figure(figsize=(20, 10))
plt.contour(X, Y, Z, 50, cmap='jet')
plt.scatter(history[:, 0], history[:, 1], c='red')
plt.plot(history[:, 0], history[:, 1], c='red', linestyle='--')
plt.title(f'{method.upper()} Method')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.show()



