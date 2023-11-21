import math
import numpy as np
from numpy import linalg as LA
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def fValue(z):
    return math.exp(-z[0])+math.exp(-z[1])+z[0]**2+5*z[1]**2-2*z[0]*z[1]
    
def fGradient(z):
    #Calulate the gradient
    return np.array([-math.exp(-z[0])+2*z[0]-2*z[1],
                     -math.exp(-z[1])+10*z[1]-2*z[0]])

def stepLength(z, p):
    #Use backtracking linear search
    alpha = 2
    rho = 0.9
    c = 0.2
    
    while True:
        Left = fValue(z+alpha*p)
        Right = fValue(z) + c*alpha*np.inner(p, fGradient(z))
        if Left <= Right:
            break
        else:
            alpha *= rho

    return alpha

z = np.array([1, 1])

df_data = pd.DataFrame(columns=['Step', 'x_k', 'y_k', 'f_k', 'p_k', 'alpha_k'])

Step = -1

while True:
    Step += 1

    # The value of the objective function at the current point
    f_k = fValue(z)

    # The gradient vector at the current point
    Gradient = fGradient(z)

    # The search direction, which is the negative gradient normalized by its Euclidean norm.
    p_k =  -Gradient/LA.norm(Gradient)

    # The step length determined using the backtracking linear search
    alpha_k = stepLength(z, p_k)

    zNew = z + alpha_k * p_k

    df_data.loc[len(df_data)] = [Step, z[0], z[1], f_k, p_k, alpha_k]

    if LA.norm(zNew - z) <= 1E-5:
        break
    else:
        z = zNew
    
print(df_data)



