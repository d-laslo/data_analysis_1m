import numpy as np
import sympy as sym 

def linear_regression(x: np.array, y: np.array) -> np.array:
    if (x.shape[0] != y.shape[0]):
        raise ValueError   
    x = np.column_stack(( np.ones(x.shape[0]), x ))
    w = [sym.symbols(f'w{i}') for i in range(x.shape[1])]

    J = np.sum((np.dot(x, w) - y)**2)
    result = sym.solve([sym.diff(J, w_index) for w_index in w], w)
    return np.array([result[w_index] for w_index in w])
    



if __name__ == '__main__':
    # y = np.array([60, 35, 20, 20, 15])
    # x = np.array([100, 150, 200, 250, 300]).reshape(5,1)
    y = np.array([9, 13, 16, 14, 21])
    x = np.array([5, 14.5, 12, 18, 6, 12, 7, 13, 8, 14]).reshape(5,2)
    res = linear_regression(x, y)
    print(res)

