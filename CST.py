import matplotlib.pyplot as plt 
import numpy as np 
import math as mh

def loaddata(filename):
    """
    the airfoil data file must be formatted in Profili saving style
    """
    with open(filename, 'r') as f:
        next(f)
        lines = f.readlines()
        X,Y,Z = [],[],[]
        for line in lines:
            value = [float(s) for s in line.split()]
            X.append(value[0])
            Y.append(value[1])
            Z.append(value[2])

    X = np.array(X)
    Y = np.array(Y)

    x_n = np.where(X == 0)
    x_id = x_n[0][0]
    size = len(X)

    return (X, Y, x_id, size)

def datasplit(x, y, x_id, size):
    x_up = sorted(x[0:x_id+1])
    y_up = y[0:x_id+1]
    x_up = np.array(x_up)
    y_up = np.array(y_up)
    y_up = y_up[::-1]

    x_low = sorted(x[x_id:size+1])
    y_low = y[x_id:size+1]
    x_low = np.array(x_low)
    y_low = np.array(y_low)

    return (x_up, y_up, x_low, y_low)

def compute_half_alpha_thickness_te(x, y):
    """
    x: x positive direction data distribution
    y: y data distribution within x positive direction
    """
    x_n = len(x)
    x1 = x[x_n-1]
    y1 = y[x_n-1]
    x2 = x[x_n-3]
    y2 = y[x_n-3]
    deltax = abs(x1-x2)
    deltay = abs(y1-y2)
    half_alpha_te = mh.atan(deltay/deltax)

    y_te = abs(y1)

    return half_alpha_te, y_te


def classfunction(x):
    """
    x: the x-direction data distribution of loaded airfoil data
    """
    N_1 = 0.5
    N_2 = 1.0
    x_n = len(x)
    C = np.empty((1, x_n))
    C = np.power(x, N_1) * np.power(1-x, N_2)
    C = np.array(C)

    return C

def bernstein(n, x):
    """
    n: the order of Bernstein Polynomial 
    x: the x-direction data distribution of loaded airfoil data
    """
    x_n = len(x)
    B = np.empty((x_n, n+1))
    for i in range(0, x_n):
        for j in range(0, n+1):
            B[i, j] = np.math.factorial(n)/(np.math.factorial(j)*np.math.factorial(n-j)) * np.power(x[i], j) * np.power(1-x[i], n-j)

    return B

def shapefunction_fit(R_le, Y_te, alpha_te, x, y):
    """
    R_le: leading edge radius
    Y_te: half of tailing edge thickness
    alpha_te: half of tailing edge alpha (/rad)
    """
    x_n = len(x)
    S = np.empty((1, x_n))
    S[0] = np.sqrt(2*R_le)
    S[0, x_n-1] = mh.tan(alpha_te) + Y_te

    C = classfunction(x)  # function calls

    for i in range(1, x_n-1):
        S[0, i] = (y[i] - x[i]*Y_te) / C[i]

    return S

def comp_initial_control_points(R_le, Y_te, alpha_te, x, y, n):
    B = bernstein(n, x)  # function calls
    S = shapefunction_fit(R_le, Y_te, alpha_te, x, y) # function calls
    B_pinv = np.linalg.pinv(B)
    a = np.dot(B_pinv, S.T)
    a = np.array(a)

    return a

def shapefunction(a, B):
    """
    a: fitting control points
    B: n-order Bernstein polynomial
    """
    x_n = B.shape[0]
    a_n = B.shape[1]
    for i in range(0, x_n):
        for j in range(0, a_n):
            B[i, j] = B[i, j] * a[j]

    S = np.empty((x_n, 1))
    for i in range(0, x_n):
        S[i, 0] = B[i, :].sum()

    return S

def CST_fitting(C, S, x, y_te):
    """
    C: class function
    S: shape function
    x: airfoil x-direction data distribution
    y_te: half of the tailing edge thickness
    """
    x_n = S.shape[0]
    y_CST = np.empty((x_n, 1))

    for i in range(0, x_n):
        y_CST[i, 0] = C[i] * S[i, 0] + x[i] * y_te

    return y_CST

def error_eval(y, y_CST):
    error = y - y_CST

    return error

def main():
    """
    This is the main function 
    """
    filename = 'NACA0012.dat'
    (_x, _y, _x_id, _size) = loaddata(filename) # function calls
    (x_up, y_up, x_low, y_low) = datasplit(_x, _y, _x_id, _size) # function calls

    R_le = 0.0125  # leading edge radius
    B_order = 5    # Bernstein polynomial order

    """
    up_surface fitting
    """
    alpha_te, Y_te = compute_half_alpha_thickness_te(x_up, y_up)
    C_up = classfunction(x_up) # function calls
    B_up = bernstein(B_order, x_up) # function calls
    a_up = comp_initial_control_points(R_le, Y_te, alpha_te, x_up, y_up, B_order) # function calls
    S_up = shapefunction(a_up, B_up)
    y_CST_up = CST_fitting(C_up, S_up, x_up, Y_te)

    """
    low_surface fitting 
    First we tranform the y value in y-positive direction
    Finally we transform the y_CST_low value in the negative direction
    """ 
    y_low = - y_low[:]
    alpha_te, Y_te = compute_half_alpha_thickness_te(x_low, y_low)
    C_low = classfunction(x_low) # function calls
    B_low = bernstein(B_order, x_low) # function calls
    a_low = comp_initial_control_points(R_le, Y_te, alpha_te, x_low, y_low, B_order) # function calls
    S_low = shapefunction(a_low, B_low)
    y_CST_low = CST_fitting(C_low, S_low, x_low, Y_te)
    y_CST_low = - y_CST_low[:, :]
    y_low = - y_low[:]


    plt.figure()
    plt.plot(x_up, y_up, label='up_surface', color='b')
    plt.plot(x_low, y_low, label='low_surface', color='k')
    plt.plot(x_up, y_CST_up[:, 0], 'ro', label='CST_fitting(n=6)', markersize=4)
    plt.plot(x_low, y_CST_low[:, 0], 'bo', label='CST_fitting(n=6)', markersize=4)
    plt.xlabel('x/c')
    plt.ylabel('y')
    plt.legend()
    plt.show()


    """
    error evaluating
    """
    error_up = error_eval(y_up, y_CST_up[:, 0])
    error_low = error_eval(y_low, y_CST_low[:, 0])
    
    plt.figure()
    plt.plot(x_up, error_up, label='up_error')
    plt.plot(x_low, error_low, label='low_error')
    plt.xlabel('x/c')
    plt.ylabel('error')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()