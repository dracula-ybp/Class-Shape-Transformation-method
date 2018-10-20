import matplotlib.pyplot as plt 
import numpy as np 
import math as mh 

class CST():
    def __init__(self, filename, Bernstein_order):
        self.filename = filename
        self.Bernstein_order = Bernstein_order

    def loaddata(self):
        filename = self.filename

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

    def datasplit(self, x, y, x_id, size):
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

    def compute_half_alpha_thickness_te(self, x, y):
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

    def classfunction(self, x):
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

    def bernstein(self, x):
        """
        n: the order of Bernstein Polynomial 
        x: the x-direction data distribution of loaded airfoil data
        """
        n = self.Bernstein_order

        x_n = len(x)
        B = np.empty((x_n, n+1))
        for i in range(0, x_n):
            for j in range(0, n+1):
                B[i, j] = np.math.factorial(n)/(np.math.factorial(j)*np.math.factorial(n-j)) * np.power(x[i], j) * np.power(1-x[i], n-j)

        return B

    def shapefunction_fit(self, R_le, Y_te, alpha_te, x, y):
        """
        R_le: leading edge radius
        Y_te: half of tailing edge thickness
        alpha_te: half of tailing edge alpha (/rad)
        """
        x_n = len(x)
        S = np.empty((1, x_n))
        S[0] = np.sqrt(2*R_le)
        S[0, x_n-1] = mh.tan(alpha_te) + Y_te

        C = self.classfunction(x)  # function calls

        for i in range(1, x_n-1):
            S[0, i] = (y[i] - x[i]*Y_te) / C[i]

        return S

    def comp_initial_control_points(self, R_le, Y_te, alpha_te, x, y):
        B = self.bernstein(x)  # function calls
        S = self.shapefunction_fit(R_le, Y_te, alpha_te, x, y) # function calls
        B_pinv = np.linalg.pinv(B)
        a = np.dot(B_pinv, S.T)
        a = np.array(a)

        return a

    def shapefunction(self, a, B):
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

    def CST_fitting(self, C, S, x, y_te):
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

    def error_eval(self, y, y_CST):
        error = y - y_CST

        return error

    def datsave(self, datname, x, y_fit):
        nu = len(x)
        with open(datname, 'w') as f:
            for i in range(nu):
                f.write(str(x[i]) + '\t' + str(y_fit[i]) + '\t' + '0.0' + '\n')
        with open(datname, 'r+') as f:
            content = f.read()
            f.seek(0, 0)
            f.write(str(nu) + '\n' + content)
