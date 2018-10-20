from airfoil_generation import class_shape_transformation
import matplotlib.pyplot as plt 
import numpy as np 
import math as mh

def CST_fitting(): 
    """
    This is the main function 
    Initial settings according to different type of airfoils
    There are only two sets needed to be modified
    (filename   and   leading edge radius)
    """

    filename = "NACA0012.dat"  # an optional choice is RAE2822.dat
    Bernstein_order = 5 # Bernstein polynomial order
    fitting_airfoil = class_shape_transformation.CST(filename, Bernstein_order)

    (_x, _y, _x_id, _size) = fitting_airfoil.loaddata()
    (x_up, y_up, x_low, y_low) = fitting_airfoil.datasplit(_x, _y, _x_id, _size)

    # R_le = 0.008496  # RAE2822 leading edge radius
    R_le = 0.0125  # NACA0012 leading edge radius

    """
    up_surface fitting
    """
    (alpha_te, Y_te) = fitting_airfoil.compute_half_alpha_thickness_te(x_up, y_up)
    C_up = fitting_airfoil.classfunction(x_up)
    B_up = fitting_airfoil.bernstein(x_up)
    a_up = fitting_airfoil.comp_initial_control_points(R_le, Y_te, alpha_te, x_up, y_up)
    S_up = fitting_airfoil.shapefunction(a_up, B_up)
    y_CST_up = fitting_airfoil.CST_fitting(C_up, S_up, x_up, Y_te)

    """
    low_surface fitting 
    First we tranform the y value in y-positive direction
    Finally we transform the y_CST_low value in the negative direction
    """ 
    y_low = - y_low[:]
    alpha_te, Y_te = fitting_airfoil.compute_half_alpha_thickness_te(x_low, y_low)
    C_low = fitting_airfoil.classfunction(x_low) # function calls
    B_low = fitting_airfoil.bernstein(x_low) # function calls
    a_low = fitting_airfoil.comp_initial_control_points(R_le, Y_te, alpha_te, x_low, y_low) # function calls
    S_low = fitting_airfoil.shapefunction(a_low, B_low)
    y_CST_low = fitting_airfoil.CST_fitting(C_low, S_low, x_low, Y_te)
    y_CST_low = - y_CST_low[:, :]
    y_low = - y_low[:]

    """
    save the fitting result to a pointwise style .dat file to do mesh generation
    """
    y_fit = np.empty((_size, 1))
    y_fit[0:len(x_up), 0] = y_CST_up[::-1, 0]
    for i in range(len(x_up), _size):
        y_fit[i, 0] = y_CST_low[i-len(x_low)+1, 0]

    datname = 'result.dat'
    fitting_airfoil.datsave(datname, _x, y_fit[:, 0])

    """
    CST-fitting result figure plotting
    """
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
    error_up = fitting_airfoil.error_eval(y_up, y_CST_up[:, 0])
    error_low = fitting_airfoil.error_eval(y_low, y_CST_low[:, 0])
    
    plt.figure()
    plt.plot(x_up, error_up, label='up_error')
    plt.plot(x_low, error_low, label='low_error')
    plt.xlabel('x/c')
    plt.ylabel('error')
    plt.legend()
    plt.show()

if __name__  == "__main__":
    CST_fitting()