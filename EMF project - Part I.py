import numpy as np
import matplotlib.pyplot as plt


def grid(*args):
    # Make a grid of squares along the XY plane
    return np.stack(np.meshgrid(*args, indexing='ij'), axis=-1)


def calculate_off_diagonal(dot1, dot2, d):
    # Make the calcuation of the potential between two area
    # elemnts according to the method of moments that are not placed on the diagonal
    epsilon_z = 8.85 * 10 ** -12
    l_ij = ((4 * np.pi * epsilon_z) ** -1) * (d ** 2) * (
        np.sqrt((dot1[0] - dot2[0]) ** 2 + (dot1[1] - dot2[1]) ** 2)) ** -1
    return l_ij


def calculate_digaonal(dot1, d):
    # Make the calcuation of the potential between two area
    # elemnts according to the method of moments that are placed on the diagonal
    epsilon_z = 8.85 * 10 ** -12
    return (d * ((np.pi * epsilon_z) ** -1) * 0.8814)


def calculate_l(In_A, d):
    l = np.zeros((len(In_A), len(In_A)))  # base for the "l" matrix
    for i in range(len(In_A)):  # fill the "l" matirx with the proper values
        for j in range(i + 1):
            if i != j:
                l[i, j] = calculate_off_diagonal(In_A[i], In_A[j], d)
                l[j, i] = l[i, j]
            else:
                l[i, i] = calculate_digaonal(In_A[i], d)
    return l


def calc_flow(R, d, V):
    # This function manages the flow of the calculations for each 'd'
    epsilon = 0.001
    side = np.arange(-R, R + epsilon, d)  # Make a side of a square with length of 2R
    # and divided to equal segments of size d.
    Square = grid(list(side), list(side))  # Make a square grid out of the side
    In_A = []  # This list will store all (x,y) that are in A
    for i in range(Square.shape[0]):
        for j in range(Square.shape[1]):
            dot = Square[i, j]
            if (dot[0] ** 2 + dot[1] ** 2 <= R ** 2):
                In_A.append(dot)

    l = calculate_l(In_A, d)
    V = np.ones(len(In_A)) * V
    sigma = np.linalg.solve(l, V)  # Calculate sigma according to the formula in the instructions
    d_squared_v = np.ones(len(In_A)) * (d ** 2)
    Q = np.dot(d_squared_v, sigma.T)  # Calculate the total charge on the disc

    return Q


def q_of_d(d_stock, R, V):
    # In this function we evaluate the total charge Q for all d's that were given in question 1.b.2
    q = np.zeros(len(d_stock))
    Q_theo = 7.08 * (10 ** -11)
    err = np.zeros(len(d_stock))

    for i in range(len(d_stock)):
        q[i] = calc_flow(R, d_stock[i], V)
        err[i] = round(100 * ((abs((q[i]) - Q_theo)) / Q_theo), 4)
        print(f'For d of length {d_stock[i]}, the total charge Q is {q[i]}, with relative error of {err[i]}%\n')

    return q, err


def plot(d_stock, R, V):
    q, err = q_of_d(d_stock, R, V)
    Q_theo_v = np.ones(len(d_stock)) * 7.08 * (10 ** -11)

    x = np.array(d_stock)
    y = q

    figure, axis = plt.subplots(1, 2, figsize=(20, 10))
    axis[0].set_xlabel('d[m]')
    axis[0].set_ylabel('Q[C]')
    axis[0].set_title('Q(d)')
    axis[0].plot(x, y, x, Q_theo_v)
    axis[0].legend(['Nummeric Calculation', 'Theoretical Value'])

    axis[1].set_xlabel('d[m]')
    axis[1].set_ylabel('Relative error[%]')
    axis[1].set_title('Relative error as function of d')
    axis[1].plot(x, err)

    plt.show()


# Setting up parameters according to the question
R = 1
V = 1
d_stock = [0.25, 0.15, 0.12, 0.1, 0.075, 0.05, 0.025, 0.02]

plot(d_stock, R, V)