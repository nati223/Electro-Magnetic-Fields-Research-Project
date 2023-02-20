import numpy as np
import matplotlib.pyplot as plt


def grid(*args):
    return np.stack(np.meshgrid(*args, indexing='ij'), axis=-1)


def calculate_3D(dot1, dot2, d, D):
    # Calculate the potential between two elemnts of different discs
    epsilon_z = 8.85 * 10 ** -12
    l_ij = ((4 * np.pi * epsilon_z) ** -1) * (d ** 2) * (
                (np.sqrt((dot1[0] - dot2[0]) ** 2 + (dot1[1] - dot2[1]) ** 2 + D ** 2)) ** -1)
    return l_ij


def calculate_off_diagonal(dot1, dot2, d):
    epsilon_z = 8.85 * 10 ** -12
    l_ij = ((4 * np.pi * epsilon_z) ** -1) * (d ** 2) * (
                (np.sqrt((dot1[0] - dot2[0]) ** 2 + (dot1[1] - dot2[1]) ** 2)) ** -1)
    return l_ij


def calculate_digaonal(dot1, d):
    epsilon_z = 8.85 * 10 ** -12
    return (d * ((np.pi * epsilon_z) ** -1) * 0.8814)


def calculate_l_AA(In_A, d):
    # Calculate the mutual potential matrix of a single disc
    l = np.zeros((len(In_A), len(In_A)))
    for i in range(len(In_A)):
        for j in range(i + 1):
            if i != j:
                l[i, j] = calculate_off_diagonal(In_A[i], In_A[j], d)
                l[j, i] = l[i, j]
            else:
                l[i, i] = calculate_digaonal(In_A[i], d)
    return l


def calculate_AB(In_A, d, D):
    # Calculating the matrix L_AB according to the formula that was presented in the intro of question 2
    l = np.zeros((len(In_A), len(In_A)))
    for i in range(len(In_A)):
        for j in range(len(In_A)):
            l[i, j] = calculate_3D(In_A[i], In_A[j], d, D)
    return l


def make_discrete_circle(R, d):
    epsilon = 0.001
    side = np.arange(-R, R + epsilon, d)
    Square = grid(list(side), list(side))
    In_A = []
    for i in range(Square.shape[0]):
        for j in range(Square.shape[1]):
            dot = Square[i, j]
            if (dot[0] ** 2 + dot[1] ** 2 <= R ** 2):
                In_A.append(dot)

    return In_A


def calc_flow(R, d, V1, V2, D, In_A, l_AA):
    l_BB = l_AA  # The calculation of "l" for each disc is the same
    l_AB = calculate_AB(In_A, d, D)
    l_BA = l_AB.T  # Transposing l_AB
    # Stack the matrixes in blocks according to the formula in the guidance section
    l_A = np.hstack([l_AA, l_AB])
    l_B = np.hstack([l_BA, l_BB])
    l = np.vstack([l_A, l_B])

    # Prepare the volatage vector
    V1 = np.ones(len(In_A)) * V1
    V2 = np.ones(len(In_A)) * V2
    V = np.hstack((V1, V2))

    # Calculate sigma
    sigma = np.linalg.solve(l, V)

    d_squared_v = np.ones(len(In_A)) * (d ** 2)
    double_d_squared_v = np.ones(2 * len(In_A)) * (d ** 2)

    Q1 = np.dot(sigma[0:len(In_A)], d_squared_v)
    Q2 = np.dot(sigma[len(In_A):], d_squared_v)
    Q = np.dot(sigma, double_d_squared_v)

    return Q1, Q2, Q


def C_of_D(D_stock, R, d, V1, V2, In_A, l_AA):
    C = np.ones(len(D_stock))
    C_theo = np.ones(len(D_stock))
    err = np.ones(len(D_stock))

    for i in range(len(D_stock)):
        Q1, Q2, Q = (calc_flow(R, d, V1, V2, D_stock[i], In_A, l_AA))
        C_theo[i] = np.pi * (8.85 * 10 ** -12) * (D_stock[i]) ** -1
        C[i] = Q1 / (V1 - V2)
        err[i] = round(100 * (abs(C[i] - C_theo[i]) / C_theo[i]), 4)
        print(
            f"For D of length {D_stock[i]} [m], the capacity C is {C[i]} [F], while the theoretical value C_theo is {C_theo[i]} [F]")
        print(f"The relative error is {err[i]}\n")

    return C, C_theo, err


def plot_range(R, d, V1, V2, In_A, l_AA):
    D_stock = np.array(
        [d / 2, 3 * d / 4, d, 5 * d / 4, 3 * d / 2, 2 * d, R / 10, R / 8, R / 5, R / 4, R / 2, 3 * R / 4, 7 * R / 8])

    C, C_theo, err = C_of_D(D_stock, R, d, V1, V2, In_A, l_AA)

    x = D_stock
    y = C

    figure, axis = plt.subplots(1, 2, figsize=(20, 10))
    axis[0].set_xlabel('D[m]')
    axis[0].set_ylabel('C[F]')
    axis[0].set_title('C(D)')
    axis[0].plot(x, y, x, C_theo)
    axis[0].legend(['Nummeric Calculation', 'Theoretical Value'])

    axis[1].set_xlabel('D[m]')
    axis[1].set_ylabel('Relative error[%]')
    axis[1].set_title('Relative error as function of D')
    axis[1].plot(x, err)

    plt.show()


print("========================= Question 2.a =============================\n")

# Setting up parameters
R = 1
V1 = 0.5
V2 = -0.5
d = 0.025
D_a = R / 2

# Calculating constant objects for the program
In_A = make_discrete_circle(R, d)
l_AA = calculate_l_AA(In_A, d)

epsilon_z = 8.85 * 10 ** -12
Q1_a, Q2_a, Q_a = calc_flow(R, d, V1, V2, D_a, In_A, l_AA)

C_a = Q1_a / (V1 - V2)
C_theo_a = (epsilon_z * (np.pi * (R ** 2))) / D_a
err_a = round(100 * (abs(C_a - C_theo_a) / C_theo_a), 4)

print(f"The charge on the upper disc is {Q1_a} [C], and the charge on the lower disc is {Q2_a} [C]")
print(f"The total charge is {Q_a} [C]")
print(f"the capacitance in the nummeric calculation is {C_a} [F], while the theoretical value is {C_theo_a} [F]")
print(f"The realtive error of is {err_a}%\n")

print("========================= Question 2.b =============================\n")

# Setting up parameters
D_b = R / 5

Q1_b, Q2_b, Q_b = calc_flow(R, d, V1, V2, D_b, In_A, l_AA)
C_b = Q1_b / (V1 - V2)
C_theo_b = (epsilon_z * (np.pi * (R ** 2))) / D_b
err_b = round(100 * (abs(C_b - C_theo_b) / C_theo_b), 4)

print(f"The charge on the upper disc is {Q1_a} [C], and the charge on the lower disc is {Q2_a} [C]")
print(f"The total charge is {Q_b} [C]")
print(f"the capacitance in the nummeric calculation is {C_theo_b} [F], while the theoretical value is {C_theo_a}")
print(f"The realtive error of is {err_b}%\n")

print("========================= Question 2.c =============================\n")

plot_range(R, d, V1, V2, In_A, l_AA)

print("========================= Question 2.d =============================\n")

print("Let's use Superposition to show the differences Between Question 2.a and 2.d")

V1 = 1
V2 = 0

print("The Voltage has now changed to V1 = 1v, V2 = 0\n")

print("Recalculating 2.a with the new voltages\n")

Q1_a_N1, Q2_a_N1, Q_a_N1 = calc_flow(R, d, V1, V2, D_a, In_A, l_AA)

print(f"The charge on the upper disc is {Q1_a_N1}, and on the lower disc is {Q2_a_N1}")
print(f"The total charge is {Q_a_N1}\n")

V1 = 0.5
V2 = 0.5

print("The Voltage has now changed to V1 = 0.5v, V2 = 0.5v\n")

print("Recalculating 2.a with the new voltages\n")

Q1_a_N2, Q2_a_N2, Q_a_N2 = calc_flow(R, d, V1, V2, D_a, In_A, l_AA)


print(f"For V1 = 0.5, V2 = -0.5 : The charge on the upper disc is {Q1_a} [C], the charge on the lower disc is {Q2_a} [C], and the total charge is {Q_a}[C]")
print(f"For V1 = 0.5, V2 = 0.5 : The charge on the upper disc is {Q1_a_N2} [C], the charge on the lower disc is {Q2_a_N2} [C], and the total charge is {Q_a_N1}[C]")
print(f"For V1 = 1v, V2 = 0v : The charge on the upper disc is {Q1_a_N1} [C],  the charge on the lower disc is {Q2_a_N1} [C], and the total charge is {Q_a_N2}[C]")
