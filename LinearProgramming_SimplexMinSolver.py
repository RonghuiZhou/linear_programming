"""
Simplex solver for maximization problems using Linear Programming

Ronghui Zhou, zhou.uf@gmail.com

YouTube example: https://www.youtube.com/watch?v=nH-MkrTqqew

Problem:
    Obj:    Minimize C = x + y + 3z
    Constraints:
            2x + y + 3z >= 6
            x + 2y + 4z >=8
            3x + y - 2z >=4
            x, y, z >= 0

M = np.array([[2, 1, 3, 6],
              [1, 2, 4, 8],
              [3, 1, -2, 4],
              [1, 1, 3, 0]])


"""
import pulp as lp
import numpy as np
# set precision
np.set_printoptions(precision=1, suppress=True)

M = np.array([[2, 1, 3, 6],
              [1, 2, 4, 8],
              [3, 1, -2, 4],
              [1, 1, 3, 0]])

################################################################################################################
################################################################################################################
# Example 1:
################################################################################################################
################################################################################################################
# Solve with PuLP


# problem statement:
# minimize: z = x1 + x2 + 3 x3

# Constraints:
# 2 x1 + x2 + 3 x3 >= 6
# x1 + 2 x2 + 4 x3 >= 8
# 3 x1 + x2 - 2 x3 >= 4
# x1, x2, x3 >= 0
#
# M = np.array([[2, 1, 3, 6],
#               [1, 2, 4, 8],
#               [3, 1, -2, 4],
#               [1, 1, 3, 0]])


x1 = lp.LpVariable('x1', lowBound=0)
x2 = lp.LpVariable('x2', lowBound=0)
x3 = lp.LpVariable('x3', lowBound=0)

# create a maximization problem
prob = lp.LpProblem('minimization', lp.LpMinimize)

# objective function

prob += 1 * x1 + 1 * x2 + 3 * x3

# add constraints
prob += 2 * x1 + 1 * x2 + 3 * x3 >= 6
prob += 1 * x1 + 2 * x2 + 4 * x3 >= 8
prob += 3 * x1 + 1 * x2 - 2 * x3 >= 4


# solve

status = prob.solve()

for variable in prob.variables():
    print(variable.name, '=', variable.value())

print(f'\nMaximum profit: {prob.objective.value()}.')



################################################################################################################
################################################################################################################
# Solving with my own defined function

# Solving with my own defined function
# reference: https://www.youtube.com/watch?v=nH-MkrTqqew
# reference: https://www.youtube.com/watch?v=IUqobQjX6Mc

def convertMin2Max(M):
    """
    Convert the minimization matrix to a maximization matrix

    INPUT:
        MATRIX for minimization problem

    OUTPUT:
        MATRIX for maximization problem
    """
    #     nrows, ncols = M.shape[:2]
    #     M = np.delete(M, range(n_rows - 1, n_cols - 1), axis=1)

    nrows, ncols = M.shape[:2]
    M_transpose = M.T
    eI = np.eye(ncols)

    M_max = np.zeros((ncols, ncols + nrows))
    M_max[:, :nrows - 1] = M_transpose[:, :-1]
    M_max[:, nrows - 1:-1] = eI
    M_max[:, -1] = M_transpose[:, -1]

    M_max[-1, range(nrows)] = - M_max[-1, range(nrows)]

    return M_max


def simplexMinSolver(M, numVar=3):
    # reference: https://www.youtube.com/watch?v=IUqobQjX6Mc

    obj_init = M[-1, :-1].copy()

    # constract a list of variables
    Vars = ['x' + str(i + 1) for i in range(numVar)]

    # construct an objective statement
    obj_string = '\nThe objective is to minimize this equation: ' + ' + '.join(
        [str(obj_init[i]) + ' * ' + Vars[i] for i in range(numVar)])

    print(obj_string)

    M = convertMin2Max(M)
    print(M)
    n_rows, n_cols = M.shape[:2]

    # get the new objective row after converting from minimization to maximization

    obj = M[-1, :-1].copy()

    # constract a new list of variables
    Vars = ['y' + str(i + 1) for i in range(numVar)]

    # construct a new objective statement after
    obj_string = '\nThe objective is converted to maximize this equation: ' + ' + '.join(
        [str(-obj[i]) + ' * ' + Vars[i] for i in range(numVar)])

    print(obj_string)

    # if there is any negative value in the objective row
    while np.sum(obj < 0) != 0:

        # find the col index of the most negative coefficient in the objetive row
        col_indx = np.argwhere(obj == obj.min())[0, 0]

        # find the ratio between the last column and the pivot column for constraint rows
        ratio = M[:-1, -1] / M[:-1, col_indx]

        # find the pivot row where the ratio is the smallest
        row_indx = np.argwhere(ratio == ratio.min())[0, 0]

        # normalize the pivot row based on the pivot column
        M[row_indx, :] = M[row_indx, :] / M[row_indx, col_indx]

        # move the pivot row to the top
        M[[0, row_indx]] = M[[row_indx, 0]]

        # make the pivot column of other rows to zero by adding a multiplication of the pivot row
        for i in range(1, n_rows):
            M[i, :] = M[i, :] - M[i, col_indx] / M[0, col_indx] * M[0, :]

        # re-initiate the objective row
        obj = M[-1, :-1]

    print(f'After optimization: \n{M}.')
    # get the maximized objective value
    obj_value = M[-1, -1]

    # get the varible values
    var_vals = M[-1, n_rows - 1:n_rows - 1 + numVar]

    print(f'\nThe final optimized minimum objective value: {obj_value}; Variable values: {var_vals}.')

    # construct the final statement
    final_statement = ' + '.join([str(obj_init[i]) + ' * ' + str(var_vals[i]) for i in range(numVar)]) + " = " + str(
        obj_value)

    print(f'\nFinal maximization result: \t{final_statement}')

    return M
################################################################################################################
################################################################################################################
M = np.array([[2, 1, 3, 6],
              [1, 2, 4, 8],
              [3, 1, -2, 4],
              [1, 1, 3, 0]])

print('\nSolve minimization problem with user defined simplex solver.')
simplexMinSolver(M, numVar=3)
