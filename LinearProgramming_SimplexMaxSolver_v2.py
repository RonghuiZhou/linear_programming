"""
Simplex solver for maximization problems using Linear Programming

Ronghui Zhou, zhou.uf@gmail.com

"""

import pulp as lp
import numpy as np

# set precision
np.set_printoptions(precision=1, suppress=True)

################################################################################################################
################################################################################################################
# Solving with my own defined function
# version 1:
def simplexMaxSolver_v1(M, numVar=2):
    """
    Simplex solver for maximization problems using Linear Programming

    INPUT:
        M: input matrix, see example here to learn how to construct
        numVar: number of variables, default 2

    OUTPUT:
        var_vals: values for values
        obj_value: value for the final optimized objective

    """

    print('\nSolve Linear Programming Problem with my own defined solver.')
    # print(f'\nM: \n{M}.')
    obj = M[-1, :-1]
    # print(f'\nObjective: \n{obj}.')

    # if there is any negative value in the objective row
    while np.sum(obj < 0) != 0:

        # print('\nThere is negative coefficient in the objetive row, continue to solve: ')
        # find the minimum column
        col_indx = np.argwhere(obj == obj.min())[0, 0]
        # print(f'\nThe most negative coefficient in the objetive row is in column no.: {col_indx}.')

        # find the ratio between the last column and the pivot column
        # for constraint rows
        ratio = M[:-1, -1] / M[:-1, col_indx]
        # print(f'\nFind the ratio between the last column and the pivot column, excluding the objective column.')

        # find the pivot row
        row_indx = np.argwhere(ratio == ratio.min())[0, 0]

        # normalize the pivot row based on the pivot column
        M[row_indx, :] = M[row_indx, :] / M[row_indx, col_indx]

        # make the pivot column of other rows to zero by adding a multiplication of the pivot row
        pivot_row = M[row_indx, :]
        # other rows exclude the pivot row
        M_excludePivot = np.delete(M.copy(), row_indx, axis=0)
        # make a stack for pivot rows based on the number of rows in MM
        pivot_row_stacks = np.tile(pivot_row, (M_excludePivot.shape[0], 1))

        for i in range(pivot_row_stacks.shape[0]):
            pivot_row_stacks[i, :] *= M_excludePivot[:, col_indx][i]

        M_excludePivot -= pivot_row_stacks

        # stack the pivot row back to get a new initial matrix to continue
        M = np.vstack([pivot_row, M_excludePivot])
        obj = M[-1, :-1]

    # get the maximum objective value
    obj_value = M[-1, -1]

    # get the varible values
    var_vals = []
    for i in range(numVar):
        var_vals.append(M[M[:, i] == 1][0, -1])

    print(f'\nThe final optimized maximum objective value: {obj_value}.')
    print(f'\nVariable values: {var_vals}.')
    return var_vals, obj_value

################################################################################################################


def simplexMaxSolver(M, numVar=2):
    """
    Simplex solver for maximization problems using Linear Programming

    INPUT:
        M: input matrix, see example here to learn how to construct
        numVar: number of variables, default 2

    OUTPUT:
        var_vals: values for values
        obj_value: value for the final optimized objective

    """

    # get the number of rows in this matrix
    num_rows = M.shape[0]

    # get the objective row
    obj_init = M[-1, :-1].copy()
    obj = obj_init.copy()

    # constract a list of variables
    Vars = ['x' + str(i + 1) for i in range(numVar)]

    # construct an objective statement
    obj_string = '\nThe objective is to maximize this equation: ' + ' + '.join(
        [str(-obj_init[i]) + ' * ' + Vars[i] for i in range(numVar)])
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
        for i in range(1, num_rows):
            M[i, :] = M[i, :] - M[i, col_indx] / M[0, col_indx] * M[0, :]

        # re-initiate the objective row
        obj = M[-1, :-1]

    # get the maximized objective value
    obj_value = M[-1, -1]

    # get the varible values
    var_vals = []
    for i in range(numVar):
        var_vals.append(M[M[:, i] == 1][0, -1])

    print(f'\nThe final optimized maximum objective value: {obj_value}; Variable values: {var_vals}.')

    # construct the final statement
    final_statement = ' + '.join([str(-obj_init[i]) + ' * ' + str(var_vals[i]) for i in range(numVar)]) + " = " + str(
        obj_value)

    print(f'\nFinal result: \t{final_statement}')

    return var_vals, obj_value
################################################################################################################
################################################################################################################
# Example 1:
################################################################################################################
################################################################################################################
# Solve with PuLP
print('\nSolve with PuLP.')
D = lp.LpVariable('D', lowBound=0)
S = lp.LpVariable('S', lowBound=0)

# create a maximization problem
profit = lp.LpProblem('Maximum_Profit', lp.LpMaximize)

# objective function

profit += 10 * S + 9 * D

# add constraints
profit += 7/10 * S + D <= 630

profit += 1/2 * S + 5/6 * D <= 600

profit += 1 * S + 2/3 * D <= 708

profit += 1/10 * S + 1/4 * D <= 135

# solve

status = profit.solve()

print(f'\nS: {lp.value(S)}; \tD: {lp.value(D)}.')

for variable in profit.variables():
    print(variable.name, '=', variable.value())

print(f'\nMaximum profit: {profit.objective.value()}.')

################################################################################################################
# Solve with my own function
# example 1
# profit += 10 * S + 9 * D

# # add constraints
# profit += 7/10 * S + D <= 630

# profit += 1/2 * S + 5/6 * D <= 600

# profit += 1 * S + 2/3 * D <= 708

# profit += 1/10 * S + 1/4 * D <= 135


M = np.array([[7/10, 1, 1, 0, 0, 0, 0, 630],
              [1/2, 5/6, 0, 1, 0, 0, 0, 600],
              [1, 2/3, 0, 0, 1, 0, 0, 708],
              [1/10, 1/4, 0, 0, 0, 1, 0, 135],
              [-10, -9, 0, 0, 0, 0, 1, 0]])

var_vals, obj_value = simplexMaxSolver(M, numVar = 2)


################################################################################################################
################################################################################################################
# Example 2:
################################################################################################################
################################################################################################################
# Solve with PuLP
# Solving with PuLP
print('\nSolve with PuLP.')
x1 = lp.LpVariable('x1', lowBound=0)
x2 = lp.LpVariable('x2', lowBound=0)

# create a maximization problem
profit = lp.LpProblem('Maximum_Profit', lp.LpMaximize)

# objective function

profit += 5 * x1 + 4 * x2

# add constraints
profit += 3 * x1 + 5 * x2 <= 78

profit += 4 * x1 + x2  <= 36

profit += x1 >= 0

profit += x2 >= 0

# solve

status = profit.solve()

# print(f'\nx1: {lp.value(x1)}; \tx2: {lp.value(x2)}.')

for variable in profit.variables():
    print(variable.name, '=', variable.value())

print(f'\nMaximum profit: {profit.objective.value()}\n.')

################################################################################################################
# Solve with my own function
# example 2


M = np.array([[3, 5, 1, 0, 0, 78],
             [1, 1/4, 0, 1/4, 0, 9],
             [-5, -4, 0, 0, 1, 0]])
var_vals, obj_value = simplexMaxSolver(M)