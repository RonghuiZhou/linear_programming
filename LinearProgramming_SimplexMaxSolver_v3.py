"""
Simplex solver for maximization problems using Linear Programming

Ronghui Zhou, zhou.uf@gmail.com

"""

import pulp as lp
import numpy as np

# set precision
np.set_printoptions(precision=1, suppress=True)

################################################################################################################
def solveMaxLP(M, lowBound=0):
    nrows = M.shape[0]
    num_vars = M.shape[1] - 1

    varNames = ['x' + str(i) for i in range(num_vars)]

    Xs = [lp.LpVariable(var, lowBound=lowBound) for var in varNames]

    # create a maximization problem
    prob = lp.LpProblem('maximization', lp.LpMaximize)

    # objective function
    prob += np.sum(M[-1, :-1] * Xs)

    # add constraints
    for i in range(nrows - 1):
        prob += np.sum(M[i, :-1] * Xs) <= M[i, -1]

    # solve
    status = prob.solve()

    for variable in prob.variables():
        print(variable.name, '=', variable.value())

    print(f'\nMaximum profit: {prob.objective.value()}.')

M = np.array([[3, 5, 78],
             [4, 1, 36],
             [5, 4, 0]])
solveMaxLP(M, lowBound = 0)
################################################################################################################
# Solving with my own defined function

def simplexMaxSolver(M, numVar=2):
    # print(f'\nOriginal M: \n{M}.')
    # convert M
    M[-1, :] = -M[-1, :]
    nrows, ncols = M.shape[:2]
    eI = np.eye(nrows)

    M = np.hstack([M, eI])
    M[:, [numVar, -1]] = M[:, [-1, numVar]]

    # print(f'\nExpanded M: \n{M}.')

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

    # print(M)

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


M = np.array([[7/10, 1, 630],
              [1/2, 5/6, 600],
              [1, 2/3, 708],
              [1/10, 1/4, 135],
              [10, 9, 0]])


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

M = np.array([[3, 5, 78],
             [4, 1, 36],
             [5, 4, 0]])

simplexMaxSolver(M, numVar = 2)