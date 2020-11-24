import pulp as lp
import numpy as np

# set precision
np.set_printoptions(precision=1, suppress=True)

## Example 1: Minimization
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

def solveMinLP(M, lowBound=0):
    nrows = M.shape[0]
    num_vars = M.shape[1] - 1

    varNames = ['x' + str(i) for i in range(num_vars)]

    Xs = [lp.LpVariable(var, lowBound=lowBound) for var in varNames]

    # create a maximization problem
    prob = lp.LpProblem('minimization', lp.LpMinimize)

    # objective function
    prob += np.sum(M[-1, :-1] * Xs)

    # add constraints
    for i in range(nrows - 1):
        prob += np.sum(M[i, :-1] * Xs) >= M[i, -1]

    # solve
    status = prob.solve()

    for variable in prob.variables():
        print(variable.name, '=', variable.value())

    print(f'\nMaximum profit: {prob.objective.value()}.')

M = np.array([[2, 1, 3, 6],
              [1, 2, 4, 8],
              [3, 1, -2, 4],
              [1, 1, 3, 0]])

solveMinLP(M)


## Example 2: Maximization

def solveMaxLP(M, lowBound=0):
    nrows = M.shape[0]
    num_vars = M.shape[1] - 1

    varNames = ['x' + str(i) for i in range(num_vars)]

    Xs = [lp.LpVariable(var, lowBound=lowBound) for var in varNames]

    # create a maximization problem
    prob = lp.LpProblem('minimization', lp.LpMaximize)

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


# Solving with PuLP in the regular way
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

print(f'\nMaximum profit: {profit.objective.value()}.')