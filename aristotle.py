#!/usr/bin/env python

"""
Python script to solve Aristotle's Number Puzzle

Author: Bill Tubbs, September 2017.

See here for a description of the puzzle:
https://hwiechers.blogspot.ca/2013/03/solving-artitotles-number-puzzle.html

Requires the following packages to be installed:
 - sympy
 - numpy

WARNING: This script can take a long time to execute!
"""


print("\nPython script to solve Aristotle's Number Puzzle")
print("------------------------------------------------")

# ----------------- First use SymPy to reduce equations -----------------

from sympy import symbols, solve
from sympy.parsing.sympy_parser import parse_expr

# Number of unknowns
n_unknowns = 19

variable_names = [chr(c + 97) for c in range(n_unknowns)]

# Create SymPy variables
variables = symbols(variable_names)

print("\n{} unknowns:".format(len(variables)))
print(variables)

possible_values = set(range(1, n_unknowns + 1))

print("\n{} possible values:".format(len(possible_values)))
print(possible_values)

# These strings define the equations to be solved
hexagon_rows = [
    'abc',
    'defg',
    'hijkl',
    'mnop',
    'qrs',
    'cgl',
    'bfkp',
    'aejos',
    'dinr',
    'hmq',
    'lps',
    'gkor',
    'cfjnq',
    'beim',
    'adh'
]

def make_expression(chars, rhs=0):
    return parse_expr('+'.join(list(chars)) + '-' + str(rhs))

expressions = []
for chars in hexagon_rows:
    expressions.append(make_expression(chars, 38))

print("\n{} equations to solve:".format(len(expressions)))

for expr in expressions:
    print("{} = 0".format(expr))

# Try to solve the equations
# (They can't be solved but SymPy reduces them down)
reduced_expressions = solve(expressions)

print("\nReduced to {} equations:".format(len(reduced_expressions)))
for var, expr in reduced_expressions.items():
    print("{} = {}".format(var, expr))


# -------------- Now set up the problem in Numpy arrays --------------

import numpy as np

dependent_variables = list(reduced_expressions.keys())
independent_variables = list(set(variables) - set(reduced_expressions.keys()))

print("\n{} independent variables:".format(len(independent_variables)))
print(independent_variables)

print("\n{} dependent variables:".format(len(dependent_variables)))
print(dependent_variables)

# Generate array representations of expressions

# First check all the terms in the experessions
# match the list of independent variables
terms = set()
for var, expr in reduced_expressions.items():
    terms = terms.union(set(expr.as_coefficients_dict().keys()))

diff = terms.difference(set(independent_variables))

if len(diff) is not 1:
    print("Error: independent variables do not match the expressions")

terms = independent_variables + list(diff)

ncols = len(terms)
test_arrays = []

for var, expr in reduced_expressions.items():
    coefficients = expr.as_coefficients_dict()
    arr = np.zeros(ncols, dtype=np.int)
    for v, c in coefficients.items():
        arr[terms.index(v)] = c
    test_arrays.append(arr)

print("\nConstructed {} arrays of expression coefficients:".format(len(test_arrays)))
for arr in test_arrays:
    print(arr)
print("\nTerms: {}".format(terms))


# ------------------------- Find solutions -------------------------

from itertools import permutations, chain
import math

def n_chose_k(n, k, fac=math.factorial):
    return fac(n)//fac(n-k)

def permutations_as_arrays_generator(r, k, rows=100000):
    n = len(r)
    total_size = n_chose_k(n, k)
    it = permutations(r, k)
    while True:
        batch_size = min(total_size, rows)
        arr = np.fromiter(chain.from_iterable(it),
                          count=batch_size*k, dtype=int)
        total_size -= batch_size
        arr.resize((batch_size, k))
        yield arr

count_tested = 0
solutions = []
batch_size = 5000000
permutations_generator = permutations_as_arrays_generator(
    possible_values,
    len(independent_variables),
    rows=batch_size
)

def quick_test(value):
    """Defines a quick test that every variable value
    must meet"""
    return (value > 0) & (value < 20)

def test_values(values):
    """A condition that the full set of variable values
    must meet"""
    return values == possible_values

while True:

    n_passed = 0

    while n_passed == 0:

        #batch_of_values = array_from_generator(permutations_of_values, batch_size)
        batch_of_values = next(permutations_generator)
        batch_size = batch_of_values.shape[0]

        if batch_size == 0:
            print("\nAll {} permutations now checked".format(count_tested))
            break

        print("\nTesting {} permutations...".format(batch_size))
        # Add column of 1s to represent constant terms
        batch_of_values = np.concatenate(
            (
                batch_of_values,
                np.ones((batch_size, 1), dtype=np.int)
            ),
            axis=1
        )

        not_yet_rejected = np.ones((batch_size,), dtype=np.bool)

        evaluated_results = np.empty((batch_size, len(dependent_variables)), dtype=int)

        for i, test_array in enumerate(test_arrays):
            expression_evaluations = np.sum(test_array*batch_of_values[not_yet_rejected], axis=1)
            evaluated_results[not_yet_rejected, i] = expression_evaluations
            quick_test_results = quick_test(expression_evaluations)
            not_yet_rejected[not_yet_rejected] = quick_test_results

        n_passed = np.sum(not_yet_rejected)

        print(" result: {} passed quick test".format(n_passed))
        count_tested += batch_size

    if batch_of_values.shape[0] == 0:
        break

    n_solutions = 0
    for values, evaluated_values in zip(batch_of_values[not_yet_rejected], evaluated_results[not_yet_rejected]):

        if test_values(set(evaluated_values).union(values[:-1])):
            n_solutions += 1
            solution = dict(zip((independent_variables + dependent_variables), values[:-1].tolist() + evaluated_values.tolist()))
            solutions.append(solution)

    print(" {} solution{} found{}".format(
        'No' if n_solutions == 0 else n_solutions,
        '' if n_solutions == 1 else 's',
        ' '*20 + '*'*n_solutions if n_solutions > 0 else ''
    ))

print("\nIn total, {} solutions found:".format(len(solutions)))
print(variables)
for i, solution in enumerate(solutions):
    values = [item[1] for item in sorted([(x[0].name, x[1]) for x in solution.items()])]
    print("{}: {}".format(i, values))

print("\nChecking solutions...")
checks = []
for solution in solutions:
    condition1 = np.all(np.array([expression.subs(solution) for expression in expressions]) == 0)
    condition2 = set(solution.values()) == possible_values
    if condition1 and condition2:
        checks.append(True)
if all(checks):
    print(" All solutions are good.")
else:
    print(" There is a problem with solutions {}".format([i for i, check in enumerate(checks) if not check]))

