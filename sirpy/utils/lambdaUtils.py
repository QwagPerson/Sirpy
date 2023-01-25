# define a lambda function that takes t, y, p and return 0 and assing it to a variable named null_lambda
null_lambda = lambda t, y, p: 0


# make a function that takes two lambda functions and add em together
def add_functions(f1, f2):
    return lambda t, y, p: f1(t, y, p) + f2(t, y, p)


# make a function that takes a list of lambda functions and difference em together
def difference_functions(f1, f2):
    return lambda t, y, p: f1(t, y, p) - f2(t, y, p)
