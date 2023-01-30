# define a lambda function that takes t, y, p and return 0 and assing it to a variable named null_lambda
null_lambda = lambda t, y, p: 0


# make a function that takes two lambda functions and add em together
def add_functions(f1, f2):
    """
    Takes two lambda functions and adds them together creating a new lambda function
    that is the sum of the two functions passed in as arguments to the function.

    Parameters
    ----------
    f1 : lambda function with signature f1(t, y, p)
        The first lambda function to add.
    f2 : lambda function with signature f2(t, y, p)
        The second lambda function to add.

    Returns
    -------
    lambda function with signature f3(t, y, p) = f1(t, y, p) + f2(t, y, p)
        The sum of the two lambda functions passed in as arguments to the function.
    """
    return lambda t, y, p: f1(t, y, p) + f2(t, y, p)


def difference_functions(f1, f2):
    """
    Takes two lambda functions and subtracts them together creating a new lambda function
    that is the difference of the two functions passed in as arguments to the function.

    Parameters
    ----------
    f1 : lambda function with signature f1(t, y, p)
        The first lambda function to subtract.

    f2 : lambda function with signature f2(t, y, p)
        The second lambda function to subtract.

    Returns
    -------
    lambda function with signature f3(t, y, p) = f1(t, y, p) - f2(t, y, p)
        The difference of the two lambda functions passed in as arguments to the function.

    """
    return lambda t, y, p: f1(t, y, p) - f2(t, y, p)
