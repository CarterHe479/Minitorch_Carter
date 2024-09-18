"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


def mul(x: float, y: float) -> float:
    """Multiply two numbers."""
    return x * y


def id(x: float) -> float:
    """Identity function."""
    return x


def add(x: float, y: float) -> float:
    """Add"""
    return x + y


def neg(x: float) -> float:
    """Negate"""
    return -x


# def lt(x: float, y: float) -> bool:
#    """Less than"""
#    return x < y


def lt(x: float, y: float) -> float:
    """Less than"""
    return 1.0 if x < y else 0.0


# def eq(x: float, y: float) -> bool:
#    """Equal"""
#    return x == y


def eq(x: float, y: float) -> float:
    """Equal"""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Maximum"""
    return x if x > y else y


# def is_close(x: float, y: float) -> bool:
#    """Is close"""
#    return abs(x - y) < 1e-2


def is_close(x: float, y: float) -> float:
    """Is close"""
    return (x - y < 1e-2) and (y - x < 1e-2)


# sigmoid - Calculates the sigmoid function.
def sigmoid(x: float) -> float:
    """Sigmoid"""
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))

    # def relu(x: float) -> float:
    """ReLU"""


#    return max(0, x)


def relu(x: float) -> float:
    """ReLU"""
    return x if x > 0 else 0


# def log(x: float) -> float:
#    """Log"""
#    return math.log(x)

EPS = 1e-6


def log(x: float) -> float:
    """Log"""
    return math.log(x + EPS)


def exp(x: float) -> float:
    """Exponential"""
    return math.exp(x)


# log_back - Computes the derivative of log times a second argument.
# def log_back(x: float, y: float) -> float:
#    """Log back"""
#    return y / x


def log_back(x: float, y: float) -> float:
    """Log back"""
    return y / (x + EPS)


def inv(x: float) -> float:
    """Inverse"""
    return 1.0 / x


# inv_back - Computes the derivative of reciprocal times a second arg
def inv_back(x: float, y: float) -> float:
    """Inverse back"""
    return -y / (x**2)


# relu_back - Computes the derivative of ReLU times a second argument.
def relu_back(x: float, y: float) -> float:
    """ReLU back"""
    return 0 if x <= 0 else y


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.


# map - Higher-order function that applies a given function to each element of an iterable
# def map(fn: Callable[[float], float], xs: Iterable[float]) -> Iterable[float]:
#    """Map"""
#    return [fn(x) for x in xs]


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Higher-order map
    Args:
    fn: Function from one value to one value
    Returns:
    A function that takes a list, applies fn to each element, and returns the list.
    """

    def _map(ls: Iterable[float]) -> Iterable[float]:
        ret = []
        for x in ls:
            ret.append(fn(x))
        return ret

    return _map


# zipWith - Higher-order function that combines elements from two iterables using a given function
# def zipWith(
#    fn: Callable[[float, float], float], xs: Iterable[float], ys: Iterable[float]
# ) -> Iterable[float]:
#    """ZipWith"""
#    return [fn(x, y) for x, y in zip(xs, ys)]


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher-order zipWith
    Args:
    fn: Function from two values to one value
    Returns:
    A function that takes two lists, applies fn to each pair of elements, and returns the list.
    """

    def _zipWith(xs: Iterable[float], ys: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(xs, ys):
            ret.append(fn(x, y))
        return ret

    return _zipWith


# reduce - Higher-order function that reduces an iterable to a single value using a given function
# def reduce(
#    fn: Callable[[float, float], float], xs: Iterable[float], init: float
# ) -> float:
#    """Reduce"""
#    xs = list(xs)  # Convert xs to a list
#    return init if not xs else reduce(fn, xs[1:], fn(init, xs[0]))


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """Higher-order reduce
    Args:
    fn: Function from two values to one value
    Returns:
    A function that takes a list, an initial value, and reduces the list to a single value using fn.
    """

    def _reduce(xs: Iterable[float]) -> float:
        val = start
        for x in xs:
            val = fn(val, x)
        return val

    return _reduce


# negList - Negate all elements in a list using map
# def negList(xs: Iterable[float]) -> Iterable[float]:
#    """NegList"""
#    return map(lambda x: -x, xs)


def negList(xs: Iterable[float]) -> Iterable[float]:
    """NegList"""
    return map(neg)(xs)


# addLists - Add corresponding elements from two lists using zipWith
# def addLists(xs: Iterable[float], ys: Iterable[float]) -> Iterable[float]:
#    """AddLists"""
#    return zipWith(add, xs, ys)


def addLists(xs: Iterable[float], ys: Iterable[float]) -> Iterable[float]:
    """AddLists"""
    return zipWith(add)(xs, ys)


# sum - Sum all elements in a list using reduce
def sum(xs: Iterable[float]) -> float:
    """Sum"""
    return reduce(add, 0.0)(xs)


# prod - Multiply all elements in a list using reduce
def prod(xs: Iterable[float]) -> float:
    """Prod"""
    return reduce(mul, 1.0)(xs)
