from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    vals_plus = list(vals)
    vals_minus = list(vals)

    # Adjust the i-th argument by epsilon
    vals_plus[arg] += epsilon
    vals_minus[arg] -= epsilon

    # Compute the central difference approximation
    return (f(*vals_plus) - f(*vals_minus)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None: ...  # noqa: D102

    @property
    def unique_id(self) -> int: ...  # noqa: D102

    def is_leaf(self) -> bool: ...  # noqa: D102

    def is_constant(self) -> bool: ...  # noqa: D102

    @property
    def parents(self) -> Iterable["Variable"]: ...  # noqa: D102

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]: ...  # noqa: D102


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    # TODO: Implement for Task 1.4.
    visited = set()
    topological_order = []

    def dfs(v: Variable) -> None:
        if v.unique_id in visited or v.is_constant():
            return
        visited.add(v)
        for parent in v.parents:
            dfs(parent)
        topological_order.append(v)

    dfs(variable)
    return reversed(topological_order)


def backpropagate(variable: Variable, deriv: Any) -> None:  # noqa: D417
    """Runs backpropagation on the computation graph in order to compute derivatives for the leave nodes.

    Args:
    ----
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    """
    # TODO: Implement for Task 1.4.
    topo_order = topological_sort(variable)
    derivative = {
        v.unique_id: 0 for v in topo_order
    }  # key: unique_id, value: initialize to 0
    derivative[variable.unique_id] = deriv

    for v in topo_order:
        if v.is_leaf():
            v.accumulate_derivative(derivative[v.unique_id])
        else:
            for parent, parent_derivative in v.chain_rule(derivative[v.unique_id]):
                derivative[parent.unique_id] += parent_derivative


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:  # noqa: D102
        return self.saved_values
