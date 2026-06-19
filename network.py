"""Substrate description for coupled-learning networks: pluggable constitutive laws,
a Network dataclass, and builders. The trainable per-edge parameter p maps to a
conductance g via the constitutive law (resistor: g=p; Poiseuille: g=C*p**4)."""
import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Optional


@dataclass(frozen=True)
class Law:
    name: str
    g_of_p: Callable      # conductance from trainable parameter
    dg_dp: Callable       # d(conductance)/d(parameter), for the coupled-learning chain rule


RESISTOR = Law(
    "resistor",
    lambda p: np.asarray(p, float),
    lambda p: np.ones_like(np.asarray(p, float)),
)


def poiseuille_law(C=1.0):
    """Hagen-Poiseuille: parameter p = radius, conductance g = C * r**4."""
    return Law(
        "poiseuille",
        lambda p: C * np.asarray(p, float) ** 4,
        lambda p: 4.0 * C * np.asarray(p, float) ** 3,
    )


@dataclass
class Network:
    n_nodes: int
    edges: List[tuple]            # list of (a, b)
    inputs: List[int]
    outputs: List[int]
    ground: Optional[int] = None


def grid_network(rows, cols, n_in=1, n_out=1):
    """2-D grid graph. Inputs = top n_in of the left column; outputs = bottom n_out of the
    right column; ground = bottom-left node. Requires n_in <= rows, n_out <= rows."""
    idx = lambda r, c: r * cols + c
    edges = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                edges.append((idx(r, c), idx(r, c + 1)))
            if r + 1 < rows:
                edges.append((idx(r, c), idx(r + 1, c)))
    inputs = [idx(r, 0) for r in range(n_in)]
    outputs = [idx(rows - 1 - j, cols - 1) for j in range(n_out)]
    ground = idx(rows - 1, 0)
    return Network(rows * cols, edges, inputs, outputs, ground)


def conductance_edges(net, g):
    """Solver edges [(a, b, g_e)] from a Network and per-edge conductances g (len = #edges)."""
    return [(a, b, float(ge)) for (a, b), ge in zip(net.edges, np.asarray(g, float))]
