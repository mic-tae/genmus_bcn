"""
Copyright 2019 Richard Feistenauer

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import enum
import operator
import itertools
import math


class EdgeRule(enum.Enum):
    """ Enum for different possibilities to handle the edge of the automaton. """
    IGNORE_EDGE_CELLS = 0
    IGNORE_MISSING_NEIGHBORS_OF_EDGE_CELLS = 1
    FIRST_AND_LAST_CELL_OF_DIMENSION_ARE_NEIGHBORS = 2


class Neighborhood:
    """ Defines which cells should be considered to be neighbors during evolution of cellular automaton."""

    def __init__(self, edge_rule=EdgeRule.IGNORE_EDGE_CELLS, radius=1):
        """ General class for all Neighborhoods.
        :param edge_rule:   Rule to define, how cells on the edge of the grid will be handled.
        :param radius:      If radius > 1 it grows the neighborhood
                            by adding the neighbors of the neighbors radius times.
        """
        self._rel_neighbors = None
        self._grid_dimensions = []
        self._radius = radius
        self.__edge_rule = edge_rule

    def calculate_cell_neighbor_coordinates(self, cell_coordinate, grid_dimensions):
        """ Get a list of absolute coordinates for the cell neighbors.
            The EdgeRule can reduce the returned neighbor count.
        :param cell_coordinate:  The coordinate of the cell.
        :param grid_dimensions:  The dimensions of the grid, to apply the edge the rule.
        :return: list of absolute coordinates for the cells neighbors.
        """
        self.__lazy_initialize_relative_neighborhood(grid_dimensions)
        return tuple(self._neighbors_generator(cell_coordinate))

    def __lazy_initialize_relative_neighborhood(self, grid_dimensions):
        self._grid_dimensions = grid_dimensions
        if self._rel_neighbors is None:
            self._create_relative_neighborhood()

    def _create_relative_neighborhood(self):
        self._rel_neighbors = tuple(self._neighborhood_generator())

    def _neighborhood_generator(self):
        for coordinate in itertools.product(range(-self._radius, self._radius + 1), repeat=len(self._grid_dimensions)):
            if self._neighbor_rule(coordinate) and coordinate != (0, ) * len(self._grid_dimensions):
                yield tuple(reversed(coordinate))

    def _neighbor_rule(self, rel_neighbor):  # pylint: disable=no-self-use, unused-argument
        return True

    def get_neighbor_by_relative_coordinate(self, neighbors, rel_coordinate):
        return neighbors[self._rel_neighbors.index(rel_coordinate)]

    def _neighbors_generator(self, cell_coordinate):
        on_edge = self.__is_coordinate_on_an_edge(cell_coordinate)
        if self.__edge_rule != EdgeRule.IGNORE_EDGE_CELLS or not on_edge:  # pylint: disable=too-many-nested-blocks
            for rel_n in self._rel_neighbors:
                if on_edge:
                    n, n_folded = zip(*[(ni + ci, (ni + di + ci) % di)
                                        for ci, ni, di in zip(cell_coordinate, rel_n, self._grid_dimensions)])
                    if self.__edge_rule == EdgeRule.FIRST_AND_LAST_CELL_OF_DIMENSION_ARE_NEIGHBORS or n == n_folded:
                        yield n_folded
                else:
                    yield tuple(map(operator.add, rel_n, cell_coordinate))

    def __is_coordinate_on_an_edge(self, coordinate):
        return any(not(self._radius-1 < ci < di-self._radius) for ci, di in zip(coordinate, self._grid_dimensions))


class MooreNeighborhood(Neighborhood):
    """ Moore defined a neighborhood with a radius applied on a the non euclidean distance to other cells in the grid.
        Example:
            2 dimensions
            C = cell of interest
            N = neighbor of cell
            X = no neighbor of cell

                  Radius 1                     Radius 2
               X  X  X  X  X                N  N  N  N  N
               X  N  N  N  X                N  N  N  N  N
               X  N  C  N  X                N  N  C  N  N
               X  N  N  N  X                N  N  N  N  N
               X  X  X  X  X                N  N  N  N  N
    """


class VonNeumannNeighborhood(Neighborhood):
    """ Von Neumann defined a neighborhood with a radius applied to Manhatten distance
        (steps between cells without diagonal movement).
        Example:
            2 dimensions
            C = cell of interest
            N = neighbor of cell
            X = no neighbor of cell

                  Radius 1                     Radius 2
               X  X  X  X  X                X  X  N  X  X
               X  X  N  X  X                X  N  N  N  X
               X  N  C  N  X                N  N  C  N  N
               X  X  N  X  X                X  N  N  N  X
               X  X  X  X  X                X  X  N  X  X
    """

    def _neighbor_rule(self, rel_neighbor):
        cross_sum = 0
        for coordinate_i in rel_neighbor:
            cross_sum += abs(coordinate_i)
        return cross_sum <= self._radius


class RadialNeighborhood(Neighborhood):
    """ Neighborhood with a radius applied to euclidean distance + delta

        Example:
            2 dimensions
            C = cell of interest
            N = neighbor of cell
            X = no neighbor of cell

                  Radius 2                     Radius 3
            X  X  X  X  X  X  X          X  X  N  N  N  X  X
            X  X  N  N  N  X  X          X  N  N  N  N  N  X
            X  N  N  N  N  N  X          N  N  N  N  N  N  N
            X  N  N  C  N  N  X          N  N  N  C  N  N  N
            X  N  N  N  N  N  X          N  N  N  N  N  N  N
            X  X  N  N  N  X  X          X  N  N  N  N  N  X
            X  X  X  X  X  X  X          X  X  N  N  N  X  X
    """

    def __init__(self, *args, delta_=.25, **kwargs):
        self.delta = delta_
        super().__init__(*args, **kwargs)

    def _neighbor_rule(self, rel_neighbor):
        cross_sum = 0
        for coordinate_i in rel_neighbor:
            cross_sum += pow(coordinate_i, 2)
        return math.sqrt(cross_sum) <= self._radius + self.delta


class HexagonalNeighborhood(Neighborhood):
    """ Defines a Hexagonal neighborhood in a rectangular two dimensional grid:

        Example:
            Von Nexagonal neighborhood in 2 dimensions with radius 1 and 2
            C = cell of interest
            N = neighbor of cell
            X = no neighbor of cell

                  Radius 1                     Radius 2
               X   X   X   X   X           X   N   N   N   X
                 X   N   N   X               N   N   N   N
               X   N   C   N   X           N   N   C   N   N
                 X   N   N   X               N   N   N   N
               X   X   X   X   X           X   N   N   N   X


        Rectangular representation: Radius 1

          Row % 2 == 0            Row % 2 == 1
            N  N  X                 X  N  N
            N  C  N                 N  C  N
            N  N  X                 X  N  N

        Rectangular representation: Radius 2
          Row % 2 == 0            Row % 2 == 1
          X  N  N  N  X           X  N  N  N  X
          N  N  N  N  X           X  N  N  N  N
          N  N  C  N  N           N  N  C  N  N
          N  N  N  N  X           X  N  N  N  N
          X  N  N  N  X           X  N  N  N  X
    """

    def __init__(self, *args, radius=1, **kwargs):
        super().__init__(radius=radius, *args, **kwargs)
        self.__calculate_hexagonal_neighborhood(radius)

    def __calculate_hexagonal_neighborhood(self, radius):
        neighbor_lists = [[(0, 0)], [(0, 0)]]
        for radius_i in range(1, radius + 1):
            for i, neighbor in enumerate(neighbor_lists):
                neighbor = _grow_neighbours(neighbor)
                neighbor = self.__add_rectangular_neighbours(neighbor, radius_i, i % 2 == 1)
                neighbor = sorted(neighbor, key=(lambda ne: [ne[1], ne[0]]))
                neighbor.remove((0, 0))
                neighbor_lists[i] = neighbor
        self._neighbor_lists = neighbor_lists

    def get_neighbor_by_relative_coordinate(self, neighbors, rel_coordinate):  # pragma: no cover
        raise NotImplementedError

    def calculate_cell_neighbor_coordinates(self, cell_coordinate, grid_dimensions):
        self._rel_neighbors = self._neighbor_lists[cell_coordinate[1] % 2]
        return super().calculate_cell_neighbor_coordinates(cell_coordinate, grid_dimensions)

    @staticmethod
    def __add_rectangular_neighbours(neighbours, radius, is_odd):
        new_neighbours = []
        for x in range(0, radius + 1):
            if is_odd:
                x -= int(radius/2)
            else:
                x -= int((radius + 1) / 2)

            for y in range(-radius, radius + 1):
                new_neighbours.append((x, y))
        new_neighbours.extend(neighbours)
        return list(set(new_neighbours))


def _grow_neighbours(neighbours):
    new_neighbours = neighbours[:]
    for n in neighbours:
        new_neighbours.append((n[0], n[1] - 1))
        new_neighbours.append((n[0] - 1, n[1]))
        new_neighbours.append((n[0] + 1, n[1]))
        new_neighbours.append((n[0], n[1] + 1))
    return list(set(new_neighbours))
