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

# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name


import pytest
from .context import cellular_automaton as ca


def check_neighbors(neighborhood, neighborhood_sets, dimension=(3, 3)):
    for neighborhood_set in neighborhood_sets:
        neighbors = neighborhood.calculate_cell_neighbor_coordinates(neighborhood_set(0), dimension)
        if neighborhood_set(1) != neighbors:
            print("\nWrong neighbors (expected, real): ", (neighborhood_set(1)), neighbors)
            return False
    return True


@pytest.mark.parametrize(('coordinate', 'expected_neighborhood'),
                         (((0, 0), ((1, 0), (0, 1), (1, 1))),
                          ((0, 1), ((0, 0), (1, 0), (1, 1), (0, 2), (1, 2))),
                          ((1, 1), ((0, 0), (1, 0), (2, 0), (0, 1), (2, 1), (0, 2), (1, 2), (2, 2))),
                          ((2, 2), ((1, 1), (2, 1), (1, 2)))))
def test_ignore_missing_neighbors(coordinate, expected_neighborhood):
    neighborhood = ca.MooreNeighborhood(ca.EdgeRule.IGNORE_MISSING_NEIGHBORS_OF_EDGE_CELLS)
    actual_neighborhood = neighborhood.calculate_cell_neighbor_coordinates(coordinate, (3, 3))
    assert actual_neighborhood == expected_neighborhood


@pytest.mark.parametrize(('coordinate', 'expected_neighborhood'),
                         (((0, 0), ()),
                          ((0, 1), ()),
                          ((2, 0), ()),
                          ((1, 1), ((0, 0), (1, 0), (2, 0), (0, 1), (2, 1), (0, 2), (1, 2), (2, 2))),
                          ((2, 2), ())))
def test_ignore_edge_cells(coordinate, expected_neighborhood):
    neighborhood = ca.MooreNeighborhood()
    actual_neighborhood = neighborhood.calculate_cell_neighbor_coordinates(coordinate, (3, 3))
    assert actual_neighborhood == expected_neighborhood


@pytest.mark.parametrize(('coordinate', 'expected_neighborhood'),
                         (((0, 0), ((2, 2), (0, 2), (1, 2), (2, 0), (1, 0), (2, 1), (0, 1), (1, 1))),
                          ((1, 1), ((0, 0), (1, 0), (2, 0), (0, 1), (2, 1), (0, 2), (1, 2), (2, 2))),
                          ((2, 2), ((1, 1), (2, 1), (0, 1), (1, 2), (0, 2), (1, 0), (2, 0), (0, 0)))))
def test_cyclic_dimensions(coordinate, expected_neighborhood):
    neighborhood = ca.MooreNeighborhood(ca.EdgeRule.FIRST_AND_LAST_CELL_OF_DIMENSION_ARE_NEIGHBORS)
    actual_neighborhood = neighborhood.calculate_cell_neighbor_coordinates(coordinate, (3, 3))
    assert actual_neighborhood == expected_neighborhood


def test_von_neumann_r1():
    neighborhood = ca.VonNeumannNeighborhood(ca.EdgeRule.FIRST_AND_LAST_CELL_OF_DIMENSION_ARE_NEIGHBORS)
    actual_neighborhood = neighborhood.calculate_cell_neighbor_coordinates((1, 1), (3, 3))
    assert actual_neighborhood == ((1, 0), (0, 1), (2, 1), (1, 2))


def test_von_neumann_r2():
    neighborhood = ca.VonNeumannNeighborhood(ca.EdgeRule.FIRST_AND_LAST_CELL_OF_DIMENSION_ARE_NEIGHBORS, radius=2)
    actual_neighborhood = neighborhood.calculate_cell_neighbor_coordinates((2, 2), (5, 5))
    assert actual_neighborhood == ((2, 0), (1, 1), (2, 1), (3, 1), (0, 2), (1, 2),
                                   (3, 2), (4, 2), (1, 3), (2, 3), (3, 3), (2, 4))


def test_von_neumann_d3():
    neighborhood = ca.VonNeumannNeighborhood(ca.EdgeRule.FIRST_AND_LAST_CELL_OF_DIMENSION_ARE_NEIGHBORS)
    actual_neighborhood = neighborhood.calculate_cell_neighbor_coordinates((1, 1, 1), (3, 3, 3))
    assert actual_neighborhood == ((1, 1, 0), (1, 0, 1), (0, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2))


def test_radial():
    neighborhood = ca.RadialNeighborhood(radius=2)
    actual_neighborhood = neighborhood.calculate_cell_neighbor_coordinates((2, 2), (5, 5))
    assert actual_neighborhood == ((1, 0), (2, 0), (3, 0),
                                   (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
                                   (0, 2), (1, 2), (3, 2), (4, 2),
                                   (0, 3), (1, 3), (2, 3), (3, 3), (4, 3),
                                   (1, 4), (2, 4), (3, 4))


def test_radial_neighbor_coords():
    neighborhood = ca.RadialNeighborhood(edge_rule=ca.EdgeRule.FIRST_AND_LAST_CELL_OF_DIMENSION_ARE_NEIGHBORS, radius=2)
    neighbor_coords = neighborhood.calculate_cell_neighbor_coordinates((0, 0), (10, 10))
    assert neighbor_coords == ((9, 8), (0, 8), (1, 8),
                               (8, 9), (9, 9), (0, 9), (1, 9), (2, 9),
                               (8, 0), (9, 0), (1, 0), (2, 0),
                               (8, 1), (9, 1), (0, 1), (1, 1), (2, 1),
                               (9, 2), (0, 2), (1, 2))


def test_radial_neighbor_coords():
    neighborhood = ca.RadialNeighborhood(edge_rule=ca.EdgeRule.FIRST_AND_LAST_CELL_OF_DIMENSION_ARE_NEIGHBORS, radius=2)
    neighbor_coords = neighborhood.calculate_cell_neighbor_coordinates((1, 1), (10, 10))
    assert neighbor_coords == ((0, 9), (1, 9), (2, 9),
                               (9, 0), (0, 0), (1, 0), (2, 0), (3, 0),
                               (9, 1), (0, 1), (2, 1), (3, 1),
                               (9, 2), (0, 2), (1, 2), (2, 2), (3, 2),
                               (0, 3), (1, 3), (2, 3))


@pytest.mark.parametrize(('coordinate', 'expected_neighborhood'),
                         (((2, 2), ((1, 0), (2, 0), (3, 0),
                                    (0, 1), (1, 1), (2, 1), (3, 1),
                                    (0, 2), (1, 2), (3, 2), (4, 2),
                                    (0, 3), (1, 3), (2, 3), (3, 3),
                                    (1, 4), (2, 4), (3, 4))),
                          ((2, 3), ((1, 1), (2, 1), (3, 1),
                                    (1, 2), (2, 2), (3, 2), (4, 2),
                                    (0, 3), (1, 3), (3, 3), (4, 3),
                                    (1, 4), (2, 4), (3, 4), (4, 4),
                                    (1, 5), (2, 5), (3, 5)))))
def test_hexagonal(coordinate, expected_neighborhood):
    neighborhood = ca.HexagonalNeighborhood(radius=2)
    actual_neighborhood = neighborhood.calculate_cell_neighbor_coordinates(coordinate, (6, 6))
    assert actual_neighborhood == expected_neighborhood


@pytest.mark.parametrize(('coordinate', 'cid'),
                         (((-1, -1), 0),
                          ((0, -1), 1),
                          ((1, -1), 2),
                          ((-1, 0), 3),
                          ((1, 0), 4),
                          ((-1, 1), 5),
                          ((0, 1), 6),
                          ((1, 1), 7)))
def test_get_neighbour_by_relative(coordinate, cid):
    neighborhood = ca.MooreNeighborhood()
    neighborhood.calculate_cell_neighbor_coordinates((0, 0), [3, 3])
    assert neighborhood.get_neighbor_by_relative_coordinate(list(range(8)), coordinate) == cid


@pytest.mark.parametrize("dimensions", (1, 2, 3, 4, 5))
def test_get_relative_11_neighbor_of_coordinate_11(dimensions):
    neighborhood = ca.MooreNeighborhood()
    neighbor = neighborhood.get_neighbor_by_relative_coordinate(
        neighborhood.calculate_cell_neighbor_coordinates((1,)*dimensions, (3,)*dimensions),
        (1,)*dimensions)
    assert neighbor == (2, )*dimensions


@pytest.mark.parametrize(
    ("dimension", "expected"),
    ((1, ((0, ), (2, ))),
     (2, ((0, 0), (1, 0), (2, 0),
          (0, 1), (2, 1),
          (0, 2), (1, 2), (2, 2))),
     (3, ((0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 0), (1, 1, 0), (2, 1, 0), (0, 2, 0), (1, 2, 0), (2, 2, 0),
          (0, 0, 1), (1, 0, 1), (2, 0, 1), (0, 1, 1), (2, 1, 1), (0, 2, 1), (1, 2, 1), (2, 2, 1),
          (0, 0, 2), (1, 0, 2), (2, 0, 2), (0, 1, 2), (1, 1, 2), (2, 1, 2), (0, 2, 2), (1, 2, 2), (2, 2, 2)))))
def test_get_neighbor_coordinates(dimension, expected):
    n = ca.MooreNeighborhood(edge_rule=ca.EdgeRule.FIRST_AND_LAST_CELL_OF_DIMENSION_ARE_NEIGHBORS)
    assert n.calculate_cell_neighbor_coordinates((1,) * dimension, (3,) * dimension) == expected
