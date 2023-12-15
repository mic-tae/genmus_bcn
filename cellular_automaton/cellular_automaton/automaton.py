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

from typing import Sequence

import abc
import itertools
import recordclass

from cellular_automaton import Neighborhood


CELL = recordclass.make_dataclass("Cell",
                                  ("state", "is_active", "is_dirty", "neighbors"),
                                  defaults=((0, ), True, True, (None, )))


class CellularAutomatonCreator(abc.ABC):
    """ Creates a cellular automaton from a dimension and a neighborhood definition """

    def __init__(self,
                 dimension,
                 neighborhood: Neighborhood,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dimension = dimension
        self._neighborhood = neighborhood

        self._current_state = {}
        self._next_state = {}
        self.__make_cellular_automaton_state()

    def get_dimension(self):
        return self._dimension

    dimension = property(get_dimension)

    def __make_cellular_automaton_state(self):
        self.__make_cells()
        self.__add_neighbors()

    def __make_cells(self):
        for coord in itertools.product(*[range(d) for d in self._dimension]):
            cell_state = self.init_cell_state(coord)
            self._current_state[coord] = CELL(cell_state)
            self._next_state[coord] = CELL(cell_state)

    def __add_neighbors(self):
        calculate_neighbor_coordinates = self._neighborhood.calculate_cell_neighbor_coordinates
        coordinates = self._current_state.keys()
        for coordinate, cell_c, cell_n in zip(coordinates, self._current_state.values(), self._next_state.values()):
            n_coord = calculate_neighbor_coordinates(coordinate, self._dimension)
            cell_c.neighbors = list([self._current_state[nc] for nc in n_coord])
            cell_n.neighbors = list([self._next_state[nc] for nc in n_coord])

    def init_cell_state(self, cell_coordinate: Sequence) -> Sequence:  # pragma: no cover
        """ Will be called to initialize a cells state.
        :param cell_coordinate: Cells coordinate.
        :return: Iterable that represents the initial cell state
        """
        raise NotImplementedError


class CellularAutomaton(CellularAutomatonCreator, abc.ABC):
    """ This class represents a cellular automaton.
    It can be created with n dimensions and can handle different neighborhood definitions.

    It is intended to be uses as base class.
    Override `init_cell_state()` to define the state the cell(s) are initiated with.
    Override `evolve()` to define the rule that is aplied on every evolution step of this automaton.
    """
    def __init__(self, neighborhood: Neighborhood, *args, **kwargs):
        """ Initiates a cellular automaton by the use of the `init_cell_state` method.
        :param neighborhood: Defines which cells are considered neighbors.
        :param dimension: Iterable of len = dimensions
                          (e.g. [4, 3, 3, 3] = 4 x 3 x 3 x 3 cells in a four dimensional cube).
        """
        super().__init__(neighborhood=neighborhood, *args, **kwargs)
        self._evolution_step = 0
        self._active = True

    def is_active(self):
        return self._active

    def reactivate(self):
        """ Sets all cells active again """
        for cell in self._current_state.values():
            cell.is_active = True
            cell.is_dirty = True
        self._active = True

    active = property(is_active)

    def get_cells(self):
        return self._current_state

    def set_cells(self, cells):
        """ Sets the cell states both as current and next states """
        for (coordinate, c_cell), n_cell in zip(self._current_state.items(), self._next_state.values()):
            new_cell_state = cells[coordinate].state
            c_cell.state = new_cell_state
            n_cell.state = new_cell_state

    cells = property(get_cells, set_cells)

    def get_evolution_step(self):
        return self._evolution_step

    evolution_step = property(get_evolution_step)

    def evolve(self, times=1):
        """ Evolve all cells x times.
        :param times: The number of evolution steps processed with one call of this method.
        """
        for _ in itertools.repeat(None, times):
            self._active = False
            self.__evolve_cells(self._current_state, self._next_state)
            self._current_state, self._next_state = self._next_state, self._current_state
            self._evolution_step += 1

    def __evolve_cells(self, this_state, next_state):
        evolve_cell = self.__evolve_cell
        evolution_rule = self.evolve_rule
        for old, new in zip(this_state.values(), next_state.values()):
            if old.is_active:
                new_state = evolution_rule(old.state.copy(), [n.state for n in old.neighbors])
                old.is_active = False
                evolve_cell(old, new, new_state)

    def __evolve_cell(self, old, cell, new_state):
        changed = new_state != old.state
        cell.state = new_state
        cell.is_dirty |= changed
        old.is_dirty |= changed
        self._active |= changed
        if changed:
            cell.is_active = True
            for n in cell.neighbors:
                n.is_active = True

    def evolve_rule(self, last_cell_state: Sequence, neighbors_last_states: Sequence) -> Sequence:  # pragma: no cover
        """ Calculates and sets new state of 'cell'.
        A cells evolution will only be called if it or at least one of its neighbors has changed last evolution_step.
        :param last_cell_state:         The cells state previous to the evolution step.
        :param neighbors_last_states:   The cells neighbors current states.
        :return: New state.             The state after this evolution step
        """
        raise NotImplementedError
