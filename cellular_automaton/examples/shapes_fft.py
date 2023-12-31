#!/usr/bin/env python3
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

# pylint: disable=wrong-import-position
# pylint: disable=missing-function-docstring

import numpy as np
import random
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cellular_automaton import CellularAutomaton, HexagonalNeighborhood, CAWindow, EdgeRule

ALIVE = [1.0]
DEAD = [0]


class ShapesFFT(CellularAutomaton):
    """ Cellular automaton with the evolution rules of conways game of life """

    def __init__(self):
        super().__init__(dimension=[100, 100],
                         neighborhood=HexagonalNeighborhood(EdgeRule.FIRST_AND_LAST_CELL_OF_DIMENSION_ARE_NEIGHBORS))

    # this gets called "from" a coordinate, so: defines the state of that coordinate
    def init_cell_state(self, coord):  # pylint: disable=no-self-use
        init = self.init_array[coord[0], coord[1]]
        return [init]

        
    def evolve_rule(self, last_cell_state, neighbors_last_states):
        new_cell_state = last_cell_state
        alive_neighbours = self.__count_alive_neighbours(neighbors_last_states)

        ### B3/S23
        if last_cell_state == DEAD and alive_neighbours == 2:
            new_cell_state = ALIVE
        elif last_cell_state == ALIVE and alive_neighbours not in [2, 3, 5]:
            new_cell_state = DEAD


        """
        ### B2/S34
        if last_cell_state == DEAD and alive_neighbours == 2:
            new_cell_state = ALIVE
        elif last_cell_state == ALIVE and alive_neighbours not in [3, 4]:
            new_cell_state = DEAD
        """
        
        """
        ### B3678/S23
        if last_cell_state == DEAD and alive_neighbours in [3, 6, 7, 8]:
            new_cell_state = ALIVE
        elif last_cell_state == ALIVE and alive_neighbours in [2, 3]:
            new_cell_state = DEAD
        """
        
        """
        ### B35678/S234
        if last_cell_state == DEAD and alive_neighbours in [3, 5, 6, 7, 8]:
            new_cell_state = ALIVE
        elif last_cell_state == ALIVE and alive_neighbours in [2, 3, 4]:
            new_cell_state = DEAD
        """

        """
        if last_cell_state == DEAD and alive_neighbours in [3, 4]:
            new_cell_state = ALIVE
        elif last_cell_state == ALIVE and alive_neighbours in [2, 3, 5]:
            new_cell_state = DEAD
        """

        return new_cell_state

    @staticmethod
    def __count_alive_neighbours(neighbours):
        alive_neighbors = []
        for n in neighbours:
            if n == ALIVE:
                alive_neighbors.append(1)
        return len(alive_neighbors)


def main():
    CAWindow(cellular_automaton=ShapesFFT(),
             window_size=(1000, 830)).run(evolutions_per_second=15)


if __name__ == "__main__":
    main()    
