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
# pylint: disable=no-self-use

import random
import sys
import os
from typing import Sequence

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cellular_automaton import CellularAutomaton, MooreNeighborhood, CAWindow, EdgeRule


class StarFallAutomaton(CellularAutomaton):
    """ Represents an automaton dropping colorful stars """

    def __init__(self):
        super().__init__(dimension=[200, 200],
                         neighborhood=MooreNeighborhood(EdgeRule.FIRST_AND_LAST_CELL_OF_DIMENSION_ARE_NEIGHBORS))

    def init_cell_state(self, __) -> Sequence:
        rand = random.randrange(0, 101, 1)
        init = max(.0, float(rand - 99))
        return [init * random.randint(0, 3)]

    def evolve_rule(self, __, neighbors_last_states: Sequence) -> Sequence:
        return self._neighborhood.get_neighbor_by_relative_coordinate(neighbors_last_states, (-1, -1))


def state_to_color(current_state: Sequence) -> Sequence:
    return 255 if current_state[0] == 1 else 0, \
           255 if current_state[0] == 2 else 0, \
           255 if current_state[0] == 3 else 0


if __name__ == "__main__":
    CAWindow(cellular_automaton=StarFallAutomaton(),
             window_size=(1000, 830),
             state_to_color_cb=state_to_color).run()
