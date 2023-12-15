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

import pstats
import random
import tempfile
import cProfile
import contextlib
import sys
import os

from typing import Sequence

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cellular_automaton import CellularAutomaton, MooreNeighborhood, EdgeRule


class StarFallAutomaton(CellularAutomaton):
    """ A basic cellular automaton that just copies one neighbour state so get some motion in the grid. """

    def __init__(self):
        super().__init__(dimension=[20, 20, 10, 10],
                         neighborhood=MooreNeighborhood(EdgeRule.FIRST_AND_LAST_CELL_OF_DIMENSION_ARE_NEIGHBORS))

    def init_cell_state(self, __) -> Sequence:
        rand = random.randrange(0, 101, 1)
        init = max(.0, float(rand - 99))
        return [init * random.randint(0, 3)]

    def evolve_rule(self, __, neighbors_last_states: Sequence) -> Sequence:
        return self._neighborhood.get_neighbor_by_relative_coordinate(neighbors_last_states, (-1, -1, -1, -1))


def profile(code):
    with tempfile.NamedTemporaryFile() as temp_file:
        cProfile.run(code, filename=temp_file.name, sort=True)
        profile_stats = pstats.Stats(temp_file.name)
        profile_stats.sort_stats("tottime").print_stats(20)


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        print("=== CREATION ===")
        profile('ca = StarFallAutomaton()')
        print("=== COMPUTATION ===")
        profile('ca.evolve(times=10)')
