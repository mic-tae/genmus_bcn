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
# pylint: disable=missing-class-docstring

import pytest

from .context import cellular_automaton as ca


class TAutomaton(ca.CellularAutomaton):

    def evolve_rule(self, last_cell_state, __):
        return [last_cell_state[0] + 1]

    def init_cell_state(self, __):
        return [0]


NEIGHBORHOOD = ca.MooreNeighborhood(ca.EdgeRule.FIRST_AND_LAST_CELL_OF_DIMENSION_ARE_NEIGHBORS)


def test_process_evolution_steps():
    automaton = TAutomaton(NEIGHBORHOOD, [3, 3])
    automaton.evolve(5)
    assert automaton.evolution_step == 5


def test_process_evolution_calls():
    automaton = TAutomaton(NEIGHBORHOOD, [3, 3])
    automaton.evolve(5)
    assert automaton.cells[(1, 1)].state[0] == 5


@pytest.mark.parametrize("dimensions", [1, 2, 3, 4, 5])
def test_dimensions(dimensions):
    automaton = TAutomaton(ca.MooreNeighborhood(), dimension=[3] * dimensions)
    automaton.evolve()
    assert automaton.cells[(1, ) * dimensions].state[0] == 1


def test_copy_cells():
    automaton = TAutomaton(NEIGHBORHOOD, [3, 3])
    automaton.evolve(5)
    automaton2 = TAutomaton(NEIGHBORHOOD, [3, 3])
    automaton2.cells = automaton.cells
    assert automaton2.cells[(1, 1)].state[0] == 5


def test_automaton_goes_inactive():
    automaton = TAutomaton(NEIGHBORHOOD, [3, 3])
    assert automaton.active

    automaton.evolve_rule = lambda x, y: x
    automaton.evolve()
    assert not automaton.active


def test_reactivation():
    automaton = TAutomaton(NEIGHBORHOOD, [3, 3])

    rule, automaton.evolve_rule = automaton.evolve_rule, lambda x, y: x
    automaton.evolve()
    assert not automaton.active

    automaton.reactivate()
    automaton.evolve_rule = rule
    automaton.evolve()
    assert automaton.active
