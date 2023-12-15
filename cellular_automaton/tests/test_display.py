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
# pylint: disable=redefined-outer-name
# pylint: disable=unused-argument

from unittest.mock import MagicMock

import sys
import pytest
from .context import cellular_automaton as ca


def import_mock(module):
    def argument_wrapper(func):
        def function_wrapper(automaton, *args, **kwargs):
            try:
                module_ = sys.modules[module]
            except KeyError:
                module_ = ""
            sys.modules[module] = MagicMock()
            return_value = func(automaton, pygame_mock=sys.modules[module], *args, **kwargs)

            if module_ != "":
                sys.modules[module] = module_
            else:
                del sys.modules[module]
            return return_value

        return function_wrapper

    return argument_wrapper


class TAutomaton(ca.CellularAutomaton):
    def init_cell_state(self, cell_coordinate):
        return [1] if cell_coordinate == (1, 1) else [0]

    def evolve_rule(self, last_cell_state, neighbors_last_states):
        ns = last_cell_state[:]
        if 0 < last_cell_state[0] < 40:
            ns[0] += 1
        return ns


@pytest.fixture
def automaton():
    return TAutomaton(ca.MooreNeighborhood(), (3, 3))


@import_mock(module='pygame')
def test_evolution_steps_per_draw(automaton, pygame_mock):
    ca.CAWindow(cellular_automaton=automaton, window_size=(10, 10)).run(evolutions_per_draw=10, last_evolution_step=1)
    assert automaton.evolution_step == 10


@import_mock(module='pygame')
def test_updated_rectangle_calls(automaton, pygame_mock):
    ca.CAWindow(cellular_automaton=automaton, window_size=(10, 10)).run(last_evolution_step=4)
    assert pygame_mock.display.update.call_count == 4 * (3 + 1)  # steps * (texts + changed cells)

@import_mock(module='pygame')
def test_ends_when_ca_is_done(automaton, pygame_mock):
    automaton.evolve(39)
    assert automaton.active == True
    assert automaton.evolution_step == 39
    ca.CAWindow(cellular_automaton=automaton, window_size=(10, 10)).run(last_evolution_step=45)
    assert automaton.active == False
    assert automaton.evolution_step == 40
