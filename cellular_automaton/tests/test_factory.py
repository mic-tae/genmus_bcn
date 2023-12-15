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


from .context import cellular_automaton as ca


class TAutomaton(ca.CellularAutomaton):
    def evolve_rule(self, last_cell_state, neighbors_last_states):
        return last_cell_state

    def init_cell_state(self, cell_coordinate):
        return cell_coordinate


def test_ca_has_correct_values():
    ca_ = TAutomaton(ca.MooreNeighborhood(), [3, 3])
    assert tuple(c.state for c in ca_.cells.values()) == ((0, 0), (0, 1), (0, 2),
                                                          (1, 0), (1, 1), (1, 2),
                                                          (2, 0), (2, 1), (2, 2))
