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

# pylint: disable=all

import datetime
import time
import operator
import collections
import contextlib
from typing import Sequence

from . import CellularAutomaton

_Rect = collections.namedtuple(typename="Rect",
                               field_names=["left", "top", "width", "height"])


class PygameEngine:
    """ This is an wrapper for the pygame engine.
        By initializing pygame lazy the dependency can be dropped.
    """

    def __init__(self, window_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import pygame
        self._pygame = pygame
        self._pygame.init()
        pygame.display.set_caption("Cellular Automaton")
        self.__screen = pygame.display.set_mode(window_size)
        self.__font = pygame.font.SysFont("monospace", 15)

        self._width = window_size[0]
        self._height = window_size[1]

    def write_text(self, pos, text, color=(0, 255, 0)):
        label = self.__font.render(text, True, color)
        update_rect = self.__screen.blit(label, pos)
        self.update_rectangles(update_rect)

    def fill_surface_with_color(self, rect, color=(0, 0, 0)):
        return self.__screen.fill(color, rect)

    def update_rectangles(self, rectangles):
        self._pygame.display.update(rectangles)

    def is_active(self):  # pragma: no cover
        for event in self._pygame.event.get():
            if event.type == self._pygame.QUIT:
                return False
        return True


class CAWindow:
    def __init__(self,
                 cellular_automaton: CellularAutomaton,
                 window_size=(1000, 800),
                 stretch_cells=False,
                 draw_engine=None,
                 state_to_color_cb=None,
                 *args, **kwargs):
        """
        Creates a window to render a 2D CellularAutomaton.
        :param cellular_automaton:  The automaton to display and evolve
        :param window_size:         The Window size (default: 1000 x 800)
        :param stretch_cells:       Stretches cells to fit into window size. (default: false)
                                    Activating it can result in black lines throughout the automaton.
        :param draw_engine:         The draw_engine (default: pygame)
        :param state_to_color_cb:   A callback to define the draw color of CA states (default: red for states != 0)
        """
        super().__init__(*args, **kwargs)
        self._cellular_automaton = cellular_automaton
        self.__rect = _Rect(left=0, top=30, width=window_size[0], height=window_size[1] - 30)
        self.__calculate_cell_display_size(stretch_cells)
        self.__draw_engine = PygameEngine(window_size) if draw_engine is None else draw_engine
        self.__state_to_color = self._get_cell_color if state_to_color_cb is None else state_to_color_cb

    def run(self,
            evolutions_per_second=0,
            evolutions_per_draw=1,
            last_evolution_step=0,):
        """
        Evolves and draws the CellularAutomaton
        :param evolutions_per_second:   0 = as fast as possible | > 0 to slow down the CellularAutomaton
        :param evolutions_per_draw:     Amount of evolutions done before screen gets redrawn.
        :param last_evolution_step:     0 = infinite | > 0 evolution step at which this method will stop
        Warning: is blocking until finished
        """
        with contextlib.suppress(KeyboardInterrupt):
            while self._is_not_user_terminated() and self._not_at_the_end(last_evolution_step):
                time_ca_start = time.time()
                self._cellular_automaton.evolve(evolutions_per_draw)
                time_ca_end = time.time()
                self._redraw_dirty_cells()
                time_ds_end = time.time()
                self.print_process_info(evolve_duration=(time_ca_end - time_ca_start),
                                        draw_duration=(time_ds_end - time_ca_end),
                                        evolution_step=self._cellular_automaton.evolution_step)
                self._sleep_to_keep_rate(time.time() - time_ca_start, evolutions_per_second)

    def _sleep_to_keep_rate(self, time_taken, evolutions_per_second):  # pragma: no cover
        if evolutions_per_second > 0:
            rest_time = 1.0 / evolutions_per_second - time_taken
            if rest_time > 0:
                screenshot_filename = f"screenshot_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png"
                #self.__draw_engine._pygame.image.save( self.__draw_engine._pygame.display.get_surface(), screenshot_filename )
                time.sleep(rest_time)

    def _not_at_the_end(self, last_evolution_step):
        return (self._cellular_automaton.evolution_step < last_evolution_step or last_evolution_step <= 0) \
               and self._cellular_automaton.active

    def __calculate_cell_display_size(self, stretch_cells):  # pragma: no cover
        grid_dimension = self._cellular_automaton.dimension
        if stretch_cells:
            self.__cell_size = [self.__rect.width / grid_dimension[0],
                                self.__rect.height / grid_dimension[1]]
        else:
            self.__cell_size = [int(self.__rect.width / grid_dimension[0]),
                                int(self.__rect.height / grid_dimension[1])]

    def _redraw_dirty_cells(self):
        self.__draw_engine.update_rectangles(list(self.__redraw_dirty_cells()))

    def __redraw_dirty_cells(self):
        for coordinate, cell in self._cellular_automaton.cells.items():
            if cell.is_dirty:
                yield self.__redraw_cell(cell, coordinate)

    def __redraw_cell(self, cell, coordinate):
        cell_color = self.__state_to_color(cell.state)
        cell_pos = self.__calculate_cell_position_in_the_grid(coordinate)
        surface_pos = self.__calculate_cell_position_on_screen(cell_pos)
        cell.is_dirty = False
        return self.__draw_cell_surface(surface_pos, cell_color)

    def _get_cell_color(self, current_state: Sequence) -> Sequence:
        """ Returns the color of the cell depending on its current state """
        return 255 if current_state[0] else 0, 0, 0

    def __calculate_cell_position_in_the_grid(self, coordinate):
        return list(map(operator.mul, self.__cell_size, coordinate))

    def __calculate_cell_position_on_screen(self, cell_pos):
        return [self.__rect.left + cell_pos[0], self.__rect.top + cell_pos[1]]

    def __draw_cell_surface(self, surface_pos, cell_color):
        return self.__draw_engine.fill_surface_with_color((surface_pos, self.__cell_size), cell_color)

    def print_process_info(self, evolve_duration, draw_duration, evolution_step):
        #self.__draw_engine.fill_surface_with_color(((0, 0), (self.__rect.width, 30)))
        #self.__draw_engine.write_text((10, 5), "CA: " + "{0:.4f}".format(evolve_duration) + "s")
        #self.__draw_engine.write_text((310, 5), "Display: " + "{0:.4f}".format(draw_duration) + "s")
        #self.__draw_engine.write_text((660, 5), "Step: " + str(evolution_step))
        0

    def _is_not_user_terminated(self):
        return self.__draw_engine.is_active()
