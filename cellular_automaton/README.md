# Cellular Automaton
This package provides an cellular automaton for [Python 3](https://www.python.org/)

A cellular automaton defines a grid of cells and a set of rules.
All cells then evolve their state depending on their neighbours state simultaneously.

For further information on cellular automatons consult e.g. [mathworld.wolfram.com](http://mathworld.wolfram.com/CellularAutomaton.html)

## Yet another cellular automaton module?
It is not the first python module to provide a cellular automaton, 
but it is to my best knowledge the first that provides all of the following features:
 - easy to use
 - n dimensional
 - speed optimized
 - documented
 - tested
 
I originally did not plan to write a new cellular automaton module, 
but when searching for one, I just found some that had little or no documentation with an API that really did not fit my requirements
and/or Code that was desperately asking for some refactoring.

So I started to write my own module with the goal to provide an user friendly API
and acceptable documentation. During the implementation I figured, why not just provide 
n dimensional support and with reading Clean Code from Robert C. Martin the urge
to have a clean and tested code with a decent coverage added some more requirements.
The speed optimization and multi process capability was more of challenge for myself.
IMHO the module now reached an acceptable speed, but there is still room for improvements (e.g. with Numba?).

## Installation
This module can be loaded and installed from [pypi](https://pypi.org/project/cellular-automaton/): `pip install cellular-automaton`

## Usage
To start and use the automaton you will have to define four things:
- The neighborhood
- The dimensions of the grid
- The evolution rule
- The initial cell state

`````python
class MyCellularAutomaton(CellularAutomaton):
    def init_cell_state(self, coordinate: tuple) -> Sequence:
        return initial_cell_state

    def evolve_rule(self, last_state: tuple, neighbors_last_states: Sequence) -> Sequence:
        return next_cell_state


neighborhood = MooreNeighborhood(EdgeRule.IGNORE_EDGE_CELLS)
ca = MyCellularAutomaton(dimension=[100, 100],
                         neighborhood=neighborhood)
``````

### Neighbourhood
The Neighborhood defines for a cell neighbours in relative coordinates.
The evolution of a cell will depend solely on those neighbours.
 
The Edge Rule passed as parameter to the Neighborhood defines, how cells on the edge of the grid will be handled.
There are three options:
- Ignore edge cells: Edge cells will have no neighbours and thus not evolve.
- Ignore missing neighbours: Edge cells will add the neighbours that exist. This results in varying count of neighbours on edge cells.
- First and last cell of each dimension are neighbours: All cells will have the same neighbour count and no edge exists.

### Dimension
A list or Tuple which states each dimensions size.
The example above defines a two dimensional grid with 100 x 100 cells.

There is no limitation in how many dimensions you choose but your memory and processor power.

### Evolution and Initial State
To define the evolution rule and the initial state create a class inheriting from `CellularAutomaton`.
- The `init_cell_state` method will be called once during the creation process for every cell.  
It will get the coordinate of that cell and is supposed to return a tuple representing that cells state.
- The `evolve_rule` gets passed the last cell state and the states of all neighbors.  
It is supposed to return a tuple representing the new cell state.  
All new states will be applied simultaneously, so the order of processing the cells is irrelevant.

## Visualisation
The package provides a module for visualization of a 2D automaton in a pygame window.

```
CAWindow(cellular_automaton=StarFallAutomaton()).run()
```

The visual part of this module is fully decoupled and thus should be easily replaceable.

## Examples
The package contains three examples:
- [simple_star_fall](https://gitlab.com/DamKoVosh/cellular_automaton/-/tree/master/examples/simple_star_fall.py)
- [conways_game_of_life](https://gitlab.com/DamKoVosh/cellular_automaton/-/tree/master/examples/conways_game_of_life.py)
- [creation_and_process_time_analysis](https://gitlab.com/DamKoVosh/cellular_automaton/-/tree/master/examples/times.py)

Those example implementations should provide a good start for your own project.

## Getting Involved
Feel free to open pull requests, send me feature requests or even join as developer.
There ist still quite some work to do.

And for all others, don't hesitate to open issues when you have problems!

## Changelog
#### 1.0.8
- Fixes automaton using edge cells with radius > 1 not working
- Fixes automaton is not stopping after evolution ended

#### 1.0.7
- Fixes automaton not active on reactivation

#### 1.0.6
- Fixes reactivation not redrawing all cells

#### 1.0.5
- Fixes black lines in the display between some cell lines/columns

#### 1.0.4
- Adds active state for automaton
- Adds reactivation method
- Fixes cells active state

#### 1.0.3
- Fixes init_cell_state called twice the necessary amount

#### 1.0.2
- Adds set CellularAutomaton.cells capability to be able to move cell states from one Automaton to another.

#### 1.0.1
- Add KeyboardInterrupt suppression to CAWindow

#### 1.0.0
- CI was added
- Coverage was increased
- Change project structure
    - removed multiprocess capability
    - reduced class count
- Improved algorithm to increase:
    - creation time by factor of ~2
    - processing speed by factor of ~15
- Changed API
    - No separate factory anymore: Just create a CellularAutomaton(...)
    - No Rule class anymore: Subclass CellularAutomaton and override `evolve_rule` and `init_cell_state`
    - Cell color is now defined by the CAWindow `state_to_color_cb` parameter.
    - Neighborhood does not need to know the dimension anymore

## Dependencies
There is only a dependency for [recordclass](https://pypi.org/project/recordclass/) at the moment. 

If you want to use the CAWindow or execute the examples you will have to install 
[pygame](https://www.pygame.org/news) for visualisation.
If you don't want to use this engine for some reason pass another draw_ending to the CAWindow.
It has to provides the same interface as the [PygameEngine](https://gitlab.com/DamKoVosh/cellular_automaton/-/blob/master/cellular_automaton/display.py) 

## Licence
This package is distributed under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0), 
see [LICENSE.txt](https://gitlab.com/DamKoVosh/cellular_automaton/-/tree/master/LICENSE.txt)