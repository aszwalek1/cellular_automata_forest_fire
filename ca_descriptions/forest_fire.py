# Name: COM3524 Group Assignment Forest Fire
# Dimensions: 2

# --- Set up executable path, do not edit ---
import sys
import inspect
this_file_loc = (inspect.stack()[0][1])
main_dir_loc = this_file_loc[:this_file_loc.index('ca_descriptions')]
sys.path.append(main_dir_loc)
sys.path.append(main_dir_loc + 'capyle')
sys.path.append(main_dir_loc + 'capyle/ca')
sys.path.append(main_dir_loc + 'capyle/guicomponents')
# ---

from capyle.ca import Grid2D, Neighbourhood, CAConfig, randomise2d
import capyle.utils as utils
import numpy as np


def transition_func(grid, neighbourstates, neighbourcounts):
    # chaparral = 0
    # lake = 1
    # forest = 2
    # canyon = 3
    # town = 4
    # burning = 5
    # burnt = 6

    chaparral, lake, forest, canyon, town, burning, burnt = neighbourcounts
    # create boolean arrays for the birth & survival rules
    # if 3 live neighbours and is dead -> cell born
    # birth = (live_neighbours == 3) & (grid == 0)
    # if 2 or 3 live neighbours and is alive -> survives
    # survive = ((live_neighbours == 2) | (live_neighbours == 3)) & (grid == 1)
    # Set all cells to 0 (dead)
    grid[:, :] = 0
    # Set cells to 1 where either cell is born or survives
    # grid[birth | survive] = 1
    return grid


def setup(args):
    config_path = args[0]
    config = utils.load(config_path)
    # ---THE CA MUST BE RELOADED IN THE GUI IF ANY OF THE BELOW ARE CHANGED---
    config.title = "Forest Fire"
    config.dimensions = 2
    config.states = (0, 1, 2, 3, 4, 5, 6)
    # ------------------------------------------------------------------------

    # ---- Override the defaults below (these may be changed at anytime) ----
    chaparral = (191/255, 190/255, 2/255)
    lake = (1/255, 176/255, 241/255)
    forest = (80/255, 98/255, 40/255)
    canyon = (253/255, 253/255, 9/255)
    town = (131/255, 105/255, 83/255)
    burning = (194/255, 24/255, 7/255)
    burnt = (0, 0, 0)
    config.state_colors = [chaparral, lake, forest, canyon, town, burning, burnt]
    scale = 1

    # config.num_generations = 150
    config.grid_dims = (50 * scale, 50 * scale)
    config.initial_grid = grid_setup("incinerator", scale)

    # ----------------------------------------------------------------------

    if len(args) == 2:
        config.save()
        sys.exit()

    return config

def grid_setup(fire_location, scale):
    initial_grid = np.full((50 * scale, 50 * scale), 0)
    initial_grid = define_state(initial_grid, 1, scale, [15, 5], [20, 20])
    initial_grid = define_state(initial_grid, 2, scale, [0, 25], [20, 35])
    initial_grid = define_state(initial_grid, 3, scale, [35, 5], [37, 45])
    initial_grid = define_state(initial_grid, 4, scale, [8, 44], [12, 47])

    return initial_grid

def define_state(grid, state, scale, bottom_left, top_right):
    bottom_left[0] *= scale
    top_right[0] *= scale

    for i in range(bottom_left[0], top_right[0]):
        for j in range(bottom_left[1], top_right[1]):
            grid[j][i] = state

    return grid

def main():
    # Open the config object
    config = setup(sys.argv[1:])

    # Create grid object
    grid = Grid2D(config, transition_func)

    # Run the CA, save grid state every generation to timeline
    timeline = grid.run()

    # save updated config to file
    config.save()
    # save timeline to file
    utils.save(timeline, config.timeline_path)


if __name__ == "__main__":
    main()
