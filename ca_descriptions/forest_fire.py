# Name: COM3524 Group Assignment Forest Fire
# Dimensions: 2

# --- Set up executable path, do not edit ---
import sys
import inspect
import math
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

# Scaling factor used globally across multiple functions
scale = 10

def transition_func(grid, neighbourstates, neighbourcounts, fuel_grid, timestep):
    # States: 
    # chaparral = 0
    # lake = 1
    # forest = 2
    # canyon = 3
    # town = 4
    # burning = 5
    # burnt = 6

    # Iterate the timestep variable
    timestep[0] += 1

    # The cartesian difference (x, y) between a given cell and its neighbours, in this order: NW, N, NE, W, E, SW, S, SE 
    deltas = [(1, 1), (0, 1), (-1, 1), (1, 0), (-1, 0), (1, -1), (0, -1), (-1, -1)]

    # The cell states that can burn, and their weighting factor.
    types = {0: 0.1, 1: 0, 2: 0.05, 3: 1}

    # Deplete all burning cells fuel by 1 every generation
    burning_cells = (grid == 5)
    fuel_grid[burning_cells] -= 1

    # Transition depleted cells to burnt state (6)
    burnt_out = (fuel_grid == 0)
    grid[burnt_out] = 6

    town_reached = (grid == 4) & (neighbourcounts[5] >= 1)

    # Print the time taken to reach the town to terminal
    if grid[town_reached].size >= 1 and timestep[1] == 0:
        print("Fire has reached town at time-step: " + str(timestep[0]))
        print("Minutes elapsed: " + str(timestep[0] * 10))
        print("Days elapsed: " + str(((timestep[0] * 10) * 0.000694444)))
        timestep[1] = 1


    for state in [0, 2, 3]:
        # Calculate the probability for each cell to burn and apply it on the grid
        for (delta, neighbour) in zip(deltas, neighbourstates):
        
            # Boolean array for cell of state 0, 2 or 3 with burning neighbours
            cell_burn = (neighbour == 5) & (grid == state)

            # Run the randomizer on each cell with a burning neighbour
            updated_grid = randomizer(state, delta, types[state], cell_burn.sum())

            # Update the grid with the new burning cells
            grid[cell_burn] = updated_grid
    
    return grid

def randomizer(current_state, deltas, type, size):
    """
    Sets the current cell(s) to state 5 (burning) with a calculated probability.

    Returns:
    new_state: Either 5 (The cell(s) starts burning) or the same state of the cell(s) initially
    """
    prob = pburn(deltas, type)
    new_grid = np.random.choice([current_state, 5], size, p=[(1-prob), prob])

    return(new_grid)

# ------------------ State and Fuel Grids Setup -------------------------

def setup(args):
    config_path = args[0]
    config = utils.load(config_path)
    # ---THE CA MUST BE RELOADED IN THE GUI IF ANY OF THE BELOW ARE CHANGED---
    config.title = "Forest Fire"
    config.dimensions = 2
    config.states = (0, 1, 2, 3, 4, 5, 6)
    config.wrap = False
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
    config.grid_dims = (50 * scale, 50 * scale)

    # Choose the burn site: incinerator or power plant
    config.initial_grid = grid_setup("power plant", scale)

    # ----------------------------------------------------------------------

    if len(args) == 2:
        config.save()
        sys.exit()

    return config

def grid_setup(fire_location, scale):
    """
    Sets up the inital grid with the different states and vegetation types
    
    Args: 
    fire_location: (NOT YET IMPLEMENTED) The starting point of the fire.
    scale: Scaling factor for the grid to adjust its size.

    Returns:
    initial_grid: The grid, fully initialized
    """
    if fire_location == "incinerator":
        burn_site = (0, int(50 * scale) - 1)
    elif fire_location == "power plant":
        burn_site = (0, int(10 * scale) - 1)

    initial_grid = np.full((50 * scale, 50 * scale), 0) # Chaparral
    initial_grid = define_state(initial_grid, 1, scale, [15, 5], [20, 20]) # Lake
    initial_grid = define_state(initial_grid, 2, scale, [0, 25], [20, 35]) # Forest
    initial_grid = define_state(initial_grid, 3, scale, [35, 5], [37, 45]) # Canyon
    initial_grid = define_state(initial_grid, 4, scale, [8, 44], [12, 47]) # Town
    initial_grid[burn_site[0]][burn_site[1]] = 5

    return initial_grid

def define_state(grid, state, scale, bottom_left, top_right):
    """
    Sets the state of cells in a given rectangular area on the grid

    Args:
    grid: The grid
    state: The state that the cells will be given
    scale: The scaling factor of the grid
    bottom_left: The coordinates of the bottom left cell in the chosen area
    top_right: The coordinates of the top right cell in the chosen area

    Returns:
    grid: The grid with the updated states
    """
    bottom_left[0] *= scale
    top_right[0] *= scale
    bottom_left[1] *= scale
    top_right[1] *= scale

    for i in range(bottom_left[0], top_right[0]):
        for j in range(bottom_left[1], top_right[1]):
            grid[j][i] = state

    return grid


def fuel_setup():
    """ Sets up the initial fuel grid to allow for varied burning duration
    depending on the type of terrain.
    """
    # Add the fuel value of chapparal (360 timesteps * 20 mins = 5 days of burning)
    fuel_grid = np.full((50 * scale, 50 * scale), 360)
    fuel_grid = define_fuel(fuel_grid, 1, scale, [15, 5], [20, 20]) # Lake
    fuel_grid = define_fuel(fuel_grid, 2, scale, [0, 25], [20, 35]) # Forest
    fuel_grid = define_fuel(fuel_grid, 3, scale, [35, 5], [37, 45]) # Canyon
    fuel_grid = define_fuel(fuel_grid, 4, scale, [8, 44], [12, 47]) # Town

    return fuel_grid

def define_fuel(grid, state, scale, bottom_left, top_right):
    """
    Sets the initial fuel values of cells based on their state in
    a given rectangular area

    Args:
    grid: The fuel grid
    state: The state of the cells in the area
    scale: The scaling factor of the grid
    bottom_left: The coordinates of the bottom left cell in the area
    top_right: The coordinates of the top right cell in the area

    Returns:
    grid: The grid with the added fuel values
    """
    # Scale up the x and y coordinates
    bottom_left[0] *= scale
    top_right[0] *= scale
    bottom_left[1] *= scale
    top_right[1] *= scale
    
    # Set the fuel value based on the state/terrain type
    match state:
            case 1:
                fuel = -1 # Lake (Impossible to ignite)
            case 2:
                fuel = 2160 # Forest (2160 * 20 minutes = 1 month)
            case 3:
                fuel = 21 # Canyon (21 * 20 minutes = 7 hours)
            case 4:
                fuel = 1 # Town

    # Loop through all cells in the given rectangular block and set the fuel value
    for i in range(bottom_left[0], top_right[0]):
        for j in range(bottom_left[1], top_right[1]):
            grid[j][i] = fuel
    
    return (grid)

# ------------------ Probability Calculation -------------------------

def pwind(v, wd, c1, c2, deltas):
    """
    Calculates the wind weighting factor.

    Args:
    v: Wind velocity (m/s)
    wd: Wind direction (0-360 degrees)
    c1, c2: Adjustable coefficients
    deltas: Difference between coordinates of current cell and burning cell.
    """

    deltaX = deltas[0]
    deltaY = deltas[1]

    degrees_temp = math.atan2(deltaX, deltaY) / math.pi * 180

    if degrees_temp <= 0:
        degrees_final = 360 + degrees_temp
    else:
        degrees_final = degrees_temp

    theta = degrees_final - wd
    ft = math.exp(v * c2 * (math.cos(math.radians(theta)) - 1))
    pw = math.exp(v * c1) * ft

    return pw

def pburn(deltas, type_weight):
    """
    Calculates the probability of a cell burning in the next timestep
    given the fact that one of the cell's neighbours is burning.

    Args:
    deltas: The difference between the x and y coordinates of the cell and the burning neighbour cell
    type_weight: A weighting factor to account for the different types of terrain.

    Returns:
    pburn: The probability of burning. If >= 1, the cell burns, else remains the same.
    """

    # Speed of wind in m/s
    wind_speed = 9

    # Direction of wind. 0 degrees is north to south, ascends clockwise
    wind_direction = 0
    
    # Base probability irrespective of wind and terrain type
    p0 = 0.58

    # Wind factor
    pw = pwind(wind_speed, wind_direction, 0.045, 0.131, deltas)
    
    # Probability of burning = Base probability * wind factor * terrain type factor
    pburn = p0 * pw * type_weight

    if pburn > 1:
        pburn = 1

    return pburn

# ------------------ Main Function -------------------------


def main():
    # Open the config object
    config = setup(sys.argv[1:])

    fuel_grid = fuel_setup()

    timestep = np.array([0, 0])

    # Create grid object
    grid = Grid2D(config, (transition_func, fuel_grid, timestep))

    # Run the CA, save grid state every generation to timeline
    timeline = grid.run()

    # save updated config to file
    config.save()
    # save timeline to file
    utils.save(timeline, config.timeline_path)


if __name__ == "__main__":
    main()
