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


# ------------------ Do your testing here! -------------------------

# Default values:
# set_location = "power plant"
# set_wind_direction = 0
# set_wind_speed = 9
# forest_state = 1
# lake_state = 1
# time_value = 10

# Set location of the ignition site, "incincerator" or "power plant"
set_location = "power plant"

# Set wind direction
# 0 - North (to South)
# 90 - East (to West)
# 180 - South (to North)
# 270 - West (to East)
set_wind_direction = 0

# Set wind speed
set_wind_speed = 9

# Extend the forest
# 1 - Default
# 2 - 5km upwards
# 3 - 5km downwards
# 4 - 15km to the right
# 5 - 10km to the right
# 6 - 5km to the right
# 7 - 10km down to the side of the town
# 8 - 15km down to the side of the town
# 9 - Combined
forest_state = 1

# Extend the lake
# 1 - Default
# 2 - 5km downwards
# 3 - 5km upwards
# 4 - Combined
lake_state = 1

# Actual time value of a generation. Values between 1 to 20 reccomended for realistic speeds.
# NOTE: This will also change the basic rate of spread (ROS) of the fire accordingly so the simulation
# visually moves at the same speed and has the same iteration count, but actual timescales will shift.
# A value of 10 was chosen to have enough time to show some of the chapparal burning.
time_value = 3


# --------------- WATER DROP INTERVENTION -------------
# Set to true to use water drop
use_water_intervention = True

# Timestep at wich water intevention is deployed
# Simulation usually complete around 2000 interventions so value lower than that recommended
intervention_timestep = 100

# Point [x, y] on the map that the water intervention will be dropped 
drop_point = [100, 100]


# ------------------ The Code -------------------------

# Global variables:
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
    types = {0: 0.1, 1: 0, 2: 0.02, 3: 1}

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
        print("Minutes elapsed: " + str(timestep[0] * time_value))
        print("Days elapsed: " + str(((timestep[0] * time_value) * 0.000694444)))
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

    if use_water_intervention:
        if timestep[0] == intervention_timestep :
            grid = drop_water(grid)
    
<<<<<<< HEAD
    # Calculate the probability for each cell to burn and apply it on the grid
    for (delta, neighbour) in zip(deltas, neighbourstates):
        
        # Chaparral cells with burning neighbours
        chaparral_burn = (neighbour == 5) & (grid == 0)
        grid[chaparral_burn] = randomizer(0, delta, types[0])
        
        # Forest cells with burning neighbours
        forest_burn = (neighbour == 5) & (grid == 2)
        grid[forest_burn] = randomizer(2, delta, types[2])

        # Canyon cells with burning neighbours
        canyon_burn = (neighbour == 5) & (grid == 3)
        grid[canyon_burn] = randomizer(3, delta, types[3])

    #if timestep[0] == intervention_timestep :
    #    grid = drop_water(grid)

=======
>>>>>>> omar
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

def drop_water(grid):
    """
    Function used to represent the water drop intervention

    Returns:
    grid: New grid with added burnt out cells where water has been dropped
    on burning cells
    """
    radius = int((math.sqrt(12500 / math.pi)) / scale)
    
<<<<<<< HEAD
    cx = drop_point[0]
    cy = drop_point[1]
=======
    cy = drop_point[0]
    cx = drop_point[1]
>>>>>>> omar

    for i in range(cy - radius, cy + radius):
        for j in range(cx - radius, cx + radius):
            if ((j - cx)**2 + (i - cy)**2) <= (radius **2):
                grid[j][i] = water_prob(grid[j][i])

    return grid

def water_prob(current_state):
    """
    Decides whether the water drop intervention was successful for a given cell 

    Returns:
    new_state: new state of the input cell
    """
    if current_state == 1:
        new_state = current_state
    elif current_state == 4:
        new_state = current_state
    else:
        prob = .9
        new_state = np.random.choice([current_state, 6], 1, p=[(1-prob), prob])
        new_state = new_state[0]
    return(new_state)

# ------------------ State and Fuel Grids Setup -------------------------

def setup(args):
    config_path = args[0]
    config = utils.load(config_path)
    config.title = "Forest Fire"
    config.dimensions = 2
    config.states = (0, 1, 2, 3, 4, 5, 6)
    config.wrap = False
    # ------------------------------------------------------------------------

    chaparral = (191/255, 190/255, 2/255)
    lake = (1/255, 176/255, 241/255)
    forest = (80/255, 98/255, 40/255)
    canyon = (253/255, 253/255, 9/255)
    town = (131/255, 105/255, 83/255)
    burning = (194/255, 24/255, 7/255)
    burnt = (0, 0, 0)
    config.state_colors = [chaparral, lake, forest, canyon, town, burning, burnt]
    config.grid_dims = (50 * scale, 50 * scale)

    config.initial_grid = grid_setup(set_location, scale)

    # ----------------------------------------------------------------------

    if len(args) == 2:
        config.save()
        sys.exit()

    return config

def grid_setup(fire_location, scale):
    """
    Sets up the inital grid with the different states and vegetation types
    
    Args: 
    fire_location: The starting point of the fire.
    scale: Scaling factor for the grid to adjust its size.

    Returns:
    initial_grid: The grid, fully initialized
    """
    if fire_location == "incinerator":
        burn_site = (0, int(50 * scale) - 1)
    elif fire_location == "power plant":
        burn_site = (0, int(10 * scale) - 1)
    
    # Checking for forest and lake extensions
    if forest_state == 1 or forest_state == 7 or forest_state == 8:
        forest1 = [0, 25]
        forest2 = [20, 35]

    elif forest_state == 2 or forest_state == 9:
        forest1 = [0, 20]
        forest2 = [20, 35]

    elif forest_state == 3:
        forest1 = [0, 25]
        forest2 = [20, 40]
    
    elif forest_state == 4:
        forest1 = [0, 25]
        forest2 = [35, 35]
    
    elif forest_state == 5:
        forest1 = [0, 25]
        forest2 = [30, 35]
    
    elif forest_state == 6:
        forest1 = [0, 25]
        forest2 = [25, 35]
    
    if lake_state == 1:
        lake1 = [15, 5]
        lake2 = [20, 20]
    
    elif lake_state == 2:
        lake1 = [15, 5]
        lake2 = [20, 25]

    elif lake_state == 3:
        lake1 = [15, 0]
        lake2 = [20, 20]
    
    elif lake_state == 4:
        lake1 = [15, 0]
        lake2 = [20, 25]

    initial_grid = np.full((50 * scale, 50 * scale), 0) # Chaparral
    initial_grid = define_state(initial_grid, 1, scale, lake1, lake2) # Lake
    initial_grid = define_state(initial_grid, 2, scale, forest1, forest2) # Forest
    initial_grid = define_state(initial_grid, 3, scale, [35, 5], [37, 45]) # Canyon
    initial_grid = define_state(initial_grid, 4, scale, [8, 44], [10.5, 46.5]) # Town
    initial_grid[burn_site[0]][burn_site[1]] = 5

    if forest_state == 7:
        initial_grid = define_state(initial_grid, 2, scale, [10.5, 35], [20, 45])
    elif forest_state == 8 or forest_state == 9:
        initial_grid = define_state(initial_grid, 2, scale, [10.5, 35], [20, 50])


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

    for i in range(int(bottom_left[0]), int(top_right[0])):
        for j in range(int(bottom_left[1]), int(top_right[1])):
            grid[j][i] = state

    return grid


def fuel_setup():
    """ Sets up the initial fuel grid to allow for varied burning duration
    depending on the type of terrain.
    """
    # Checking for forest and lake extensions
    if forest_state == 1 or forest_state == 7 or forest_state == 8:
        forest1 = [0, 25]
        forest2 = [20, 35]

    elif forest_state == 2 or forest_state == 9:
        forest1 = [0, 20]
        forest2 = [20, 35]

    elif forest_state == 3:
        forest1 = [0, 25]
        forest2 = [20, 40]
    
    elif forest_state == 4:
        forest1 = [0, 25]
        forest2 = [35, 35]
    
    elif forest_state == 5:
        forest1 = [0, 25]
        forest2 = [30, 35]
    
    elif forest_state == 6:
        forest1 = [0, 25]
        forest2 = [25, 35]
    
    if lake_state == 1:
        lake1 = [15, 5]
        lake2 = [20, 20]
    
    elif lake_state == 2:
        lake1 = [15, 5]
        lake2 = [20, 25]

    elif lake_state == 3:
        lake1 = [15, 0]
        lake2 = [20, 20]
    
    elif lake_state == 4:
        lake1 = [15, 0]
        lake2 = [20, 25]
    
    # Add the fuel value of chapparal (7200min / time_value = 5 days of burning)
    fuel_grid = np.full((50 * scale, 50 * scale), int(7200/time_value))
    fuel_grid = define_fuel(fuel_grid, 1, scale, lake1, lake2) # Lake
    fuel_grid = define_fuel(fuel_grid, 2, scale, forest1, forest2) # Forest
    fuel_grid = define_fuel(fuel_grid, 3, scale, [35, 5], [37, 45]) # Canyon
    fuel_grid = define_fuel(fuel_grid, 4, scale, [8, 44], [10.5, 46.5]) # Town

    if forest_state == 7:
        fuel_grid = define_fuel(fuel_grid, 2, scale, [10.5, 35], [20, 45])
    
    elif forest_state == 8 or forest_state == 9:
        fuel_grid = define_fuel(fuel_grid, 2, scale, [10.5, 35], [20, 50])

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
    if state == 1:
        fuel = -1 # Lake (Impossible to ignite)
    elif state == 2:
        fuel = int(43800/time_value) # Forest (43800m / time_value = 1 month)
    elif state == 3:
        fuel = int(420/time_value) # Canyon (420 / time_value = 7 hours)
    elif state == 4:
        fuel = 1 # Town
    else:
        fuel = 0

    # Loop through all cells in the given rectangular block and set the fuel value
    for i in range(int(bottom_left[0]), int(top_right[0])):
        for j in range(int(bottom_left[1]), int(top_right[1])):
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
    wind_speed = set_wind_speed

    # Direction of wind. 0 degrees is north to south, ascends clockwise
    wind_direction = set_wind_direction
    
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
    
    timestep = np.array([0])
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
