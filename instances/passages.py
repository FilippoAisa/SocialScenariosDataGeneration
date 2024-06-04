import numpy as np



# initial states, each entry is the position, velocity and goal of a pedestrian 
# in the form of (px, py, vx, vy, gx, gy) Please define only positive positions
initial_state = np.array(
        [
            [25.0, 45.0, 0.0, 0.0, 25.0, 45.0],
            [27.0, 43.0, 0.0, 0.0, 27.0, 43.0],
            [29.0, 45.0, 0.0, 0.0, 29.0, 45.0],
            [31.0, 41.0, 0.0, 0.0, 31.0, 41.0],
            [32.0, 44.0, 0.0, 0.0, 32.0, 44.0]
        ]
    )

# social groups informoation is represented as lists of indices of the state array
groups = [[]]

# list of linear obstacles given in the form of (x_min, x_max, y_min, y_max) Please define only positive coordinates
obs = np.array([[27, 27, 0, 41], [27, 27, 48, 71], [27, 27, 76, 110], [45, 45, 105, 107]])
# obs = [[1, 2, 7, 8]]
# obs = None

x_dim_grid = 60

simulation_steps = 1

robot_start_cont = (10, 20)
robot_goal_cont = (40, 10)