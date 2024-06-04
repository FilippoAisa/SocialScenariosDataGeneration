import numpy as np



# initial states, each entry is the position, velocity and goal of a pedestrian 
# in the form of (px, py, vx, vy, gx, gy) Please define only positive positions
initial_state = np.array(
        [
            [9.0, 70.0, -5.0, -5.0, 9.0, 70.0],
            [5.0, 70.0, -5.0, -5.0, 5.0, 70.0],
            # [10.0, 10.0, 0.0, 5, 15.0, 110.0],
            # [1.0, 0.0, 0.0, 0.5, 2.0, 10.0],
            # [2.0, 0.0, 0.0, 0.5, 3.0, 10.0],
            # [3.0, 0.0, 0.0, 0.5, 4.0, 10.0],
        ]
    )

# social groups informoation is represented as lists of indices of the state array
groups = [[1, 0]]

# list of linear obstacles given in the form of (x_min, x_max, y_min, y_max) Please define only positive coordinates
obs = np.array([[2, 2, 10, 110], [20, 20, 10, 110]])
# obs = [[1, 2, 7, 8]]
# obs = None


dim_grid = 25

simulation_steps = 1

robot_start_cont = (9,15)
robot_goal_cont = (9,110)