import numpy as np



# initial states, each entry is the position, velocity and goal of a pedestrian 
# in the form of (px, py, vx, vy, gx, gy) Please define only positive positions
""" initial_state = np.array(
        [
            [15.0, 15.0, 0.0, 0.0, 25.0, 5.0],
            [14.0, 19.0, 0.0, 0.0, 24.0, 9.0],
            [15.0, 22.0, 0.0, 0.0, 25.0, 2.0],
            [19.0, 23.0, 0.0, 0.0, 29.0, 3.0],
            [22.0, 21.0, 0.0, 0.0, 32.0, 1.0],
            [22.0, 16.0, 0.0, 0.0, 32.0, 6.0],
            [19.0, 14.0, 0.0, 0.0, 29.0, 4.0],
        ]
    )
    
 """
initial_state = np.array([[]])

# social groups informoation is represented as lists of indices of the state array
#groups = [[0, 1, 2, 3, 4, 5, 6]]
groups = [[]]
# list of linear obstacles given in the form of (x_min, x_max, y_min, y_max) Please define only positive coordinates
obs = np.array([])
queue_obs = np.array([])
scaling_obs = np.array([[16,16,16,16]])
group_radius = 0
people_in_group = 0
# obs = [[1, 2, 7, 8]]
# obs = None

dim_grid = 40

simulation_steps = 1

robot_start_cont = (0.5,0.5)
robot_goal_cont = (12.0, 8.0)