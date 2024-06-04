import numpy as np



# initial states, each entry is the position, velocity and goal of a pedestrian 
# in the form of (px, py, vx, vy, gx, gy) Please define only positive positions
initial_state = np.array([[1.0,1.0,1.0,1.0,1.0,1.0]])

# social groups informoation is represented as lists of indices of the state array
groups = []

# list of linear obstacles given in the form of (x_min, x_max, y_min, y_max) Please define only positive coordinates
obs = np.array([])
queue_obs = np.array([[14.0, 14.0, 7.0, 9.0 ], [4.0, 14.0, 9.0, 9.0], [4.0, 14.0, 7.0, 7.0]])
scaling_obs = np.array([[16,16,16,16]])
# obs = [[1, 2, 7, 8]]
# obs = None
group_radius = 0
people_in_group = 0
dim_grid = 40

simulation_steps = 1

robot_start_cont = (0.5,0.5)
robot_goal_cont = (13.5, 8.0)