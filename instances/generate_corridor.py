import numpy as np
import random
import matplotlib.pyplot as plt
import math
import cv2

def draw_wall(map, x0, y0, x1, y1):
    Dx = x1 - x0
    Dy = y1 - y0

    n_of_samples = max(math.floor(max(abs(Dx), abs(Dy)) * 2000), 1)  # max to not divide by zero
    dy = Dy / n_of_samples
    dx = Dx / n_of_samples

    for step in range(n_of_samples + 1):
        x = math.floor(x0 + step * dx)
        y = math.floor(y0 + step * dy)
        # Ensure x and y are within the bounds of the map
        if 0 <= x < map.shape[0] and 0 <= y < map.shape[1]:
            map[x, y] = 255  # occupied

    return map

def rotate_point(cx, cy, angle, px, py):
    s = math.sin(angle)
    c = math.cos(angle)
    
    # Translate point back to origin
    px -= cx
    py -= cy
    
    # Rotate point
    x_new = px * c - py * s
    y_new = px * s + py * c
    
    # Translate point back
    px = x_new + cx
    py = y_new + cy
    
    return int(px), int(py)

def create_gridmap_with_hallways(grid_size, min_distance=6):
    gridmap = np.zeros((grid_size, grid_size), dtype=np.uint8)

    # Ensure lines are at least 3 cells apart and the center is between them
    center = grid_size // 2
    start_y = random.randint(center - min_distance - 1, center - 1)
    end_y = random.randint(center + 1, center + min_distance + 1)
    
    # Define two parallel horizontal lines
    p1 = (-10, start_y)
    p2 = (grid_size + 10, start_y)
    q1 = (-10, end_y)
    q2 = (grid_size + 10, end_y)
    
    # Random angle for rotation
    angle = random.uniform(0, 2 * np.pi)

    # Center of the grid
    cx, cy = center, center
    
    # Rotate the points
    p1_rot = rotate_point(cx, cy, angle, *p1)
    p2_rot = rotate_point(cx, cy, angle, *p2)
    q1_rot = rotate_point(cx, cy, angle, *q1)
    q2_rot = rotate_point(cx, cy, angle, *q2)
    
    # Draw the walls on the gridmap
    gridmap = draw_wall(gridmap, *p1_rot, *p2_rot)
    gridmap = draw_wall(gridmap, *q1_rot, *q2_rot)
    
    return gridmap, angle, start_y, end_y, cx, cy

def create_symmetric_cost_map(grid_size, angle, start_y, end_y, goal_x, goal_y):
    cost_map = np.zeros((grid_size, grid_size), dtype=np.float64)
    center = grid_size // 2

    # Distance between the parallel walls
    distance_between_walls = abs(end_y - start_y)

    for x in range(grid_size):
        for y in range(grid_size):
            # Calculate direction based on the perpendicular angle
            dx = x - goal_x
            dy = y - goal_y
            distance_perpendicular_to_hallway = dx * math.sin(angle) - dy * math.cos(angle)
            distance_parallel_to_hallway = dx * math.cos(angle) + dy * math.sin(angle)
            sigma = distance_between_walls / 3.25
            
            # Calculate the cost based on the perpendicular distance to the hallway
            if 0 <= distance_perpendicular_to_hallway <= distance_between_walls:
                if distance_parallel_to_hallway>=0:
                    cost_map[x, y] = 255 * np.exp(-0.5 * (distance_perpendicular_to_hallway / sigma) ** 2)#255 - (255 * (distance_perpendicular_to_hallway / distance_between_walls))
                else:
                    cost_map[x, y] = 255 * np.exp(-0.5 * ((distance_between_walls-distance_perpendicular_to_hallway) / sigma) ** 2)#(255 * (distance_perpendicular_to_hallway / distance_between_walls))
    
    return cost_map

grid_size = 40
min_distance = 6

version = 0
for i in range(500):
    gridmap, angle, start_y, end_y, goal_x, goal_y = create_gridmap_with_hallways(grid_size, min_distance)
    cost_map = create_symmetric_cost_map(grid_size, angle, start_y, end_y, goal_x, goal_y)

    transposed_gridmap = np.transpose(gridmap)
    transposed_cost = np.transpose(cost_map)
    mirrored_gridmap = np.flip(gridmap, axis=1)
    mirrored_cost= np.flip(cost_map, axis=1)
    rotated_gridmap = np.flip(transposed_gridmap, axis=0)
    rotated_costmap = np.flip(transposed_cost, axis=0)
    cv2.imwrite(f'/home/ais/USAN/src/PySocialForce/images/in_out/hallway_step_{version}_out.jpg',cost_map)
    cv2.imwrite(f'/home/ais/USAN/src/PySocialForce/images/in_out/hallway_step_{version}_in.jpg', gridmap)
    version = version + 1
    cv2.imwrite(f'/home/ais/USAN/src/PySocialForce/images/in_out/hallway_step_{version}_out.jpg', transposed_cost)
    cv2.imwrite(f'/home/ais/USAN/src/PySocialForce/images/in_out/hallway_step_{version}_in.jpg', transposed_gridmap)
    version = version + 1
    cv2.imwrite(f'/home/ais/USAN/src/PySocialForce/images/in_out/hallway_step_{version}_out.jpg', mirrored_cost)
    cv2.imwrite(f'/home/ais/USAN/src/PySocialForce/images/in_out/hallway_step_{version}_in.jpg', mirrored_gridmap)
    version = version + 1
    cv2.imwrite(f'/home/ais/USAN/src/PySocialForce/images/in_out/hallway_step_{version}_out.jpg', rotated_costmap)
    cv2.imwrite(f'/home/ais/USAN/src/PySocialForce/images/in_out/hallway_step_{version}_in.jpg', rotated_gridmap)
    version = version + 1


