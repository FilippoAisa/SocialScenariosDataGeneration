import numpy as np
import random
import math
import matplotlib.pyplot as plt
import cv2
import os

def draw_wall(map, x0, y0, x1, y1):
    Dx = x1 - x0
    Dy = y1 - y0

    n_of_samples = max(math.floor(max(abs(Dx), abs(Dy)) * 10000), 1)  # max to not divide by zero
    dy = Dy / n_of_samples
    dx = Dx / n_of_samples

    for step in range(n_of_samples + 1):
        x = math.floor(x0 + step * dx)
        y = math.floor(y0 + step * dy)
        # Ensure x and y are within the bounds of the map
        if 0 <= x < map.shape[0] and 0 <= y < map.shape[1]:
            map[x, y] = 255  # occupied

    return map

def is_obtuse_angle(angle):
    return np.pi / 6 < angle < 5 * np.pi / 6

def generate_corner(map_size):
    gridmap_walls = np.zeros((map_size, map_size), dtype=np.uint8)
    gridmap_cost = np.zeros((map_size, map_size), dtype=np.uint8)
    
    corner_x = random.randint(map_size // 4, 3 * map_size // 4)
    corner_y = random.randint(map_size // 4, 3 * map_size // 4)
    
    angle1 = random.uniform(0, np.pi / 2)
    angle2 = angle1 + random.uniform(np.pi / 6, np.pi / 2)
    
    while not is_obtuse_angle(abs(angle2 - angle1)):
        angle2 = angle1 + random.uniform(np.pi / 6, np.pi / 2)
    
    wall_length = map_size
    x1_1 = int(corner_x + wall_length * np.cos(angle1))
    y1_1 = int(corner_y + wall_length * np.sin(angle1))
    x1_2 = int(corner_x + wall_length * np.cos(angle2))
    y1_2 = int(corner_y + wall_length * np.sin(angle2))
    
    x1_1 = max(0, min(map_size - 1, x1_1))
    y1_1 = max(0, min(map_size - 1, y1_1))
    x1_2 = max(0, min(map_size - 1, x1_2))
    y1_2 = max(0, min(map_size - 1, y1_2))
    
    gridmap_walls = draw_wall(gridmap_walls, corner_x, corner_y, x1_1, y1_1)
    gridmap_walls = draw_wall(gridmap_walls, corner_x, corner_y, x1_2, y1_2)
    
    sigma = map_size / 8
    for i in range(map_size):
        for j in range(map_size):
            v1 = np.array([x1_1 - corner_x, y1_1 - corner_y])
            v2 = np.array([x1_2 - corner_x, y1_2 - corner_y])
            v_point = np.array([i - corner_x, j - corner_y])
            angle_to_v1 = np.arctan2(np.cross(v1, v_point), np.dot(v1, v_point))
            angle_to_v2 = np.arctan2(np.cross(v2, v_point), np.dot(v2, v_point))
            if not (angle_to_v1 > 0 and angle_to_v2 < 0):
                dist = np.sqrt((i - corner_x) ** 2 + (j - corner_y) ** 2)
                cost = int(np.floor(255 * np.exp(-dist ** 2 / (2 * sigma ** 2))))
                gridmap_cost[i, j] = max(gridmap_cost[i, j], cost)
    
    return gridmap_walls, gridmap_cost

map_size = 40

version = 0
parent_dir = os.path.dirname(os.path.realpath(__file__))
output_dir = os.path.join(parent_dir, 'images/in_out')
os.makedirs(output_dir, exist_ok=True)
for i in range(500):
    gridmap_walls, gridmap_cost = generate_corner(map_size)
    transposed_gridmap = np.transpose(gridmap_walls)
    transposed_cost = np.transpose(gridmap_cost)
    mirrored_gridmap = np.flip(gridmap_walls, axis=1)
    mirrored_cost= np.flip(gridmap_cost, axis=1)
    rotated_gridmap = np.flip(transposed_gridmap, axis=0)
    rotated_costmap = np.flip(transposed_cost, axis=0)
    cv2.imwrite(os.path.join(output_dir, f'corner_step_{version}_out.jpg'), gridmap_cost)
    cv2.imwrite(os.path.join(output_dir, f'corner_step_{version}_in.jpg'), gridmap_walls)
    version +=1
    cv2.imwrite(os.path.join(output_dir, f'corner_step_{version}_out.jpg'), transposed_cost)
    cv2.imwrite(os.path.join(output_dir, f'corner_step_{version}_in.jpg'), transposed_gridmap)
    version +=1
    cv2.imwrite(os.path.join(output_dir, f'corner_step_{version}_out.jpg'), mirrored_cost)
    cv2.imwrite(os.path.join(output_dir, f'corner_step_{version}_in.jpg'), mirrored_gridmap)
    version +=1
    cv2.imwrite(os.path.join(output_dir, f'corner_step_{version}_out.jpg'), rotated_costmap)
    cv2.imwrite(os.path.join(output_dir, f'corner_step_{version}_in.jpg'), rotated_gridmap)
    version +=1