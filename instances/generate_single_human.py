import numpy as np
import pysocialforce as psf
import math
import cv2
import random
import angles
import training.queue_9m as instance

def humans_in_gridmap(humans, human_gridmap):
    # Determine the maximum x and y values for scaling
    x_max = np.max(scaling_obs[:, 1])
    y_max = np.max(scaling_obs[:, 3])

    # Scale humans' positions to grid dimensions
    scale_hum2grid = np.diag([x_dim_grid / x_max, y_dim_grid / y_max])
    humans_grid_pos = np.uint16(np.floor(humans[:, :2] @ scale_hum2grid))

    # Update the human_gridmap in place
    for i in range(humans_grid_pos.shape[0]):
        human_gridmap[humans_grid_pos[i, 0], humans_grid_pos[i, 1]] = 100
        dx = int(np.sign(humans[i, 2]))
        dy = int(np.sign(humans[i, 3]))
        
        if 0 <= humans_grid_pos[i, 0] + dx < human_gridmap.shape[0] and 0 <= humans_grid_pos[i, 1] + dy < human_gridmap.shape[1]:
            human_gridmap[int(humans_grid_pos[i,0] + dx), int(humans_grid_pos[i,1] + dy)] = max(25, human_gridmap[int(humans_grid_pos[i,0] + dx), int(humans_grid_pos[i,1] + dy)])
            
            #print(f'x = {human[0]} y = {human[1]}, value = {human_gridmap[human[0], human[1]]}')

    return human_gridmap

def people_costs(people_label, people):
    
    amplitude = 255.0
    covariance_front_height = 0.25
    covariance_front_width = 0.25
    covariance_rear_height = 0.25
    covariance_rear_width = 0.25
    covariance_right_height = 0.75
    covariance_right_width = 0.25
    covariance_when_still = 0.25
    use_passing = True
    resolution = 0.4# resolution assumed to be 40 cm per square
    tolerance_vel_still = 0.5
    cutoff = 10
    use_vel_factor = True
    speed_factor = 1

    # Calculate velocity magnitude and angle for each person
    for person in people:
        angle = np.arctan2(person[3], person[2])
        angle_right = angle - 1.57  # 90 degrees in radians
        cx = person[0] * resolution
        cy = person[1] * resolution

        # Iterate through each cell in the cost layer
        for i in range(x_dim_grid):
            for j in range(y_dim_grid):
                x = i * resolution
                y = j * resolution
                mag = (person[3]**2 + person[2]**2)**0.5
                if mag < tolerance_vel_still:
                    # PERSON STANDS STILL
                    a = calculate_gaussian(x, y, cx, cy, amplitude, covariance_when_still, covariance_when_still, 0)
                else:
                    ma = np.arctan2(y - cy, x - cx)
                    diff = angles.shortest_angular_distance(angle, ma)


                    # FRONT
                    if math.fabs(diff) < math.pi / 2:
                        if use_vel_factor:
                            factor = 1.0 + mag * speed_factor
                            a = calculate_gaussian(x, y, cx, cy, amplitude, covariance_front_height * factor, covariance_front_width, angle)
                        else:
                            a = calculate_gaussian(x, y, cx, cy, amplitude, covariance_front_height, covariance_front_width, angle)
                    else:  # REAR
                        a = calculate_gaussian(x, y, cx, cy, amplitude, covariance_rear_height, covariance_rear_width, angle)

                    # RIGHT SIDE
                    if use_passing:
                        diff_right = angles.shortest_angular_distance(angle_right, ma)
                        if math.fabs(diff_right) < math.pi / 2:
                            a_right = calculate_gaussian(x, y, cx, cy, amplitude, covariance_right_height, covariance_right_width, angle_right)
                            a = max(a, a_right)
                    
                    

                if a < cutoff:
                    continue

                # Update the cost in the costmap
                old_cost = people_label[i, j]
                people_label[i, j] = max(a, old_cost)
                
    return people_label


# Function to calculate Gaussian value
def calculate_gaussian(x, y, x0, y0, A, varx, vary, skew):
    dx = x - x0
    dy = y - y0
    h = np.sqrt(dx ** 2 + dy ** 2)
    angle = np.arctan2(dy, dx)
    mx = np.cos(angle - skew) * h
    my = np.sin(angle - skew) * h
    f1 = mx ** 2 / (2 * varx)
    f2 = my ** 2 / (2 * vary)
    return A * np.exp(-(f1 + f2))


version = 0
for i in range(1):
    x = random.uniform(0.0, 16.0)
    y = random.uniform(0.0, 16.0)
    vx = random.uniform(0.001,1)
    vy = random.uniform(0.001,1)
    person = np.array([x,y,vx,vy])
    person_input =  np.zeros((x_dim_grid,y_dim_grid), dtype=np.uint8)
    person_label = np.zeros((x_dim_grid,y_dim_grid), dtype=np.uint8)
    person_input = humans_in_gridmap(person,person_input)
    person_label = people_costs(person_label, np.concatenate([person[:,:2]@scale_hum2grid, person[:,2:4]], axis=1))
    cv2.imwrite(f'/home/ais/USAN/src/PySocialForce/images/in_out/single_person_step_{version}_out.jpg', person_label)
    cv2.imwrite(f'/home/ais/USAN/src/PySocialForce/images/in_out/single_person_step_{version}_in.jpg', person_input)
    version = version+1