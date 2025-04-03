import os
import numpy as np 
import pandas as pd
import time 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
# Check if GPU is available 
is_gpu = torch.cuda.is_available()
device = torch.device("cuda" if is_gpu else "cpu")
device = torch.device("cpu")

print("Using GPU:", is_gpu)

# # Function to generate a structured grid

# Input Parameters
Mixer_radius = 50
Mixer_anuglar_velocity_z = 1
domain_size_x = 100
domain_size_y = 100
domain_size_z = 18

half_domain_size_x = int(domain_size_x/2)
half_domain_size_y = int(domain_size_y/2)
half_domain_size_z = int(domain_size_z/2)

cell_size =0.00054  # Cell size and particle radius
kn =  200           # Normal stiffness of the spring
rho_p = 2500 
particle_mass = 4/3*3.1415*cell_size**3*rho_p #4188.7902
K_graph = 160*10000000000*1
S_graph = K_graph * (cell_size / domain_size_x) ** 2
restitution_coefficient = 0.3  # coefficient of restitution
friction_coefficient = 0.8  # coefficient of friction
friction_coefficient_wall_to_sphere = 5 # coefficient of friction
k_values   = np.array([1,8])
k_values_2 = np.array([3,5])

# coefficient of damping calculation 
# Discrete particle simulation of two-dimensional fluidized bed
damping_coefficient_Alpha      = -1 * math.log(restitution_coefficient)/math.pi
damping_coefficient_Gamma      = damping_coefficient_Alpha/math.sqrt(damping_coefficient_Alpha**2+1)
damping_coefficient_Eta        = 2 * damping_coefficient_Gamma * math.sqrt(kn * particle_mass/2)
damping_coefficient_Eta_wall   = 2 * damping_coefficient_Gamma * math.sqrt(kn * particle_mass)
print('Damping Coefficient:', damping_coefficient_Eta)

# Module 1: Domain discretisation and initial particle insertion
# Create grid
input_shape_global = (1, 1, domain_size_z, domain_size_y, domain_size_x)

# Generate particles
'''
x_grid = np.zeros(input_shape_global)
y_grid = np.zeros(input_shape_global)
z_grid = np.zeros(input_shape_global)

vx_grid = np.zeros(input_shape_global)
vy_grid = np.zeros(input_shape_global)
vz_grid = np.zeros(input_shape_global)
mask = np.zeros(input_shape_global)

angular_velocity_x = np.zeros(input_shape_global)
angular_velocity_y = np.zeros(input_shape_global)
angular_velocity_z = np.zeros(input_shape_global)

mask_particle_group_1 = np.zeros(input_shape_global)
mask_particle_group_2 = np.zeros(input_shape_global)
mask_particle_group_3 = np.zeros(input_shape_global)

angular_x = np.zeros(input_shape_global)
angular_y = np.zeros(input_shape_global)
angular_z = np.zeros(input_shape_global)
i, j ,k= np.meshgrid(np.arange(1, half_domain_size_x), np.arange(1, half_domain_size_y),np.arange(1, half_domain_size_z))
x_grid[0, 0, k*2, j*2, i*2] = i * cell_size * 2
y_grid[0, 0, k*2, j*2, i*2] = j * cell_size * 2
z_grid[0, 0, k*2, j*2, i*2] = k * cell_size * 2
vx_grid[0, 0, k*2, j*2, i*2] = 0
vy_grid[0, 0, k*2, j*2, i*2] = 0
vz_grid[0, 0, k*2, j*2, i*2] = random.uniform(-1.0,1.0)*0.00001
mask[0, 0, k*2, j*2, i*2] = 1
distance_between_particle_and_circle_center = np.sqrt((x_grid-(Mixer_radius-1)*cell_size)**2 + (y_grid-(Mixer_radius-1)*cell_size)**2)
distance_between_particle_and_circle_center = torch.tensor(distance_between_particle_and_circle_center, device=device)
mask = np.where(
    torch.lt(distance_between_particle_and_circle_center,(Mixer_radius-5)*cell_size)&
    torch.gt(distance_between_particle_and_circle_center,10*cell_size),    
    torch.tensor(1, device=device), 
    torch.tensor(0.0, device=device))
mask_particle_group_2 = np.where(
    torch.gt(distance_between_particle_and_circle_center, (Mixer_radius - 8) * cell_size), 
    torch.tensor(1, device=device), 
    torch.tensor(0.0, device=device))

mask_particle_group_3 = np.where(
    torch.gt(distance_between_particle_and_circle_center, (Mixer_radius - 8) * cell_size)&
    torch.lt(distance_between_particle_and_circle_center, (Mixer_radius - 5) * cell_size), 
    torch.tensor(1, device=device), 
    torch.tensor(0.0, device=device))

mask_particle_group_1 = mask.copy()

i, j ,k= np.meshgrid(np.arange(1, half_domain_size_x), np.arange(1, half_domain_size_y), k_values)
mask_particle_group_2[0, 0, k*2, j*2, i*2] = 1
mask_particle_group_1[mask_particle_group_2 == 1] = 0
angular_velocity_z[mask_particle_group_2 == 1] = Mixer_anuglar_velocity_z * Mixer_radius
print('Number of particles:', np.count_nonzero(mask))
device = torch.device("cuda" if is_gpu else "cpu")
mask = torch.from_numpy(mask).float().to(device)
x_grid = torch.from_numpy(x_grid).float().to(device)
y_grid = torch.from_numpy(y_grid).float().to(device)
z_grid = torch.from_numpy(z_grid).float().to(device)
vx_grid = torch.from_numpy(vx_grid).float().to(device)
vy_grid = torch.from_numpy(vy_grid).float().to(device)
vz_grid = torch.from_numpy(vz_grid).float().to(device)
mask_particle_group_1 = torch.from_numpy(mask_particle_group_1).float().to(device)
mask_particle_group_2 = torch.from_numpy(mask_particle_group_2).float().to(device)
mask_particle_group_3 = torch.from_numpy(mask_particle_group_3).float().to(device)

angular_velocity_x = torch.from_numpy(angular_velocity_x).float().to(device)
angular_velocity_y = torch.from_numpy(angular_velocity_y).float().to(device)
angular_velocity_z = torch.from_numpy(angular_velocity_z).float().to(device)

angular_x = torch.from_numpy(angular_x).float().to(device)
angular_y = torch.from_numpy(angular_y).float().to(device)
angular_z = torch.from_numpy(angular_z).float().to(device)
'''
x_grid = torch.load('initialization_results/x_grid.pt')
y_grid = torch.load('initialization_results/y_grid.pt')
z_grid = torch.load('initialization_results/z_grid.pt')

vx_grid = torch.load('initialization_results/vx_grid.pt')
vy_grid = torch.load('initialization_results/vy_grid.pt')
vz_grid = torch.load('initialization_results/vz_grid.pt')

angular_velocity_x= torch.load('initialization_results/angular_velocity_x.pt')
angular_velocity_y= torch.load('initialization_results/angular_velocity_y.pt')
angular_velocity_z= torch.load('initialization_results/angular_velocity_z.pt')

mask_particle_group_1 = torch.load('initialization_results/mask_particle_group_1.pt')
mask_particle_group_2 = torch.load('initialization_results/mask_particle_group_2.pt')
mask_particle_group_3 = torch.load('initialization_results/mask_particle_group_3.pt')
mask  = torch.load('initialization_results/mask.pt')


device = torch.device("cuda" if is_gpu else "cpu")

angular_x = torch.zeros(input_shape_global,device=device)
angular_y = torch.zeros(input_shape_global,device=device)
angular_z = torch.zeros(input_shape_global,device=device)

compressed_x_grid = x_grid[mask!= 0]
compressed_y_grid = y_grid[mask!= 0]
compressed_z_grid = z_grid[mask!= 0]
compressed_vx_grid = vx_grid[mask!= 0]
compressed_vy_grid = vy_grid[mask!= 0]
compressed_vz_grid = vz_grid[mask!= 0]
compressed_angular_velocity_x  = angular_velocity_x[mask!= 0]
compressed_angular_velocity_y  = angular_velocity_y[mask!= 0]
compressed_angular_velocity_z  = angular_velocity_z[mask!= 0]
compressed_angular_x  = angular_x[mask!= 0]
compressed_angular_y  = angular_y[mask!= 0]
compressed_angular_z  = angular_z[mask!= 0]
compressed_mask_particle_group_1 = mask_particle_group_1[[mask!= 0]]
compressed_mask_particle_group_3 = mask_particle_group_3[[mask!= 0]]
particle_number = len(compressed_x_grid)
particle_inertia = (2/5) * particle_mass * cell_size**2
zeros = torch.zeros(particle_number, device=device) 
eplis = torch.ones(particle_number, device=device) * 1e-04
ones  = torch.ones(particle_number, device=device) 
# Define the AI4DEM model
class AI4DEM(nn.Module):
    """AI4DEM model for particle interaction detection and force calculation"""
    def __init__(self):
        super(AI4DEM, self).__init__()

    def detector(self, grid, i, j, k):
        """Detect neighboring particles and calculate the distance between them"""
        diff = grid - torch.roll(grid, shifts=(k - 2, j - 2, i - 2), dims=(2, 3, 4))
        return diff
    def detector_angular_velocity(self, grid, i, j, k):
        """Detect neighboring particles and calculate the distance between them"""
        diff = grid + torch.roll(grid, shifts=(k - 2, j - 2, i - 2), dims=(2, 3, 4))
        return diff
    
    def forward(self, compressed_x_grid, compressed_y_grid, compressed_z_grid, compressed_vx_grid, compressed_vy_grid, compressed_vz_grid, compressed_angular_velocity_x, compressed_angular_velocity_y, compressed_angular_velocity_z, compressed_angular_x, compressed_angular_y, compressed_angular_z, d, kn, damping_coefficient_Eta, friction_coefficient, dt, input_shape, filter_size):
        cell_xold = torch.round(compressed_x_grid / d).long()
        cell_yold = torch.round(compressed_y_grid / d).long()
        cell_zold = torch.round(compressed_z_grid / d).long()
        mask.fill_(0)
        x_grid.fill_(0)
        y_grid.fill_(0)
        z_grid.fill_(0)
        
        vx_grid.fill_(0)
        vy_grid.fill_(0)
        vz_grid.fill_(0)
        
        angular_velocity_x.fill_(0)
        angular_velocity_y.fill_(0)
        angular_velocity_z.fill_(0)
        
        mask[0,0,cell_zold,cell_yold,cell_xold]=1

        compressed_vx_grid[compressed_mask_particle_group_3!=0]= - Mixer_anuglar_velocity_z*(compressed_y_grid[compressed_mask_particle_group_3!=0]-(Mixer_radius-1)*cell_size)
        compressed_vy_grid[compressed_mask_particle_group_3!=0]=   Mixer_anuglar_velocity_z*(compressed_x_grid[compressed_mask_particle_group_3!=0]-(Mixer_radius-1)*cell_size)
        compressed_vz_grid[compressed_mask_particle_group_3!=0]=   0
        
        compressed_x_grid[compressed_mask_particle_group_3!=0] = compressed_x_grid[compressed_mask_particle_group_3!=0] + dt*compressed_vx_grid[compressed_mask_particle_group_3!=0] 
        compressed_y_grid[compressed_mask_particle_group_3!=0] = compressed_y_grid[compressed_mask_particle_group_3!=0] + dt*compressed_vy_grid[compressed_mask_particle_group_3!=0] 
        compressed_z_grid[compressed_mask_particle_group_3!=0] = compressed_z_grid[compressed_mask_particle_group_3!=0] + dt*compressed_vz_grid[compressed_mask_particle_group_3!=0]          
        
        x_grid[0,0,cell_zold,cell_yold,cell_xold] = compressed_x_grid
        y_grid[0,0,cell_zold,cell_yold,cell_xold] = compressed_y_grid
        z_grid[0,0,cell_zold,cell_yold,cell_xold] = compressed_z_grid

        vx_grid[0,0,cell_zold,cell_yold,cell_xold] = compressed_vx_grid
        vy_grid[0,0,cell_zold,cell_yold,cell_xold] = compressed_vy_grid
        vz_grid[0,0,cell_zold,cell_yold,cell_xold] = compressed_vz_grid
        
        angular_velocity_x[0,0,cell_zold,cell_yold,cell_xold] = compressed_angular_velocity_x
        angular_velocity_y[0,0,cell_zold,cell_yold,cell_xold] = compressed_angular_velocity_y
        angular_velocity_z[0,0,cell_zold,cell_yold,cell_xold] = compressed_angular_velocity_z
        
        particle_number = len(mask[mask!=0])        
        fx_grid_collision = torch.zeros(particle_number, device=device)
        fy_grid_collision = torch.zeros(particle_number, device=device)
        fz_grid_collision = torch.zeros(particle_number, device=device)

        fx_grid_damping = torch.zeros(particle_number, device=device)
        fy_grid_damping = torch.zeros(particle_number, device=device)
        fz_grid_damping = torch.zeros(particle_number, device=device)

        fx_grid_friction = torch.zeros(particle_number, device=device)
        fy_grid_friction = torch.zeros(particle_number, device=device)
        fz_grid_friction = torch.zeros(particle_number, device=device)       

        collision_torque_x = torch.zeros(particle_number, device=device)
        collision_torque_y = torch.zeros(particle_number, device=device)
        collision_torque_z = torch.zeros(particle_number, device=device)
        for i in range(filter_size):
            for j in range(filter_size):
                for k in range(filter_size):
                    # calculate distance between the two particles
                    diffx = self.detector(x_grid, i, j, k) # individual
                    diffy = self.detector(y_grid, i, j, k) # individual
                    diffz = self.detector(z_grid, i, j, k) # individual
                    
                    # calculate nodal velocity difference between the two particles
                    diffvx = self.detector(vx_grid, i, j, k) # individual
                    diffvy = self.detector(vy_grid, i, j, k) # individual
                    diffvz = self.detector(vz_grid, i, j, k) # individual

                    diff_angular_velocity_x = self.detector_angular_velocity(angular_velocity_x, i, j, k) # individual
                    diff_angular_velocity_y = self.detector_angular_velocity(angular_velocity_y, i, j, k) # individual
                    diff_angular_velocity_z = self.detector_angular_velocity(angular_velocity_z, i, j, k) # individual

                    compressed_diffx=diffx[0,0,cell_zold,cell_yold,cell_xold]
                    compressed_diffy=diffy[0,0,cell_zold,cell_yold,cell_xold]
                    compressed_diffz=diffz[0,0,cell_zold,cell_yold,cell_xold]
                    compressed_dist = torch.sqrt(compressed_diffx**2 + compressed_diffy**2 + compressed_diffz**2)  
                    
                    compressed_diffvx=diffvx[0,0,cell_zold,cell_yold,cell_xold]
                    compressed_diffvy=diffvy[0,0,cell_zold,cell_yold,cell_xold]
                    compressed_diffvz=diffvz[0,0,cell_zold,cell_yold,cell_xold]
                                       
                    compressed_diffvx_Vn =  compressed_diffvx * compressed_diffx /  torch.maximum(eplis, compressed_dist)
                    compressed_diffvy_Vn =  compressed_diffvy * compressed_diffy /  torch.maximum(eplis, compressed_dist)
                    compressed_diffvz_Vn =  compressed_diffvz * compressed_diffz /  torch.maximum(eplis, compressed_dist) 
                    compressed_diffv_Vn = compressed_diffvx_Vn + compressed_diffvy_Vn + compressed_diffvz_Vn
                    fn_grid_damping =  torch.where(torch.lt(compressed_dist, 2*d), - damping_coefficient_Eta * compressed_diffv_Vn, zeros)

                    compressed_diff_angular_velocity_x = diff_angular_velocity_x[0,0,cell_zold,cell_yold,cell_xold]
                    compressed_diff_angular_velocity_y = diff_angular_velocity_y[0,0,cell_zold,cell_yold,cell_xold]
                    compressed_diff_angular_velocity_z = diff_angular_velocity_z[0,0,cell_zold,cell_yold,cell_xold]
                    
                    compressed_moment_arm_x = d * compressed_diffx / torch.maximum(eplis, compressed_dist) 
                    compressed_moment_arm_y = d * compressed_diffy / torch.maximum(eplis, compressed_dist)
                    compressed_moment_arm_z = d * compressed_diffz / torch.maximum(eplis, compressed_dist)
                    
                    compressed_diffv_Vn_x =  compressed_diffv_Vn * compressed_diffx /  torch.maximum(eplis, compressed_dist)
                    compressed_diffv_Vn_y =  compressed_diffv_Vn * compressed_diffy /  torch.maximum(eplis, compressed_dist)
                    compressed_diffv_Vn_z =  compressed_diffv_Vn * compressed_diffz /  torch.maximum(eplis, compressed_dist)                     
                    
                    compressed_vx_angular = compressed_diff_angular_velocity_y * compressed_moment_arm_z - compressed_diff_angular_velocity_z * compressed_moment_arm_y
                    compressed_vy_angular = compressed_diff_angular_velocity_z * compressed_moment_arm_x - compressed_diff_angular_velocity_x * compressed_moment_arm_z
                    compressed_vz_angular = compressed_diff_angular_velocity_x * compressed_moment_arm_y - compressed_diff_angular_velocity_y * compressed_moment_arm_x 

                    # calculate collision force between the two particles
                    fx_grid_collision =  fx_grid_collision + torch.where(torch.lt(compressed_dist,2 * d), kn * (compressed_dist - 2 * d ) * compressed_diffx / torch.maximum(eplis, compressed_dist), zeros) # individual
                    fy_grid_collision =  fy_grid_collision + torch.where(torch.lt(compressed_dist,2 * d), kn * (compressed_dist - 2 * d ) * compressed_diffy / torch.maximum(eplis, compressed_dist), zeros) # individual
                    fz_grid_collision =  fz_grid_collision + torch.where(torch.lt(compressed_dist,2 * d), kn * (compressed_dist - 2 * d ) * compressed_diffz / torch.maximum(eplis, compressed_dist), zeros) # individual            
                    
                    fn_grid_collision = torch.where(torch.lt(compressed_dist,2*d), kn * (compressed_dist - 2 * d ) , zeros)
                    fn_grid =torch.abs(fn_grid_collision + fn_grid_damping)
                    
                    # calculate the damping force between the two particles
                    fx_grid_damping =  fx_grid_damping + torch.where(torch.lt(compressed_dist, 2*d), damping_coefficient_Eta * compressed_diffv_Vn_x, zeros) # individual   
                    fy_grid_damping =  fy_grid_damping + torch.where(torch.lt(compressed_dist, 2*d), damping_coefficient_Eta * compressed_diffv_Vn_y, zeros) # individual 
                    fz_grid_damping =  fz_grid_damping + torch.where(torch.lt(compressed_dist, 2*d), damping_coefficient_Eta * compressed_diffv_Vn_z, zeros) # individual 
                    
                    compressed_diffv_Vt_x = (compressed_diffvx) * (compressed_diffy**2 + compressed_diffz**2) / torch.maximum(eplis, compressed_dist**2)  + compressed_vx_angular
                    compressed_diffv_Vt_y = (compressed_diffvy) * (compressed_diffx**2 + compressed_diffz**2) / torch.maximum(eplis, compressed_dist**2)  + compressed_vy_angular
                    compressed_diffv_Vt_z = (compressed_diffvz) * (compressed_diffx**2 + compressed_diffy**2) / torch.maximum(eplis, compressed_dist**2)  + compressed_vz_angular
                    
                    compressed_diffv_Vt = torch.sqrt (compressed_diffv_Vt_x**2 + compressed_diffv_Vt_y**2 + compressed_diffv_Vt_z**2)
                    
                    fx_grid_friction =  fx_grid_friction - torch.where(torch.lt(compressed_dist, 2*d), friction_coefficient * fn_grid * compressed_diffv_Vt_x /  torch.maximum(eplis, compressed_diffv_Vt), zeros) # fx_grid_friction - torch.where(torch.lt(dist,2*d), friction_coefficient * fn_grid * diffv_Vt_x / torch.maximum(eplis,diffv_Vt), zeros)
                    fy_grid_friction =  fy_grid_friction - torch.where(torch.lt(compressed_dist, 2*d), friction_coefficient * fn_grid * compressed_diffv_Vt_y /  torch.maximum(eplis, compressed_diffv_Vt), zeros)
                    fz_grid_friction =  fz_grid_friction - torch.where(torch.lt(compressed_dist, 2*d), friction_coefficient * fn_grid * compressed_diffv_Vt_z /  torch.maximum(eplis, compressed_diffv_Vt), zeros)

                    collision_torque_x = collision_torque_x + torch.where(torch.lt(compressed_dist, 2*d), fz_grid_friction * compressed_moment_arm_y  - fy_grid_friction * compressed_moment_arm_z , zeros)
                    collision_torque_y = collision_torque_y + torch.where(torch.lt(compressed_dist, 2*d), fx_grid_friction * compressed_moment_arm_z  - fz_grid_friction * compressed_moment_arm_x , zeros)
                    collision_torque_z = collision_torque_z + torch.where(torch.lt(compressed_dist, 2*d), fy_grid_friction * compressed_moment_arm_x  - fx_grid_friction * compressed_moment_arm_y , zeros)

        del diffx, diffy, diffz, compressed_diffvx_Vn, compressed_diffvy_Vn, compressed_diffvz_Vn, compressed_diffv_Vn, compressed_diffv_Vn_x, compressed_diffv_Vn_y, compressed_diffv_Vn_z
        compressed_is_round_boundary_overlap = torch.gt((compressed_x_grid-(Mixer_radius-1)*d)**2 + (compressed_y_grid-(Mixer_radius-1)*d)**2, ((Mixer_radius-1)*d)**2)
        compressed_is_forward_overlap  = torch.ne(compressed_z_grid, 0.0000) & torch.lt(compressed_z_grid, d) # Overlap with bottom wall
        compressed_is_backward_overlap = torch.gt(compressed_z_grid, domain_size_z*cell_size-2*d ) # Overlap with bottom wall 
        
        f_grid_round_boundary  = torch.where(compressed_is_round_boundary_overlap, ones, zeros) * (torch.sqrt((compressed_x_grid-(Mixer_radius-1)*d)**2 + (compressed_y_grid-(Mixer_radius-1)*d)**2)-(Mixer_radius-1)*d)
        fx_grid_round_boundary = 0# kn * (Mixer_radius*d-compressed_x_grid) / (Mixer_radius*d) * torch.where(compressed_is_round_boundary_overlap, ones, zeros) * (torch.sqrt((compressed_x_grid-(Mixer_radius-1)*d)**2 + (compressed_y_grid-(Mixer_radius-1)*d)**2)-(Mixer_radius-1)*d)
        fy_grid_round_boundary = 0# kn * (Mixer_radius*d-compressed_y_grid) / (Mixer_radius*d) * torch.where(compressed_is_round_boundary_overlap, ones, zeros) * (torch.sqrt((compressed_x_grid-(Mixer_radius-1)*d)**2 + (compressed_y_grid-(Mixer_radius-1)*d)**2)-(Mixer_radius-1)*d)
        fz_grid_boundary_forward  = kn * torch.where(compressed_is_forward_overlap, ones, zeros) * (d - compressed_z_grid)
        fz_grid_boundary_backward = kn * torch.where(compressed_is_backward_overlap,ones, zeros) * (compressed_z_grid - domain_size_z*cell_size + 2*d)

        # calculate damping force from the boundaries        
        fz_grid_forward_damping   = damping_coefficient_Eta_wall * compressed_vz_grid *torch.where(compressed_is_forward_overlap, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
        fz_grid_backward_damping  = damping_coefficient_Eta_wall * compressed_vz_grid *torch.where(compressed_is_backward_overlap,torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))

        # calculate friction force from the boundaries
        fz_grid_forward_friction_x   = 0# - torch.where(compressed_is_forward_overlap, friction_coefficient * torch.abs (  fz_grid_boundary_forward  - fz_grid_forward_damping )   * (compressed_vx_grid - d*compressed_angular_velocity_y) / torch.maximum(eplis, torch.sqrt((compressed_vx_grid - d*compressed_angular_velocity_y)**2+(compressed_vy_grid + d*compressed_angular_velocity_x)**2+compressed_vz_grid**2)), torch.tensor(0.0, device=device))
        fz_grid_forward_friction_y   = 0# - torch.where(compressed_is_forward_overlap, friction_coefficient * torch.abs (  fz_grid_boundary_forward  - fz_grid_forward_damping )   * (compressed_vy_grid + d*compressed_angular_velocity_x) / torch.maximum(eplis, torch.sqrt((compressed_vx_grid - d*compressed_angular_velocity_y)**2+(compressed_vy_grid + d*compressed_angular_velocity_x)**2+compressed_vz_grid**2)), torch.tensor(0.0, device=device))
        
        fz_grid_backward_friction_x  =  0#- torch.where(compressed_is_backward_overlap,friction_coefficient * torch.abs (- fz_grid_boundary_backward - fz_grid_backward_damping)   * (compressed_vx_grid + d*compressed_angular_velocity_y) / torch.maximum(eplis, torch.sqrt((compressed_vx_grid + d*compressed_angular_velocity_y)**2+(compressed_vy_grid - d*compressed_angular_velocity_x)**2+compressed_vz_grid**2)), torch.tensor(0.0, device=device))
        fz_grid_backward_friction_y  =  0#- torch.where(compressed_is_backward_overlap,friction_coefficient * torch.abs (- fz_grid_boundary_backward - fz_grid_backward_damping)   * (compressed_vy_grid - d*compressed_angular_velocity_x) / torch.maximum(eplis, torch.sqrt((compressed_vx_grid + d*compressed_angular_velocity_y)**2+(compressed_vy_grid - d*compressed_angular_velocity_x)**2+compressed_vz_grid**2)), torch.tensor(0.0, device=device))
        
        angle_sin_theta = (compressed_x_grid-Mixer_radius*d)/(Mixer_radius*d)
        angle_cos_theta = (compressed_y_grid-Mixer_radius*d)/(Mixer_radius*d)
        
        compressed_Mixer_Vn   = compressed_vy_grid  * angle_cos_theta + compressed_vx_grid * angle_sin_theta
        compressed_Mixer_Vn_x = compressed_Mixer_Vn * angle_sin_theta
        compressed_Mixer_Vn_y = compressed_Mixer_Vn * angle_cos_theta
        
        compressed_Mixer_damping   = - 0#torch.where(compressed_is_round_boundary_overlap,    damping_coefficient_Eta_wall * compressed_Mixer_Vn,   torch.tensor(0.0, device=device))
        compressed_Mixer_damping_x = - 0#torch.where(compressed_is_round_boundary_overlap,    damping_coefficient_Eta_wall * compressed_Mixer_Vn_x, torch.tensor(0.0, device=device))
        compressed_Mixer_damping_y = - 0#torch.where(compressed_is_round_boundary_overlap,    damping_coefficient_Eta_wall * compressed_Mixer_Vn_y, torch.tensor(0.0, device=device))

        compressed_Mixer_Vt   =  torch.sqrt((compressed_vz_grid + compressed_angular_velocity_x * d * angle_cos_theta - compressed_angular_velocity_y * d * angle_sin_theta)**2+( (compressed_angular_velocity_z-Mixer_anuglar_velocity_z*Mixer_radius) * d + compressed_vy_grid * angle_sin_theta - compressed_vx_grid * angle_sin_theta)**2)
        compressed_Mixer_Vt_x = - angle_cos_theta * ((compressed_angular_velocity_z-Mixer_anuglar_velocity_z*Mixer_radius) * d + compressed_vy_grid * angle_sin_theta - compressed_vx_grid * angle_cos_theta )
        compressed_Mixer_Vt_y = + angle_sin_theta * ((compressed_angular_velocity_z-Mixer_anuglar_velocity_z*Mixer_radius) * d + compressed_vy_grid * angle_sin_theta - compressed_vx_grid * angle_cos_theta )
        compressed_Mixer_Vt_z = compressed_vz_grid + compressed_angular_velocity_x *d * angle_cos_theta - compressed_angular_velocity_y * d * angle_sin_theta

        compressed_Mixer_Friction    = -0# torch.where(compressed_is_round_boundary_overlap,    friction_coefficient_wall_to_sphere * torch.abs ( f_grid_round_boundary + compressed_Mixer_damping ), torch.tensor(0.0, device=device))
        compressed_Mixer_Friction_x  =  0#torch.where(compressed_is_round_boundary_overlap,    compressed_Mixer_Friction * compressed_Mixer_Vt_x / compressed_Mixer_Vt , zeros)
        compressed_Mixer_Friction_y  =  0#torch.where(compressed_is_round_boundary_overlap,    compressed_Mixer_Friction * compressed_Mixer_Vt_y / compressed_Mixer_Vt , zeros)
        compressed_Mixer_Friction_z  =  0#torch.where(compressed_is_round_boundary_overlap,    compressed_Mixer_Friction * compressed_Mixer_Vt_z / compressed_Mixer_Vt , zeros)
        
        ''''''
        torque_x = d * (   compressed_Mixer_Friction_z * angle_cos_theta  + fz_grid_forward_friction_y   -  fz_grid_backward_friction_y) + collision_torque_x
        torque_y = d * ( - compressed_Mixer_Friction_z * angle_sin_theta  + fz_grid_backward_friction_x  -  fz_grid_forward_friction_x ) + collision_torque_y
        torque_z = d * (   compressed_Mixer_Friction_y * angle_sin_theta  - compressed_Mixer_Friction_x * angle_cos_theta  ) + collision_torque_z
        
        # calculate the new velocity with acceleration calculated by forces
        compressed_vx_grid = compressed_mask_particle_group_1 *compressed_vx_grid   + compressed_mask_particle_group_1 *  (dt / particle_mass) * ( - 0   * particle_mass)  +  compressed_mask_particle_group_1 *  (dt / particle_mass) * (- fx_grid_collision - fx_grid_damping + fx_grid_friction + fx_grid_round_boundary    +  compressed_Mixer_damping_x                                                    + compressed_Mixer_Friction_x + fz_grid_forward_friction_x + fz_grid_backward_friction_x) 
        compressed_vy_grid = compressed_mask_particle_group_1 *compressed_vy_grid   + compressed_mask_particle_group_1 *  (dt / particle_mass) * ( - 9.8 * particle_mass)  +  compressed_mask_particle_group_1 *  (dt / particle_mass) * (- fy_grid_collision - fy_grid_damping + fy_grid_friction + fy_grid_round_boundary    +  compressed_Mixer_damping_y                                                    + compressed_Mixer_Friction_y + fz_grid_forward_friction_y + fz_grid_backward_friction_y)  
        compressed_vz_grid = compressed_mask_particle_group_1 *compressed_vz_grid   + compressed_mask_particle_group_1 *  (dt / particle_mass) * ( - 0   * particle_mass)  +  compressed_mask_particle_group_1 *  (dt / particle_mass) * (- fz_grid_collision - fz_grid_damping + fz_grid_friction - fz_grid_boundary_backward +  fz_grid_boundary_forward - fz_grid_forward_damping - fz_grid_backward_damping + compressed_Mixer_Friction_z) 
        
        # Update particle coordniates
        compressed_x_grid  = compressed_x_grid + compressed_mask_particle_group_1 * dt * compressed_vx_grid
        compressed_y_grid  = compressed_y_grid + compressed_mask_particle_group_1 * dt * compressed_vy_grid
        compressed_z_grid  = compressed_z_grid + compressed_mask_particle_group_1 * dt * compressed_vz_grid

        compressed_angular_velocity_x =   compressed_mask_particle_group_1 *compressed_angular_velocity_x + compressed_mask_particle_group_1 *  torque_x * dt / particle_inertia
        compressed_angular_velocity_y =   compressed_mask_particle_group_1 *compressed_angular_velocity_y + compressed_mask_particle_group_1 *  torque_y * dt / particle_inertia
        compressed_angular_velocity_z =  compressed_angular_velocity_z + compressed_mask_particle_group_1 *  torque_z * dt / particle_inertia
        
        compressed_angular_x = compressed_angular_x + compressed_mask_particle_group_1 * compressed_angular_velocity_x * dt
        compressed_angular_y = compressed_angular_y + compressed_mask_particle_group_1 * compressed_angular_velocity_y * dt
        compressed_angular_z = compressed_angular_z + compressed_mask_particle_group_1 * compressed_angular_velocity_z * dt
        return compressed_x_grid, compressed_y_grid, compressed_z_grid, compressed_vx_grid, compressed_vy_grid, compressed_vz_grid, compressed_angular_velocity_x, compressed_angular_velocity_y, compressed_angular_velocity_z, compressed_angular_x, compressed_angular_y, compressed_angular_z, (fx_grid_collision+fx_grid_damping), (fy_grid_collision+fy_grid_damping), (fz_grid_collision+fz_grid_damping)
model = AI4DEM().to(device)
# model = torch.compile(model, mode="reduce-overhead")
# Module 2: Contact detection and force calculation
t = 0
dt = 0.00001 # 0.0001
ntime = 300000000000000
# Convert np.array into torch.tensor and transfer it to GPU
filter_size = 5 

# Initialize tensors
eplis_global = torch.ones(input_shape_global, device=device) * 1e-04

# Main simulation loop
start = time.time()
with torch.no_grad():
    for itime in range(1, ntime + 1):
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            [compressed_x_grid, compressed_y_grid, compressed_z_grid, compressed_vx_grid, compressed_vy_grid, compressed_vz_grid, compressed_angular_velocity_x, compressed_angular_velocity_y, compressed_angular_velocity_z, compressed_angular_x, compressed_angular_y, compressed_angular_z, Fx, Fy, Fz] = model(compressed_x_grid, compressed_y_grid, compressed_z_grid, compressed_vx_grid, compressed_vy_grid, compressed_vz_grid, compressed_angular_velocity_x, compressed_angular_velocity_y, compressed_angular_velocity_z, compressed_angular_x, compressed_angular_y, compressed_angular_z, cell_size, kn, damping_coefficient_Eta, friction_coefficient, dt, input_shape_global, filter_size)
            print('Time step:', itime, 'Number of particles:', torch.count_nonzero(compressed_x_grid).item()) 
            if itime % 2000 == 0:
                save_name = "packing_results/"+str(itime)
                torch.save(compressed_x_grid, save_name + 'compressed_x_grid.pt')
                torch.save(compressed_y_grid, save_name + 'compressed_y_grid.pt')
                torch.save(compressed_z_grid, save_name + 'compressed_z_grid.pt')
                torch.save(compressed_vx_grid, save_name + 'compressed_vx_grid.pt')
                torch.save(compressed_vy_grid, save_name + 'compressed_vy_grid.pt')
                torch.save(compressed_vz_grid, save_name + 'compressed_vz_grid.pt')
                torch.save(mask, save_name + 'mask.pt')
                torch.save(compressed_mask_particle_group_1, save_name + 'compressed_mask_particle_group_1.pt')
                torch.save(compressed_mask_particle_group_3, save_name + 'compressed_mask_particle_group_3.pt')
                torch.save(mask_particle_group_2, save_name + 'mask_particle_group_2.pt')
                
            if itime % 2000== 0:
                # 创建散点图
                plt.figure(figsize=(8, 6))  # 设置图形大小
                xp=compressed_x_grid*compressed_mask_particle_group_1
                yp=compressed_y_grid*compressed_mask_particle_group_1
                c_color=compressed_angular_velocity_z*compressed_mask_particle_group_1
                sc = plt.scatter(
                xp.cpu().numpy(), 
                yp.cpu().numpy(), 
                c=c_color.cpu().numpy(), 
                cmap='viridis', 
                s=S_graph,  # Ensure S_graph is also a valid NumPy array or scalar
                edgecolor='k',
                vmax = 100,
                vmin = -100)                # 添加颜色条
                cbar = plt.colorbar(sc)
                cbar.set_label('angular velocity z (rad/s)')
                # 设置坐标轴标签
                plt.xlabel('X-axis (m)')
                plt.ylabel('Y-axis (m)')
                # 设置坐标轴范围
                plt.xlim([-1*cell_size, (domain_size_x-1)*cell_size])
                plt.ylim([-1*cell_size, (domain_size_y-1)*cell_size])
                # 创建并添加一个圆
                circle_center = ((Mixer_radius-1)*cell_size, (Mixer_radius-1)*cell_size)  # 圆心坐标 (根据你的数据调整)
                circle_radius = Mixer_radius*cell_size        # 圆的半径
                circle = patches.Circle(circle_center, circle_radius, color='red', fill=False, linewidth=2)
                ax = plt.gca()
                ax.add_patch(circle)
                # 设置图形标题
                plt.title('Augular velocity of rotary drum is 50 rad/s')
                # 保存图像
                save_name = "3D_new/"+str(itime)+".jpg"
                plt.savefig(save_name, dpi=1000, bbox_inches='tight')              
                
end = time.time()
print('Elapsed time:', end - start)
