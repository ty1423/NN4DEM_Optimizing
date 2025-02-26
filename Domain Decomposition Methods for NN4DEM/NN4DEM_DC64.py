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

# Check if GPU is available 
is_gpu = torch.cuda.is_available()
device = torch.device("cuda" if is_gpu else "cpu")
print("Using GPU:", is_gpu)
# # Function to generate a structured grid
# def create_grid(domain_size, cell_size):
#     num_cells = int(domain_size / cell_size)
#     grid = np.zeros((num_cells, num_cells, num_cells), dtype=int)
#     return grid

# Input Parameters 
domain_size_x = 50 # Direction x size of the domain 
domain_size_y = 50 # Direction y size of the domain 
domain_size_z = 50 # Direction z size of the domain 
half_domain_size_x = int(domain_size_x/2) # half of direction x size of the domain 
half_domain_size_y = int(domain_size_y/2) # half of direction y size of the domain 
half_domain_size_z = int(domain_size_z/2)  # half of direction z size of the domain 
quarter_domain_size_z       = int(domain_size_z/4)
one_eighth_domain_size_z    = int(domain_size_z/8) 
one_sixteenth_domain_size_z = int(domain_size_z/16)
# domain_size = 1500  # Size of the square domain
# domain_height = 100
# half_domain_size = int(domain_size/2)+1
cell_size =0.05   # Cell size and particle radius
simulation_time = 1
kn = 600000#0#000  # Normal stiffness of the spring
dn = 0.5  # Normal damping coefficient
rho_p = 2700 
particle_mass = 4/3*3.1416*cell_size**3*rho_p #4188.7902
particle_inertia = (2/5) * particle_mass * cell_size**2
rolling_friction_coefficient = 0.02

K_graph = 57*1000000*1
S_graph = K_graph * (cell_size / domain_size_x) ** 2
restitution_coefficient = 0.5  # coefficient of restitution
friction_coefficient = 0.5  # coefficient of friction
# coefficient of damping calculation 
# Discrete particle simulation of two-dimensional fluidized bed
damping_coefficient_Alpha      = -1 * math.log(restitution_coefficient)/math.pi
damping_coefficient_Gamma      = damping_coefficient_Alpha/math.sqrt(damping_coefficient_Alpha**2+1)
damping_coefficient_Eta        = 2 * damping_coefficient_Gamma * math.sqrt(kn * particle_mass/2)
damping_coefficient_Eta_wall   = 2 * damping_coefficient_Gamma * math.sqrt(kn * particle_mass)

print('Damping Coefficient:', damping_coefficient_Eta)

# Module 1: Domain discretisation and initial particle insertion
# Create grid
# grid = create_grid(domain_size, cell_size)
# grid_shape = grid.shape

# input shape for the whole domain
input_shape_global = (1, 1, domain_size_z, domain_size_y, domain_size_x)

# input shape for each bigger domain
input_shape_domain = (1, 1, one_sixteenth_domain_size_z+4, half_domain_size_y+2, half_domain_size_x+2)

# Generate particles
# npt = int(domain_size ** 3)
# define the 3D coordinate and velocity grid for the whole domain
total_x_grid = np.zeros(input_shape_global)
total_y_grid = np.zeros(input_shape_global)
total_z_grid = np.zeros(input_shape_global)

total_vx_grid = np.zeros(input_shape_global)
total_vy_grid = np.zeros(input_shape_global)
total_vz_grid = np.zeros(input_shape_global)
total_mask = np.zeros(input_shape_global)

# define the 3D coordinate, mask and velocity grid for the each bigger domain
x_grid = np.zeros(input_shape_domain)
y_grid = np.zeros(input_shape_domain)
z_grid = np.zeros(input_shape_domain)

vx_grid = np.zeros(input_shape_domain)
vy_grid = np.zeros(input_shape_domain)
vz_grid = np.zeros(input_shape_domain)
mask_grid = np.zeros(input_shape_domain)

x_decomposition_boundary_left_1    = torch.tensor([-1, half_domain_size_x-1-2, -1, half_domain_size_x-1-2,
                                                   -1, half_domain_size_x-1-2, -1, half_domain_size_x-1-2, 
                                                   -1, half_domain_size_x-1-2, -1, half_domain_size_x-1-2,
                                                   -1, half_domain_size_x-1-2, -1, half_domain_size_x-1-2,
                                                   -1, half_domain_size_x-1-2, -1, half_domain_size_x-1-2,
                                                   -1, half_domain_size_x-1-2, -1, half_domain_size_x-1-2,
                                                   -1, half_domain_size_x-1-2, -1, half_domain_size_x-1-2,
                                                   -1, half_domain_size_x-1-2, -1, half_domain_size_x-1-2,
                                                   -1, half_domain_size_x-1-2, -1, half_domain_size_x-1-2,
                                                   -1, half_domain_size_x-1-2, -1, half_domain_size_x-1-2,
                                                   -1, half_domain_size_x-1-2, -1, half_domain_size_x-1-2,
                                                   -1, half_domain_size_x-1-2, -1, half_domain_size_x-1-2,
                                                   -1, half_domain_size_x-1-2, -1, half_domain_size_x-1-2,
                                                   -1, half_domain_size_x-1-2, -1, half_domain_size_x-1-2,
                                                   -1, half_domain_size_x-1-2, -1, half_domain_size_x-1-2,
                                                   -1, half_domain_size_x-1-2, -1, half_domain_size_x-1-2 ])

x_decomposition_boundary_right_1   = torch.tensor([half_domain_size_x+2, domain_size_x, half_domain_size_x+2, domain_size_x,
                                                   half_domain_size_x+2, domain_size_x, half_domain_size_x+2, domain_size_x,
                                                   half_domain_size_x+2, domain_size_x, half_domain_size_x+2, domain_size_x, 
                                                   half_domain_size_x+2, domain_size_x, half_domain_size_x+2, domain_size_x,
                                                   half_domain_size_x+2, domain_size_x, half_domain_size_x+2, domain_size_x,
                                                   half_domain_size_x+2, domain_size_x, half_domain_size_x+2, domain_size_x, 
                                                   half_domain_size_x+2, domain_size_x, half_domain_size_x+2, domain_size_x, 
                                                   half_domain_size_x+2, domain_size_x, half_domain_size_x+2, domain_size_x,
                                                   half_domain_size_x+2, domain_size_x, half_domain_size_x+2, domain_size_x,
                                                   half_domain_size_x+2, domain_size_x, half_domain_size_x+2, domain_size_x, 
                                                   half_domain_size_x+2, domain_size_x, half_domain_size_x+2, domain_size_x, 
                                                   half_domain_size_x+2, domain_size_x, half_domain_size_x+2, domain_size_x,
                                                   half_domain_size_x+2, domain_size_x, half_domain_size_x+2, domain_size_x,
                                                   half_domain_size_x+2, domain_size_x, half_domain_size_x+2, domain_size_x, 
                                                   half_domain_size_x+2, domain_size_x, half_domain_size_x+2, domain_size_x, 
                                                   half_domain_size_x+2, domain_size_x, half_domain_size_x+2, domain_size_x])

y_decomposition_boundary_left_1    = torch.tensor([-1, -1, half_domain_size_y-1-2, half_domain_size_y-1-2,
                                                   -1, -1, half_domain_size_y-1-2, half_domain_size_y-1-2,
                                                   -1, -1, half_domain_size_y-1-2, half_domain_size_y-1-2,
                                                   -1, -1, half_domain_size_y-1-2, half_domain_size_y-1-2,
                                                   -1, -1, half_domain_size_y-1-2, half_domain_size_y-1-2,
                                                   -1, -1, half_domain_size_y-1-2, half_domain_size_y-1-2,
                                                   -1, -1, half_domain_size_y-1-2, half_domain_size_y-1-2,
                                                   -1, -1, half_domain_size_y-1-2, half_domain_size_y-1-2,
                                                   -1, -1, half_domain_size_y-1-2, half_domain_size_y-1-2,
                                                   -1, -1, half_domain_size_y-1-2, half_domain_size_y-1-2,
                                                   -1, -1, half_domain_size_y-1-2, half_domain_size_y-1-2,
                                                   -1, -1, half_domain_size_y-1-2, half_domain_size_y-1-2,
                                                   -1, -1, half_domain_size_y-1-2, half_domain_size_y-1-2,
                                                   -1, -1, half_domain_size_y-1-2, half_domain_size_y-1-2,
                                                   -1, -1, half_domain_size_y-1-2, half_domain_size_y-1-2,
                                                   -1, -1, half_domain_size_y-1-2, half_domain_size_y-1-2])

y_decomposition_boundary_right_1   = torch.tensor([half_domain_size_y+2, half_domain_size_y+2,domain_size_y,domain_size_y,
                                                   half_domain_size_y+2, half_domain_size_y+2,domain_size_y,domain_size_y,
                                                   half_domain_size_y+2, half_domain_size_y+2,domain_size_y,domain_size_y,
                                                   half_domain_size_y+2, half_domain_size_y+2,domain_size_y,domain_size_y,
                                                   half_domain_size_y+2, half_domain_size_y+2,domain_size_y,domain_size_y,
                                                   half_domain_size_y+2, half_domain_size_y+2,domain_size_y,domain_size_y,
                                                   half_domain_size_y+2, half_domain_size_y+2,domain_size_y,domain_size_y,
                                                   half_domain_size_y+2, half_domain_size_y+2,domain_size_y,domain_size_y,
                                                   half_domain_size_y+2, half_domain_size_y+2,domain_size_y,domain_size_y,
                                                   half_domain_size_y+2, half_domain_size_y+2,domain_size_y,domain_size_y,
                                                   half_domain_size_y+2, half_domain_size_y+2,domain_size_y,domain_size_y,
                                                   half_domain_size_y+2, half_domain_size_y+2,domain_size_y,domain_size_y,
                                                   half_domain_size_y+2, half_domain_size_y+2,domain_size_y,domain_size_y,
                                                   half_domain_size_y+2, half_domain_size_y+2,domain_size_y,domain_size_y,
                                                   half_domain_size_y+2, half_domain_size_y+2,domain_size_y,domain_size_y,
                                                   half_domain_size_y+2, half_domain_size_y+2,domain_size_y,domain_size_y])

z_decomposition_boundary_left_1    = torch.tensor([-1,-1,-1,-1,
                                                    1/16*domain_size_z-2-1, 1/16*domain_size_z-2-1, 1/16*domain_size_z-2-1, 1/16*domain_size_z-2-1,
                                                    2/16*domain_size_z-2-1, 2/16*domain_size_z-2-1, 2/16*domain_size_z-2-1, 2/16*domain_size_z-2-1,
                                                    3/16*domain_size_z-2-1, 3/16*domain_size_z-2-1, 3/16*domain_size_z-2-1, 3/16*domain_size_z-2-1,
                                                    4/16*domain_size_z-2-1, 4/16*domain_size_z-2-1, 4/16*domain_size_z-2-1, 4/16*domain_size_z-2-1,
                                                    5/16*domain_size_z-2-1, 5/16*domain_size_z-2-1, 5/16*domain_size_z-2-1, 5/16*domain_size_z-2-1,
                                                    6/16*domain_size_z-2-1, 6/16*domain_size_z-2-1, 6/16*domain_size_z-2-1, 6/16*domain_size_z-2-1,
                                                    7/16*domain_size_z-2-1, 7/16*domain_size_z-2-1, 7/16*domain_size_z-2-1, 7/16*domain_size_z-2-1,
                                                    8/16*domain_size_z-2-1, 8/16*domain_size_z-2-1, 8/16*domain_size_z-2-1, 8/16*domain_size_z-2-1,
                                                    9/16*domain_size_z-2-1, 9/16*domain_size_z-2-1, 9/16*domain_size_z-2-1, 9/16*domain_size_z-2-1,
                                                   10/16*domain_size_z-2-1,10/16*domain_size_z-2-1,10/16*domain_size_z-2-1,10/16*domain_size_z-2-1,
                                                   11/16*domain_size_z-2-1,11/16*domain_size_z-2-1,11/16*domain_size_z-2-1,11/16*domain_size_z-2-1,
                                                   12/16*domain_size_z-2-1,12/16*domain_size_z-2-1,12/16*domain_size_z-2-1,12/16*domain_size_z-2-1,
                                                   13/16*domain_size_z-2-1,13/16*domain_size_z-2-1,13/16*domain_size_z-2-1,13/16*domain_size_z-2-1,
                                                   14/16*domain_size_z-2-1,14/16*domain_size_z-2-1,14/16*domain_size_z-2-1,14/16*domain_size_z-2-1,
                                                   15/16*domain_size_z-2-1,15/16*domain_size_z-2-1,15/16*domain_size_z-2-1,15/16*domain_size_z-2-1])

z_decomposition_boundary_right_1   = torch.tensor([ 1/16*domain_size_z+2, 1/16*domain_size_z+2, 1/16*domain_size_z+2, 1/16*domain_size_z+2,
                                                    2/16*domain_size_z+2, 2/16*domain_size_z+2, 2/16*domain_size_z+2, 2/16*domain_size_z+2,
                                                    3/16*domain_size_z+2, 3/16*domain_size_z+2, 3/16*domain_size_z+2, 3/16*domain_size_z+2,
                                                    4/16*domain_size_z+2, 4/16*domain_size_z+2, 4/16*domain_size_z+2, 4/16*domain_size_z+2,
                                                    5/16*domain_size_z+2, 5/16*domain_size_z+2, 5/16*domain_size_z+2, 5/16*domain_size_z+2,
                                                    6/16*domain_size_z+2, 6/16*domain_size_z+2, 6/16*domain_size_z+2, 6/16*domain_size_z+2,
                                                    7/16*domain_size_z+2, 7/16*domain_size_z+2, 7/16*domain_size_z+2, 7/16*domain_size_z+2,
                                                    8/16*domain_size_z+2, 8/16*domain_size_z+2, 8/16*domain_size_z+2, 8/16*domain_size_z+2,
                                                    9/16*domain_size_z+2, 9/16*domain_size_z+2, 9/16*domain_size_z+2, 9/16*domain_size_z+2,
                                                   10/16*domain_size_z+2,10/16*domain_size_z+2,10/16*domain_size_z+2,10/16*domain_size_z+2,
                                                   11/16*domain_size_z+2,11/16*domain_size_z+2,11/16*domain_size_z+2,11/16*domain_size_z+2,
                                                   12/16*domain_size_z+2,12/16*domain_size_z+2,12/16*domain_size_z+2,12/16*domain_size_z+2,
                                                   13/16*domain_size_z+2,13/16*domain_size_z+2,13/16*domain_size_z+2,13/16*domain_size_z+2,
                                                   14/16*domain_size_z+2,14/16*domain_size_z+2,14/16*domain_size_z+2,14/16*domain_size_z+2,
                                                   15/16*domain_size_z+2,15/16*domain_size_z+2,15/16*domain_size_z+2,15/16*domain_size_z+2,
                                                           domain_size_z,        domain_size_z,        domain_size_z,        domain_size_z])

x_decomposition_boundary_left_2    = torch.tensor([-1, half_domain_size_x-1,-1, half_domain_size_x-1,
                                                   -1, half_domain_size_x-1,-1, half_domain_size_x-1,
                                                   -1, half_domain_size_x-1,-1, half_domain_size_x-1,
                                                   -1, half_domain_size_x-1,-1, half_domain_size_x-1,
                                                   -1, half_domain_size_x-1,-1, half_domain_size_x-1,
                                                   -1, half_domain_size_x-1,-1, half_domain_size_x-1,
                                                   -1, half_domain_size_x-1,-1, half_domain_size_x-1,
                                                   -1, half_domain_size_x-1,-1, half_domain_size_x-1,
                                                   -1, half_domain_size_x-1,-1, half_domain_size_x-1,
                                                   -1, half_domain_size_x-1,-1, half_domain_size_x-1,
                                                   -1, half_domain_size_x-1,-1, half_domain_size_x-1,
                                                   -1, half_domain_size_x-1,-1, half_domain_size_x-1,
                                                   -1, half_domain_size_x-1,-1, half_domain_size_x-1,
                                                   -1, half_domain_size_x-1,-1, half_domain_size_x-1,
                                                   -1, half_domain_size_x-1,-1, half_domain_size_x-1,
                                                   -1, half_domain_size_x-1,-1, half_domain_size_x-1])

x_decomposition_boundary_right_2  = torch.tensor([half_domain_size_x, domain_size_x, half_domain_size_x, domain_size_x,
                                                  half_domain_size_x, domain_size_x, half_domain_size_x, domain_size_x,
                                                  half_domain_size_x, domain_size_x, half_domain_size_x, domain_size_x,
                                                  half_domain_size_x, domain_size_x, half_domain_size_x, domain_size_x,
                                                  half_domain_size_x, domain_size_x, half_domain_size_x, domain_size_x,
                                                  half_domain_size_x, domain_size_x, half_domain_size_x, domain_size_x,
                                                  half_domain_size_x, domain_size_x, half_domain_size_x, domain_size_x,
                                                  half_domain_size_x, domain_size_x, half_domain_size_x, domain_size_x,
                                                  half_domain_size_x, domain_size_x, half_domain_size_x, domain_size_x,
                                                  half_domain_size_x, domain_size_x, half_domain_size_x, domain_size_x,
                                                  half_domain_size_x, domain_size_x, half_domain_size_x, domain_size_x,
                                                  half_domain_size_x, domain_size_x, half_domain_size_x, domain_size_x,
                                                  half_domain_size_x, domain_size_x, half_domain_size_x, domain_size_x,
                                                  half_domain_size_x, domain_size_x, half_domain_size_x, domain_size_x,
                                                  half_domain_size_x, domain_size_x, half_domain_size_x, domain_size_x,
                                                  half_domain_size_x, domain_size_x, half_domain_size_x, domain_size_x])

y_decomposition_boundary_left_2    = torch.tensor([-1, -1, half_domain_size_y-1, half_domain_size_y-1,
                                                   -1, -1, half_domain_size_y-1, half_domain_size_y-1,
                                                   -1, -1, half_domain_size_y-1, half_domain_size_y-1,
                                                   -1, -1, half_domain_size_y-1, half_domain_size_y-1,
                                                   -1, -1, half_domain_size_y-1, half_domain_size_y-1,
                                                   -1, -1, half_domain_size_y-1, half_domain_size_y-1,
                                                   -1, -1, half_domain_size_y-1, half_domain_size_y-1,
                                                   -1, -1, half_domain_size_y-1, half_domain_size_y-1,
                                                   -1, -1, half_domain_size_y-1, half_domain_size_y-1,
                                                   -1, -1, half_domain_size_y-1, half_domain_size_y-1,
                                                   -1, -1, half_domain_size_y-1, half_domain_size_y-1,
                                                   -1, -1, half_domain_size_y-1, half_domain_size_y-1,
                                                   -1, -1, half_domain_size_y-1, half_domain_size_y-1,
                                                   -1, -1, half_domain_size_y-1, half_domain_size_y-1,
                                                   -1, -1, half_domain_size_y-1, half_domain_size_y-1,
                                                   -1, -1, half_domain_size_y-1, half_domain_size_y-1])

y_decomposition_boundary_right_2  = torch.tensor([half_domain_size_y,half_domain_size_y,domain_size_y,domain_size_y,
                                                  half_domain_size_y,half_domain_size_y,domain_size_y,domain_size_y,
                                                  half_domain_size_y,half_domain_size_y,domain_size_y,domain_size_y,
                                                  half_domain_size_y,half_domain_size_y,domain_size_y,domain_size_y,
                                                  half_domain_size_y,half_domain_size_y,domain_size_y,domain_size_y,
                                                  half_domain_size_y,half_domain_size_y,domain_size_y,domain_size_y,
                                                  half_domain_size_y,half_domain_size_y,domain_size_y,domain_size_y,
                                                  half_domain_size_y,half_domain_size_y,domain_size_y,domain_size_y,
                                                  half_domain_size_y,half_domain_size_y,domain_size_y,domain_size_y,
                                                  half_domain_size_y,half_domain_size_y,domain_size_y,domain_size_y,
                                                  half_domain_size_y,half_domain_size_y,domain_size_y,domain_size_y,
                                                  half_domain_size_y,half_domain_size_y,domain_size_y,domain_size_y,
                                                  half_domain_size_y,half_domain_size_y,domain_size_y,domain_size_y,
                                                  half_domain_size_y,half_domain_size_y,domain_size_y,domain_size_y,
                                                  half_domain_size_y,half_domain_size_y,domain_size_y,domain_size_y,
                                                  half_domain_size_y,half_domain_size_y,domain_size_y,domain_size_y])

z_decomposition_boundary_left_2    = torch.tensor([                   -1,                   -1,                   -1,                   -1,
                                                    1/16*domain_size_z-1, 1/16*domain_size_z-1, 1/16*domain_size_z-1, 1/16*domain_size_z-1,
                                                    2/16*domain_size_z-1, 2/16*domain_size_z-1, 2/16*domain_size_z-1, 2/16*domain_size_z-1,
                                                    3/16*domain_size_z-1, 3/16*domain_size_z-1, 3/16*domain_size_z-1, 3/16*domain_size_z-1,
                                                    4/16*domain_size_z-1, 4/16*domain_size_z-1, 4/16*domain_size_z-1, 4/16*domain_size_z-1,
                                                    5/16*domain_size_z-1, 5/16*domain_size_z-1, 5/16*domain_size_z-1, 5/16*domain_size_z-1,
                                                    6/16*domain_size_z-1, 6/16*domain_size_z-1, 6/16*domain_size_z-1, 6/16*domain_size_z-1,
                                                    7/16*domain_size_z-1, 7/16*domain_size_z-1, 7/16*domain_size_z-1, 7/16*domain_size_z-1,
                                                    8/16*domain_size_z-1, 8/16*domain_size_z-1, 8/16*domain_size_z-1, 8/16*domain_size_z-1,
                                                    9/16*domain_size_z-1, 9/16*domain_size_z-1, 9/16*domain_size_z-1, 9/16*domain_size_z-1,
                                                   10/16*domain_size_z-1,10/16*domain_size_z-1,10/16*domain_size_z-1,10/16*domain_size_z-1,
                                                   11/16*domain_size_z-1,11/16*domain_size_z-1,11/16*domain_size_z-1,11/16*domain_size_z-1,
                                                   12/16*domain_size_z-1,12/16*domain_size_z-1,12/16*domain_size_z-1,12/16*domain_size_z-1,
                                                   13/16*domain_size_z-1,13/16*domain_size_z-1,13/16*domain_size_z-1,13/16*domain_size_z-1,
                                                   14/16*domain_size_z-1,14/16*domain_size_z-1,14/16*domain_size_z-1,14/16*domain_size_z-1,
                                                   15/16*domain_size_z-1,15/16*domain_size_z-1,15/16*domain_size_z-1,15/16*domain_size_z-1])

z_decomposition_boundary_right_2   = torch.tensor([ 1/16*domain_size_z, 1/16*domain_size_z, 1/16*domain_size_z, 1/16*domain_size_z,
                                                    2/16*domain_size_z, 2/16*domain_size_z, 2/16*domain_size_z, 2/16*domain_size_z,
                                                    3/16*domain_size_z, 3/16*domain_size_z, 3/16*domain_size_z, 3/16*domain_size_z,
                                                    4/16*domain_size_z, 4/16*domain_size_z, 4/16*domain_size_z, 4/16*domain_size_z,
                                                    5/16*domain_size_z, 5/16*domain_size_z, 5/16*domain_size_z, 5/16*domain_size_z,
                                                    6/16*domain_size_z, 6/16*domain_size_z, 6/16*domain_size_z, 6/16*domain_size_z,
                                                    7/16*domain_size_z, 7/16*domain_size_z, 7/16*domain_size_z, 7/16*domain_size_z,
                                                    8/16*domain_size_z, 8/16*domain_size_z, 8/16*domain_size_z, 8/16*domain_size_z,
                                                    9/16*domain_size_z, 9/16*domain_size_z, 9/16*domain_size_z, 9/16*domain_size_z,
                                                   10/16*domain_size_z,10/16*domain_size_z,10/16*domain_size_z,10/16*domain_size_z,
                                                   11/16*domain_size_z,11/16*domain_size_z,11/16*domain_size_z,11/16*domain_size_z,
                                                   12/16*domain_size_z,12/16*domain_size_z,12/16*domain_size_z,12/16*domain_size_z,
                                                   13/16*domain_size_z,13/16*domain_size_z,13/16*domain_size_z,13/16*domain_size_z,
                                                   14/16*domain_size_z,14/16*domain_size_z,14/16*domain_size_z,14/16*domain_size_z,
                                                   15/16*domain_size_z,15/16*domain_size_z,15/16*domain_size_z,15/16*domain_size_z,
                                                         domain_size_z,      domain_size_z,      domain_size_z,      domain_size_z])


z_decomposition_boundary_left_1 = torch.round(z_decomposition_boundary_left_1).int()
z_decomposition_boundary_right_1 = torch.round(z_decomposition_boundary_right_1).int()

z_decomposition_boundary_left_2 = torch.round(z_decomposition_boundary_left_2).int()
z_decomposition_boundary_right_2 = torch.round(z_decomposition_boundary_right_2).int()


cell_adjustment_x = torch.tensor([0, half_domain_size_x, 0, half_domain_size_x, 
                                  0, half_domain_size_x, 0, half_domain_size_x, 
                                  0, half_domain_size_x, 0, half_domain_size_x, 
                                  0, half_domain_size_x, 0, half_domain_size_x,
                                  0, half_domain_size_x, 0, half_domain_size_x, 
                                  0, half_domain_size_x, 0, half_domain_size_x, 
                                  0, half_domain_size_x, 0, half_domain_size_x, 
                                  0, half_domain_size_x, 0, half_domain_size_x,
                                  0, half_domain_size_x, 0, half_domain_size_x, 
                                  0, half_domain_size_x, 0, half_domain_size_x, 
                                  0, half_domain_size_x, 0, half_domain_size_x, 
                                  0, half_domain_size_x, 0, half_domain_size_x,
                                  0, half_domain_size_x, 0, half_domain_size_x, 
                                  0, half_domain_size_x, 0, half_domain_size_x, 
                                  0, half_domain_size_x, 0, half_domain_size_x, 
                                  0, half_domain_size_x, 0, half_domain_size_x])

cell_adjustment_y = torch.tensor([0,0,half_domain_size_y,half_domain_size_y,
                                  0,0,half_domain_size_y,half_domain_size_y,
                                  0,0,half_domain_size_y,half_domain_size_y,
                                  0,0,half_domain_size_y,half_domain_size_y,
                                  0,0,half_domain_size_y,half_domain_size_y,
                                  0,0,half_domain_size_y,half_domain_size_y,
                                  0,0,half_domain_size_y,half_domain_size_y,
                                  0,0,half_domain_size_y,half_domain_size_y,
                                  0,0,half_domain_size_y,half_domain_size_y,
                                  0,0,half_domain_size_y,half_domain_size_y,
                                  0,0,half_domain_size_y,half_domain_size_y,
                                  0,0,half_domain_size_y,half_domain_size_y,
                                  0,0,half_domain_size_y,half_domain_size_y,
                                  0,0,half_domain_size_y,half_domain_size_y,
                                  0,0,half_domain_size_y,half_domain_size_y,
                                  0,0,half_domain_size_y,half_domain_size_y])

cell_adjustment_z = torch.tensor([                                     0,                    0,                    0,                    0,   
                                                    1/16*domain_size_z-2, 1/16*domain_size_z-2, 1/16*domain_size_z-2, 1/16*domain_size_z-2,
                                                    2/16*domain_size_z-2, 2/16*domain_size_z-2, 2/16*domain_size_z-2, 2/16*domain_size_z-2,
                                                    3/16*domain_size_z-2, 3/16*domain_size_z-2, 3/16*domain_size_z-2, 3/16*domain_size_z-2,
                                                    4/16*domain_size_z-2, 4/16*domain_size_z-2, 4/16*domain_size_z-2, 4/16*domain_size_z-2,
                                                    5/16*domain_size_z-2, 5/16*domain_size_z-2, 5/16*domain_size_z-2, 5/16*domain_size_z-2,
                                                    6/16*domain_size_z-2, 6/16*domain_size_z-2, 6/16*domain_size_z-2, 6/16*domain_size_z-2,
                                                    7/16*domain_size_z-2, 7/16*domain_size_z-2, 7/16*domain_size_z-2, 7/16*domain_size_z-2,
                                                    8/16*domain_size_z-2, 8/16*domain_size_z-2, 8/16*domain_size_z-2, 8/16*domain_size_z-2,
                                                    9/16*domain_size_z-2, 9/16*domain_size_z-2, 9/16*domain_size_z-2, 9/16*domain_size_z-2,
                                                   10/16*domain_size_z-2,10/16*domain_size_z-2,10/16*domain_size_z-2,10/16*domain_size_z-2,
                                                   11/16*domain_size_z-2,11/16*domain_size_z-2,11/16*domain_size_z-2,11/16*domain_size_z-2,
                                                   12/16*domain_size_z-2,12/16*domain_size_z-2,12/16*domain_size_z-2,12/16*domain_size_z-2,
                                                   13/16*domain_size_z-2,13/16*domain_size_z-2,13/16*domain_size_z-2,13/16*domain_size_z-2,
                                                   14/16*domain_size_z-2,14/16*domain_size_z-2,14/16*domain_size_z-2,14/16*domain_size_z-2,
                                                   15/16*domain_size_z-2,15/16*domain_size_z-2,15/16*domain_size_z-2,15/16*domain_size_z-2])

cell_adjustment_z = torch.round(cell_adjustment_z).int()

i, j ,k= np.meshgrid(np.arange(2, half_domain_size_x-2), np.arange(2, half_domain_size_y-2),np.arange(2, half_domain_size_z-2))
total_x_grid[0, 0, k*2, j*2, i*2] = i * cell_size * 2
total_y_grid[0, 0, k*2, j*2, i*2] = j * cell_size * 2
total_z_grid[0, 0, k*2, j*2, i*2] = k * cell_size * 2
total_vx_grid[0, 0, k*2, j*2, i*2] = -0.001
total_vy_grid[0, 0, k*2, j*2, i*2] = random.uniform(-1.0,1.0)*0.001
total_vz_grid[0, 0, k*2, j*2, i*2] = 0.001
total_mask[0, 0, k*2, j*2, i*2] = 1
     
compressed_x_grid = total_x_grid[total_mask!=0]
compressed_y_grid = total_y_grid[total_mask!=0]
compressed_z_grid = total_z_grid[total_mask!=0]

compressed_vx_grid = total_vx_grid[total_mask!=0]
compressed_vy_grid = total_vy_grid[total_mask!=0]
compressed_vz_grid = total_vz_grid[total_mask!=0]

total_particle_number = len(compressed_vx_grid)

total_mask = torch.from_numpy(total_mask).to(device)
del total_x_grid, total_y_grid, total_z_grid, total_vx_grid, total_vy_grid, total_vz_grid, total_mask

compressed_x_grid = torch.from_numpy(compressed_x_grid).float().to(device)
compressed_y_grid = torch.from_numpy(compressed_y_grid).float().to(device)
compressed_z_grid = torch.from_numpy(compressed_z_grid).float().to(device)

compressed_vx_grid = torch.from_numpy(compressed_vx_grid).float().to(device)
compressed_vy_grid = torch.from_numpy(compressed_vy_grid).float().to(device)
compressed_vz_grid = torch.from_numpy(compressed_vz_grid).float().to(device)

mask_grid = torch.from_numpy(mask_grid).float().to(device)
x_grid = torch.from_numpy(x_grid).float().to(device)
y_grid = torch.from_numpy(y_grid).float().to(device)
z_grid = torch.from_numpy(z_grid).float().to(device)

vx_grid = torch.from_numpy(vx_grid).float().to(device)
vy_grid = torch.from_numpy(vy_grid).float().to(device)
vz_grid = torch.from_numpy(vz_grid).float().to(device)
print('Number of particles:', torch.count_nonzero(compressed_y_grid))

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

    def forward(self, compressed_x_grid, compressed_y_grid, compressed_z_grid, compressed_vx_grid, compressed_vy_grid, compressed_vz_grid, d, kn, damping_coefficient_Eta, friction_coefficient, dt, input_shape, filter_size):
        # calculate the total particle number
        cell_xold = compressed_x_grid/d
        cell_yold = compressed_y_grid/d
        cell_zold = compressed_z_grid/d
        
        cell_xold = torch.round(cell_xold).long()
        cell_yold = torch.round(cell_yold).long()
        cell_zold = torch.round(cell_zold).long()
        particle_number = len(compressed_x_grid)
        # loop in each domain
        for domain_number in range(64):
            print("domain number", domain_number)
            mask = torch.zeros([particle_number], device=device)
            mask  [(cell_xold  >  x_decomposition_boundary_left_1[domain_number]) & (cell_xold  <  x_decomposition_boundary_right_1[domain_number]) & (cell_yold  >  y_decomposition_boundary_left_1[domain_number]) & (cell_yold  <  y_decomposition_boundary_right_1[domain_number]) & (cell_zold  >  z_decomposition_boundary_left_1[domain_number]) & (cell_zold  <  z_decomposition_boundary_right_1[domain_number])]=1
            mask_2 = torch.zeros([particle_number], device=device)
            mask_2[(cell_xold  >  x_decomposition_boundary_left_2[domain_number]) & (cell_xold  <  x_decomposition_boundary_right_2[domain_number]) & (cell_yold  >  y_decomposition_boundary_left_2[domain_number]) & (cell_yold  <  y_decomposition_boundary_right_2[domain_number]) & (cell_zold  >  z_decomposition_boundary_left_2[domain_number]) & (cell_zold  <  z_decomposition_boundary_right_2[domain_number])]=1
            # compress the cell serial number for each bigger domain          
            cell_xold_domain = cell_xold[mask != 0]
            cell_yold_domain = cell_yold[mask != 0]
            cell_zold_domain = cell_zold[mask != 0]
            cell_xold_domain = cell_xold_domain - cell_adjustment_x[domain_number]
            cell_yold_domain = cell_yold_domain - cell_adjustment_y[domain_number]
            cell_zold_domain = cell_zold_domain - cell_adjustment_z[domain_number]
            # reconstruct coordinate grid for the bigger domain
            x_grid = torch.zeros(input_shape_domain, device = device).float()
            y_grid = torch.zeros(input_shape_domain, device = device).float()
            z_grid = torch.zeros(input_shape_domain, device = device).float()
            vx_grid = torch.zeros(input_shape_domain, device = device).float()
            vy_grid = torch.zeros(input_shape_domain, device = device).float()
            vz_grid = torch.zeros(input_shape_domain, device = device).float()
            mask_grid = torch.zeros(input_shape_domain)           
            x_grid[0,0,cell_zold_domain, cell_yold_domain, cell_xold_domain] = compressed_x_grid[mask!=0]
            y_grid[0,0,cell_zold_domain, cell_yold_domain, cell_xold_domain] = compressed_y_grid[mask!=0]
            z_grid[0,0,cell_zold_domain, cell_yold_domain, cell_xold_domain] = compressed_z_grid[mask!=0]
            
            vx_grid[0,0,cell_zold_domain, cell_yold_domain, cell_xold_domain] = compressed_vx_grid[mask!=0]
            vy_grid[0,0,cell_zold_domain, cell_yold_domain, cell_xold_domain] = compressed_vy_grid[mask!=0]
            vz_grid[0,0,cell_zold_domain, cell_yold_domain, cell_xold_domain] = compressed_vz_grid[mask!=0]
            
            mask_grid[0,0,cell_zold_domain, cell_yold_domain, cell_xold_domain] = 1
            # calculate the domain particle number for each bigger domain
            domain_particle_number =  torch.count_nonzero(mask_grid)
            
            # define the collision and damping force grid for each bigger domain            
            fx_grid_collision, fy_grid_collision, fz_grid_collision = [torch.zeros(domain_particle_number, device=device) for _ in range(3)]
            fx_grid_damping, fy_grid_damping, fz_grid_damping = [torch.zeros(domain_particle_number, device=device) for _ in range(3)]            

            # define the eplis, ones and zeros for each bigger domain
            domain_particle_eplis = torch.ones(domain_particle_number, device=device) * 1e-04
            domain_particle_ones = torch.ones(domain_particle_number, device=device)
            domain_particle_zeros = torch.zeros(domain_particle_number, device=device)            
            # loop in the 5*5 detector in each bigger domain
            for i in range(filter_size):
                for j in range(filter_size):
                    for k in range(filter_size):
                        # calculate distance between the two particles in each bigger domain
                        diffx = self.detector(x_grid, i, j, k) # individual
                        diffy = self.detector(y_grid, i, j, k) # individual
                        diffz = self.detector(z_grid, i, j, k) # individual
                        # compress the distance between the two particles in each bigger domain
                        compressed_diffx=diffx[0, 0, cell_zold_domain, cell_yold_domain, cell_xold_domain]
                        compressed_diffy=diffy[0, 0, cell_zold_domain, cell_yold_domain, cell_xold_domain]
                        compressed_diffz=diffz[0, 0, cell_zold_domain, cell_yold_domain, cell_xold_domain]
                        compressed_dist = torch.sqrt(compressed_diffx**2 + compressed_diffy**2 + compressed_diffz**2)
                        
                        # calculate collision force between the two particles in each bigger domain
                        fx_grid_collision =  fx_grid_collision + torch.where(torch.lt(compressed_dist,2 * d), kn * (compressed_dist - 2 * d ) * compressed_diffx / torch.maximum(domain_particle_eplis, compressed_dist), domain_particle_zeros) # individual
                        fy_grid_collision =  fy_grid_collision + torch.where(torch.lt(compressed_dist,2 * d), kn * (compressed_dist - 2 * d ) * compressed_diffy / torch.maximum(domain_particle_eplis, compressed_dist), domain_particle_zeros) # individual
                        fz_grid_collision =  fz_grid_collision + torch.where(torch.lt(compressed_dist,2 * d), kn * (compressed_dist - 2 * d ) * compressed_diffz / torch.maximum(domain_particle_eplis, compressed_dist), domain_particle_zeros) # individual            
                        
                        diffvx = self.detector(vx_grid, i, j, k) # individual
                        diffvy = self.detector(vy_grid, i, j, k) # individual
                        diffvz = self.detector(vz_grid, i, j, k) # individual
                        
                        compressed_diffvx=diffvx[0, 0, cell_zold_domain, cell_yold_domain, cell_xold_domain]
                        compressed_diffvy=diffvy[0, 0, cell_zold_domain, cell_yold_domain, cell_xold_domain]
                        compressed_diffvz=diffvz[0, 0, cell_zold_domain, cell_yold_domain, cell_xold_domain]
                        
                        compressed_diffvx_Vn =  compressed_diffvx * compressed_diffx /  torch.maximum(domain_particle_eplis, compressed_dist)
                        compressed_diffvy_Vn =  compressed_diffvy * compressed_diffy /  torch.maximum(domain_particle_eplis, compressed_dist)
                        compressed_diffvz_Vn =  compressed_diffvz * compressed_diffz /  torch.maximum(domain_particle_eplis, compressed_dist) 
                        compressed_diffv_Vn  =   compressed_diffvx_Vn + compressed_diffvy_Vn + compressed_diffvz_Vn

                        # calculate the damping force between the two particles in each bigger domain
                        compressed_diffv_Vn_x =  compressed_diffv_Vn * compressed_diffx /  torch.maximum(domain_particle_eplis, compressed_dist)
                        compressed_diffv_Vn_y =  compressed_diffv_Vn * compressed_diffy /  torch.maximum(domain_particle_eplis, compressed_dist)
                        compressed_diffv_Vn_z =  compressed_diffv_Vn * compressed_diffz /  torch.maximum(domain_particle_eplis, compressed_dist)         
                        
                        # compress the damping force between the two particles in each bigger domain
                        fx_grid_damping =  fx_grid_damping + torch.where(torch.lt(compressed_dist, 2*d), damping_coefficient_Eta * compressed_diffv_Vn_x, domain_particle_zeros) # individual   
                        fy_grid_damping =  fy_grid_damping + torch.where(torch.lt(compressed_dist, 2*d), damping_coefficient_Eta * compressed_diffv_Vn_y, domain_particle_zeros) # individual 
                        fz_grid_damping =  fz_grid_damping + torch.where(torch.lt(compressed_dist, 2*d), damping_coefficient_Eta * compressed_diffv_Vn_z, domain_particle_zeros) # individual 
            del x_grid, y_grid, z_grid, compressed_diffx, compressed_diffy,compressed_diffz,compressed_diffvx, compressed_diffvy,compressed_diffvz,compressed_diffvx_Vn,compressed_diffvy_Vn,compressed_diffvz_Vn,
            # compress the velocity and coordinates in each bigger domain
            compressed_x_grid_domain = compressed_x_grid[mask!=0]
            compressed_y_grid_domain = compressed_y_grid[mask!=0]
            compressed_z_grid_domain = compressed_z_grid[mask!=0]

            compressed_vx_grid_domain = compressed_vx_grid[mask!=0]
            compressed_vy_grid_domain = compressed_vy_grid[mask!=0]
            compressed_vz_grid_domain = compressed_vz_grid[mask!=0]
            # del diffx, diffy, diffz, compressed_diffvx_Vn, compressed_diffvy_Vn, compressed_diffvz_Vn, compressed_diffv_Vn, compressed_diffv_Vn_x, compressed_diffv_Vn_y, compressed_diffv_Vn_z
            # judge whether the particle is colliding the boundaries in each bigger domain
            compressed_is_left_overlap     = torch.ne(compressed_x_grid_domain, 0.0000) & torch.lt(compressed_x_grid_domain, d) # Overlap with bottom wall
            compressed_is_right_overlap    = torch.gt(compressed_x_grid_domain, domain_size_x*cell_size-2*d)# Overlap with bottom wall
            compressed_is_bottom_overlap   = torch.ne(compressed_y_grid_domain, 0.0000) & torch.lt(compressed_y_grid_domain, d) # Overlap with bottom wall
            compressed_is_top_overlap      = torch.gt(compressed_y_grid_domain, domain_size_y*cell_size-2*d ) # Overlap with bottom wall
            compressed_is_forward_overlap  = torch.ne(compressed_z_grid_domain, 0.0000) & torch.lt(compressed_z_grid_domain, d) # Overlap with bottom wall
            compressed_is_backward_overlap = torch.gt(compressed_z_grid_domain, domain_size_z*cell_size-2*d ) # Overlap with bottom wall

            # calculate the elastic force from the boundaries in each bigger domain
            fx_grid_boundary_left     = kn * torch.where(compressed_is_left_overlap,    domain_particle_ones, domain_particle_zeros) * (d - compressed_x_grid_domain)
            fx_grid_boundary_right    = kn * torch.where(compressed_is_right_overlap,   domain_particle_ones, domain_particle_zeros) * (compressed_x_grid_domain - domain_size_x*cell_size + 2*d)
            fy_grid_boundary_bottom   = kn * torch.where(compressed_is_bottom_overlap,  domain_particle_ones, domain_particle_zeros) * (d - compressed_y_grid_domain)
            fy_grid_boundary_top      = kn * torch.where(compressed_is_top_overlap,     domain_particle_ones, domain_particle_zeros) * (compressed_y_grid_domain - domain_size_y*cell_size + 2*d)
            fz_grid_boundary_forward  = kn * torch.where(compressed_is_forward_overlap, domain_particle_ones, domain_particle_zeros) * (d - compressed_z_grid_domain)
            fz_grid_boundary_backward = kn * torch.where(compressed_is_backward_overlap,domain_particle_ones, domain_particle_zeros) * (compressed_z_grid_domain - domain_size_z*cell_size + 2*d)

            # calculate damping force from the boundaries in each bigger domain
            fx_grid_left_damping      = damping_coefficient_Eta_wall * compressed_vx_grid_domain *torch.where(compressed_is_left_overlap,    torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
            fx_grid_right_damping     = damping_coefficient_Eta_wall * compressed_vx_grid_domain *torch.where(compressed_is_right_overlap,   torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
            fy_grid_bottom_damping    = damping_coefficient_Eta_wall * compressed_vy_grid_domain *torch.where(compressed_is_bottom_overlap,  torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
            fy_grid_top_damping       = damping_coefficient_Eta_wall * compressed_vy_grid_domain *torch.where(compressed_is_top_overlap,     torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
            fz_grid_forward_damping   = damping_coefficient_Eta_wall * compressed_vz_grid_domain *torch.where(compressed_is_forward_overlap, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
            fz_grid_backward_damping  = damping_coefficient_Eta_wall * compressed_vz_grid_domain *torch.where(compressed_is_backward_overlap,torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
            
            # calculate the new velocity with acceleration calculated by forces in each bigger domain
            compressed_vx_grid_domain = compressed_vx_grid_domain  +  (dt / particle_mass) * ( - 0   * particle_mass)  + (dt / particle_mass) * ( - fx_grid_boundary_right    + fx_grid_boundary_left    - fx_grid_collision - fx_grid_damping - fx_grid_left_damping    - fx_grid_right_damping    ) 
            compressed_vy_grid_domain = compressed_vy_grid_domain  +  (dt / particle_mass) * ( - 0   * particle_mass)  + (dt / particle_mass) * ( - fy_grid_boundary_top      + fy_grid_boundary_bottom  - fy_grid_collision - fy_grid_damping - fy_grid_bottom_damping  - fy_grid_top_damping      )  
            compressed_vz_grid_domain = compressed_vz_grid_domain  +  (dt / particle_mass) * ( -9.8  * particle_mass)  + (dt / particle_mass) * ( - fz_grid_boundary_backward + fz_grid_boundary_forward - fz_grid_collision - fz_grid_damping - fz_grid_forward_damping - fz_grid_backward_damping ) 
             
            del fx_grid_boundary_left, fx_grid_boundary_right, fy_grid_boundary_bottom, fy_grid_boundary_top, fz_grid_boundary_forward, fz_grid_boundary_backward
            del fx_grid_left_damping, fx_grid_right_damping, fy_grid_bottom_damping, fy_grid_top_damping, fz_grid_forward_damping, fz_grid_backward_damping

            # Update particle coordniates
            compressed_x_grid_domain = compressed_x_grid_domain + dt * compressed_vx_grid_domain
            compressed_y_grid_domain = compressed_y_grid_domain + dt * compressed_vy_grid_domain
            compressed_z_grid_domain = compressed_z_grid_domain + dt * compressed_vz_grid_domain
            new_compressed_grid = torch.zeros(particle_number, device=device)
            new_compressed_grid[mask!=0] = compressed_x_grid_domain
            compressed_x_grid[mask_2!=0] = new_compressed_grid[mask_2!=0]

            new_compressed_grid[mask!=0] = compressed_y_grid_domain
            compressed_y_grid[mask_2!=0] = new_compressed_grid[mask_2!=0]
            
            new_compressed_grid[mask!=0] = compressed_z_grid_domain
            compressed_z_grid[mask_2!=0] = new_compressed_grid[mask_2!=0]

            # reconstruct the 3D grid serial number for each bigger domain
            new_compressed_grid[mask!=0] = compressed_vx_grid_domain
            compressed_vx_grid[mask_2!=0] = new_compressed_grid[mask_2!=0]

            new_compressed_grid[mask!=0] = compressed_vy_grid_domain
            compressed_vy_grid[mask_2!=0] = new_compressed_grid[mask_2!=0]
            
            new_compressed_grid[mask!=0] = compressed_vz_grid_domain
            compressed_vz_grid[mask_2!=0] = new_compressed_grid[mask_2!=0]
            
            del fx_grid_collision, fx_grid_damping, fy_grid_collision, fy_grid_damping, fz_grid_collision,fz_grid_damping
        return compressed_x_grid, compressed_y_grid, compressed_z_grid, compressed_vx_grid, compressed_vy_grid, compressed_vz_grid
model = AI4DEM().to(device)
# model = torch.compile(model, mode="reduce-overhead")
# Module 2: Contact detection and force calculation
t = 0
dt = 0.001  # 0.0001
ntime = 100000

# Convert np.array into torch.tensor and transfer it to GPU
filter_size = 5 
# Initialize tensors
eplis_global = torch.ones(input_shape_global, device=device) * 1e-04
eplis_domain = torch.ones(input_shape_domain, device=device) * 1e-04

# Main simulation loop
start = time.time()
with torch.no_grad():
    for itime in range(1, ntime + 1):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            [compressed_x_grid, compressed_y_grid, compressed_z_grid, compressed_vx_grid, compressed_vy_grid, compressed_vz_grid] = model(compressed_x_grid, compressed_y_grid, compressed_z_grid, compressed_vx_grid, compressed_vy_grid, compressed_vz_grid, cell_size, kn, damping_coefficient_Eta, friction_coefficient, dt, input_shape_global, filter_size)
            print('Time step:', itime, 'Number of particles:', torch.count_nonzero(compressed_z_grid).item()) 
            if itime % 10 == 0:
                xp = compressed_x_grid.cpu() 
                yp = compressed_y_grid.cpu() 
                zp = compressed_z_grid.cpu() 
                # # Visualize particles
                fig = plt.figure(figsize=(20, 10))
                ax = fig.add_subplot(111, projection="3d")
                sc = ax.scatter(xp,yp,zp, c=compressed_vx_grid.cpu(), cmap="turbo", s=S_graph, vmin=-10, vmax=10)
                cbar = plt.colorbar(sc, orientation='horizontal', shrink=0.35)
                cbar.set_label('$V_{z}$')
                ax = plt.gca()
                ax.set_xlim([0, domain_size_x*cell_size])
                ax.set_ylim([0, domain_size_y*cell_size])
                ax.set_zlim([0, domain_size_z*cell_size])
                ax.view_init(elev=45, azim=45)
                ax.set_xlabel('x')
                ax.set_ylabel('y')            
                ax.set_zlabel('z')  
                if itime < 10:
                    save_name = "3D_new/"+str(itime)+".jpg"
                elif itime >= 10 and itime < 100:
                    save_name = "3D_new/"+str(itime)+".jpg"
                elif itime >= 100 and itime < 1000:
                    save_name = "3D_new/"+str(itime)+".jpg"
                elif itime >= 1000 and itime < 10000:
                    save_name = "3D_new/"+str(itime)+".jpg"
                else:
                    save_name = "3D_new/"+str(itime)+".jpg"
                plt.savefig(save_name, dpi=1000, bbox_inches='tight')
                plt.close()
                
            save_path = 'DEM_no_friction'
end = time.time()
print('Elapsed time:', end - start)