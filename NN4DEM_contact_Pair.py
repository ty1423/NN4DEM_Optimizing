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

domain_size_x = 50
domain_size_y = 50
domain_size_z = 50

half_domain_size_x = int(domain_size_x/2)+1
half_domain_size_y = int(domain_size_y/2)+1
half_domain_size_z = int(domain_size_z/2)+1

# domain_size = 1500  # Size of the square domain
# domain_height = 100
# half_domain_size = int(domain_size/2)+1
cell_size =0.05   # Cell size and particle radius
simulation_time = 1
kn = 60000#0#000  # Normal stiffness of the spring
dn = 0.5  # Normal damping coefficient
rho_p = 2700 
particle_mass = 4/3*3.1415*cell_size**3*rho_p #4188.7902

K_graph = 57*100000*1
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
input_shape_global = (1, 1, domain_size_z, domain_size_y, domain_size_x)

# Generate particles
# npt = int(domain_size ** 3)

x_grid = np.zeros(input_shape_global)
y_grid = np.zeros(input_shape_global)
z_grid = np.zeros(input_shape_global)

vx_grid = np.zeros(input_shape_global)
vy_grid = np.zeros(input_shape_global)
vz_grid = np.zeros(input_shape_global)
mask = np.zeros(input_shape_global)

particle_number = len(mask[mask!=0])
compressed_x_grid = torch.zeros(particle_number, device=device)
compressed_y_grid = torch.zeros(particle_number, device=device)
compressed_z_grid = torch.zeros(particle_number, device=device)

compressed_vx_grid = torch.zeros(particle_number, device=device)
compressed_vy_grid = torch.zeros(particle_number, device=device)
compressed_vz_grid = torch.zeros(particle_number, device=device)

fx_grid_collision = torch.zeros(particle_number, device=device)
fy_grid_collision = torch.zeros(particle_number, device=device)
fz_grid_collision = torch.zeros(particle_number, device=device)
        
fx_grid_damping = torch.zeros(particle_number, device=device)
fy_grid_damping = torch.zeros(particle_number, device=device)
fz_grid_damping = torch.zeros(particle_number, device=device)

for i in range(2, half_domain_size_x-2):
    for j in range(2, half_domain_size_y-2):
        for k in range(2,  half_domain_size_z-2):
            x_grid[0, 0, k*2, j*2, i*2] = i * cell_size * 2
            y_grid[0, 0, k*2, j*2, i*2] = j * cell_size * 2
            z_grid[0, 0, k*2, j*2, i*2] = k * cell_size * 2
            vx_grid[0, 0, k*2, j*2, i*2] = 0
            vy_grid[0, 0, k*2, j*2, i*2] = random.uniform(-1.0,1.0)*0.1
            vz_grid[0, 0, k*2, j*2, i*2] = 0.001
            mask[0, 0, k*2, j*2, i*2] = 1
            
                      
mask = torch.from_numpy(mask).float().to(device)
x_grid = torch.from_numpy(x_grid).float().to(device)
y_grid = torch.from_numpy(y_grid).float().to(device)
z_grid = torch.from_numpy(z_grid).float().to(device)

vx_grid = torch.from_numpy(vx_grid).float().to(device)
vy_grid = torch.from_numpy(vy_grid).float().to(device)
vz_grid = torch.from_numpy(vz_grid).float().to(device)

'''
x_grid[0, 0, 4, 4, 4] = 2 * cell_size * 2
y_grid[0, 0, 4, 4, 4] = 2 * cell_size * 2
z_grid[0, 0, 4, 4, 4] = 2 * cell_size * 2
vx_grid[0, 0, 4, 4, 4] = 0
vy_grid[0, 0, 4, 4, 4] = -0.5
vz_grid[0, 0, 4, 4, 4] = 0
mask[0, 0, 4, 4, 4] = 1

x_grid[0, 0, 4, 2, 4] = 2 * cell_size * 2
y_grid[0, 0, 4, 2, 4] = 1 * cell_size * 2
z_grid[0, 0, 4, 2, 4] = 2 * cell_size * 2
vx_grid[0, 0, 4, 2, 4] = 0
vy_grid[0, 0, 4, 2, 4] = 0.5
vz_grid[0, 0, 4, 2, 4] = 0
mask[0, 0, 4, 2, 4] = 1
'''

compressed_x_grid = x_grid[mask!= 0]
compressed_y_grid = y_grid[mask!= 0]
compressed_z_grid = z_grid[mask!= 0]

compressed_vx_grid = vx_grid[mask!= 0]
compressed_vy_grid = vy_grid[mask!= 0]
compressed_vz_grid = vz_grid[mask!= 0]


# Define the AI4DEM model
class AI4DEM(nn.Module):
    """AI4DEM model for particle interaction detection and force calculation"""
    def __init__(self):
        super(AI4DEM, self).__init__()

    def forward(self, compressed_x_grid, compressed_y_grid, compressed_z_grid, compressed_vx_grid, compressed_vy_grid, compressed_vz_grid, d, kn, damping_coefficient_Eta, friction_coefficient, dt, input_shape, filter_size):
        cell_xold = compressed_x_grid / d
        cell_yold = compressed_y_grid / d 
        cell_zold = compressed_z_grid / d 
        
        cell_xold = torch.round(cell_xold).long()
        cell_yold = torch.round(cell_yold).long()
        cell_zold = torch.round(cell_zold).long()
        
        mask = torch.zeros(input_shape_global, device = device)
        mask[0,0,cell_zold,cell_yold,cell_xold] = 1
        
        particle_number = len(mask[mask!=0])
        zeros = torch.zeros(particle_number, device=device) 
        eplis = torch.ones(particle_number, device=device) * 1e-04
        ones  = torch.ones(particle_number, device=device) 
        
        particle_number_record = torch.arange(0, particle_number, device=device)
        contact_particle_number_total_A = torch.tensor([1], device=device)
        contact_particle_number_total_B = torch.tensor([1], device=device)
        particle_number_tensor = torch.zeros(input_shape_global, device = device)
        particle_number = len(mask[0,0,cell_zold,cell_yold,cell_xold])
        mask[0,0,cell_zold,cell_yold,cell_xold] = 1
        particle_number_tensor[0, 0, cell_zold, cell_yold, cell_xold] = particle_number_record.float()
        for i in range(filter_size):
            for j in range(filter_size):
                for k in range(filter_size):
                    mask_contact_A = mask + torch.roll(mask, shifts=(k - 2, j - 2, i - 2), dims=(2, 3, 4))
                    contact_particle_number_A = particle_number_tensor[mask_contact_A == 2]
                    contact_particle_number_total_A = torch.cat((contact_particle_number_total_A, contact_particle_number_A))

                    mask_contact_B = mask + torch.roll(mask, shifts=(2 - k, 2 - j, 2 - i), dims=(2, 3, 4))
                    contact_particle_number_B = particle_number_tensor[mask_contact_B == 2]
                    contact_particle_number_total_B = torch.cat((contact_particle_number_total_B, contact_particle_number_B))
                    
        #Calculate the number of contact pair in the domain
        contact_particle_number_total_A_filtered, contact_particle_number_total_B_filtered = contact_particle_number_total_A[contact_particle_number_total_A != contact_particle_number_total_B], contact_particle_number_total_B[contact_particle_number_total_B != contact_particle_number_total_A]
        
        contact_particle_number_total_A_filtered = contact_particle_number_total_A_filtered.long()
        contact_particle_number_total_B_filtered = contact_particle_number_total_B_filtered.long()        
        
        contact_pair_number = len(contact_particle_number_total_A_filtered)
        #Initialize the zeros, eplis and ones with the number scale of contact pairs
        contact_pair_zeros = torch.zeros(contact_pair_number, device=device) 
        contact_pair_eplis = torch.ones(contact_pair_number, device=device) * 1e-04
        contact_pair_ones  = torch.ones(contact_pair_number, device=device) 
        # Calculate the x, y, z grid related to the contact pair
        # As are the location details related to the first contacted particles
        # Bs are the location details related to the Second contacted particles
        contact_pair_x_A = compressed_x_grid[contact_particle_number_total_A_filtered]
        contact_pair_y_A = compressed_y_grid[contact_particle_number_total_A_filtered]
        contact_pair_z_A = compressed_z_grid[contact_particle_number_total_A_filtered]
        
        # Calculate the x, y, z velocity grid related to the contact pair
        # As are the velocity details related to the first contacted particles
        contact_pair_vx_A = compressed_vx_grid[contact_particle_number_total_A_filtered]
        contact_pair_vy_A = compressed_vy_grid[contact_particle_number_total_A_filtered]
        contact_pair_vz_A = compressed_vz_grid[contact_particle_number_total_A_filtered]

        # Calculate the x, y, z grid related to the contact pair
        # Bs are the location details related to the Second contacted particles
        contact_pair_x_B = compressed_x_grid[contact_particle_number_total_B_filtered]
        contact_pair_y_B = compressed_y_grid[contact_particle_number_total_B_filtered]
        contact_pair_z_B = compressed_z_grid[contact_particle_number_total_B_filtered]
        
        # Bs are the velocity details related to the second contacted particles
        contact_pair_vx_B = compressed_vx_grid[contact_particle_number_total_B_filtered]
        contact_pair_vy_B = compressed_vy_grid[contact_particle_number_total_B_filtered]
        contact_pair_vz_B = compressed_vz_grid[contact_particle_number_total_B_filtered]
        
        # diffs are the coordinates differences between the first and second particles
        contact_pair_diffx = contact_pair_x_A - contact_pair_x_B
        contact_pair_diffy = contact_pair_y_A - contact_pair_y_B
        contact_pair_diffz = contact_pair_z_A - contact_pair_z_B

        # dists are the distances between the first and second particles
        contact_pair_dist = torch.sqrt(contact_pair_diffx**2 + contact_pair_diffy**2 + contact_pair_diffz**2)  

        # diffvs are the velocity differences between the first and second particles
        contact_pair_diffvx = contact_pair_vx_A - contact_pair_vx_B
        contact_pair_diffvy = contact_pair_vy_A - contact_pair_vy_B
        contact_pair_diffvz = contact_pair_vz_A - contact_pair_vz_B
        
        # contact_pair_contaction are the judgements that indicates the particles are contacted with each other or not
        contact_pair_contaction = torch.where(torch.lt(contact_pair_dist, 2*d), True, False)
        
        contact_pair_fx_collision =  torch.where(contact_pair_contaction, kn * (contact_pair_dist - 2 * d ) * contact_pair_diffx / torch.maximum(contact_pair_eplis, contact_pair_dist), contact_pair_zeros) # individual
        contact_pair_fy_collision =  torch.where(contact_pair_contaction, kn * (contact_pair_dist - 2 * d ) * contact_pair_diffy / torch.maximum(contact_pair_eplis, contact_pair_dist), contact_pair_zeros) # individual
        contact_pair_fz_collision =  torch.where(contact_pair_contaction, kn * (contact_pair_dist - 2 * d ) * contact_pair_diffz / torch.maximum(contact_pair_eplis, contact_pair_dist), contact_pair_zeros) # individual            
        
        contact_pair_fx_collision_aggregated = torch.zeros_like(particle_number_record, dtype=torch.float)
        contact_pair_fx_collision_aggregated.scatter_add_(0, contact_particle_number_total_A_filtered, contact_pair_fx_collision)
        contact_pair_fy_collision_aggregated = torch.zeros_like(particle_number_record, dtype=torch.float)
        contact_pair_fy_collision_aggregated.scatter_add_(0, contact_particle_number_total_A_filtered, contact_pair_fy_collision)
        contact_pair_fz_collision_aggregated = torch.zeros_like(particle_number_record, dtype=torch.float)
        contact_pair_fz_collision_aggregated.scatter_add_(0, contact_particle_number_total_A_filtered, contact_pair_fz_collision)

        contact_pair_diffvx_Vn =  contact_pair_diffvx * contact_pair_diffx /  torch.maximum(contact_pair_eplis, contact_pair_dist)
        contact_pair_diffvy_Vn =  contact_pair_diffvy * contact_pair_diffy /  torch.maximum(contact_pair_eplis, contact_pair_dist)
        contact_pair_diffvz_Vn =  contact_pair_diffvz * contact_pair_diffz /  torch.maximum(contact_pair_eplis, contact_pair_dist) 
        
        contact_pair_diffv_Vn = contact_pair_diffvx_Vn + contact_pair_diffvy_Vn + contact_pair_diffvz_Vn
        
        contact_pair_diffv_Vn_x =  contact_pair_diffv_Vn * contact_pair_diffx /  torch.maximum(contact_pair_eplis, contact_pair_dist)
        contact_pair_diffv_Vn_y =  contact_pair_diffv_Vn * contact_pair_diffy /  torch.maximum(contact_pair_eplis, contact_pair_dist)
        contact_pair_diffv_Vn_z =  contact_pair_diffv_Vn * contact_pair_diffz /  torch.maximum(contact_pair_eplis, contact_pair_dist)
        
        contact_pair_fx_grid_damping =   torch.where(contact_pair_contaction, damping_coefficient_Eta * contact_pair_diffv_Vn_x, contact_pair_zeros) # individual   
        contact_pair_fy_grid_damping =   torch.where(contact_pair_contaction, damping_coefficient_Eta * contact_pair_diffv_Vn_y, contact_pair_zeros) # individual 
        contact_pair_fz_grid_damping =   torch.where(contact_pair_contaction, damping_coefficient_Eta * contact_pair_diffv_Vn_z, contact_pair_zeros) # individual 
        
        contact_pair_fx_grid_damping_aggregated = torch.zeros_like(particle_number_record, dtype=torch.float)
        contact_pair_fx_grid_damping_aggregated.scatter_add_(0, contact_particle_number_total_A_filtered, contact_pair_fx_grid_damping)
        contact_pair_fy_grid_damping_aggregated = torch.zeros_like(particle_number_record, dtype=torch.float)
        contact_pair_fy_grid_damping_aggregated.scatter_add_(0, contact_particle_number_total_A_filtered, contact_pair_fy_grid_damping)
        contact_pair_fz_grid_damping_aggregated = torch.zeros_like(particle_number_record, dtype=torch.float)
        contact_pair_fz_grid_damping_aggregated.scatter_add_(0, contact_particle_number_total_A_filtered, contact_pair_fz_grid_damping)

        # judge whether the particle is colliding the boundaries
        compressed_is_left_overlap     = torch.ne(compressed_x_grid, 0.0000) & torch.lt(compressed_x_grid, d) # Overlap with bottom wall
        compressed_is_right_overlap    = torch.gt(compressed_x_grid, domain_size_x*cell_size-2*d)# Overlap with bottom wall
        compressed_is_bottom_overlap   = torch.ne(compressed_y_grid, 0.0000) & torch.lt(compressed_y_grid, d) # Overlap with bottom wall
        compressed_is_top_overlap      = torch.gt(compressed_y_grid, domain_size_y*cell_size-2*d ) # Overlap with bottom wall
        compressed_is_forward_overlap  = torch.ne(compressed_z_grid, 0.0000) & torch.lt(compressed_z_grid, d) # Overlap with bottom wall
        compressed_is_backward_overlap = torch.gt(compressed_z_grid, domain_size_z*cell_size-2*d ) # Overlap with bottom wall             

        # calculate the elastic force from the boundaries
        fx_grid_boundary_left     = kn * torch.where(compressed_is_left_overlap,    ones, zeros) * (d - compressed_x_grid)
        fx_grid_boundary_right    = kn * torch.where(compressed_is_right_overlap,   ones, zeros) * (compressed_x_grid - domain_size_x*cell_size + 2*d)
        fy_grid_boundary_bottom   = kn * torch.where(compressed_is_bottom_overlap,  ones, zeros) * (d - compressed_y_grid)
        fy_grid_boundary_top      = kn * torch.where(compressed_is_top_overlap,     ones, zeros) * (compressed_y_grid - domain_size_y*cell_size + 2*d)
        fz_grid_boundary_forward  = kn * torch.where(compressed_is_forward_overlap, ones, zeros) * (d - compressed_z_grid)
        fz_grid_boundary_backward = kn * torch.where(compressed_is_backward_overlap,ones, zeros) * (compressed_z_grid - domain_size_z*cell_size + 2*d)
                
        # calculate damping force from the boundaries
        fx_grid_left_damping      = damping_coefficient_Eta_wall * compressed_vx_grid *torch.where(compressed_is_left_overlap,    torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
        fx_grid_right_damping     = damping_coefficient_Eta_wall * compressed_vx_grid *torch.where(compressed_is_right_overlap,   torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
        fy_grid_bottom_damping    = damping_coefficient_Eta_wall * compressed_vy_grid *torch.where(compressed_is_bottom_overlap,  torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
        fy_grid_top_damping       = damping_coefficient_Eta_wall * compressed_vy_grid *torch.where(compressed_is_top_overlap,     torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
        fz_grid_forward_damping   = damping_coefficient_Eta_wall * compressed_vz_grid *torch.where(compressed_is_forward_overlap, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
        fz_grid_backward_damping  = damping_coefficient_Eta_wall * compressed_vz_grid *torch.where(compressed_is_backward_overlap,torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
        # calculate the new velocity with acceleration calculated by forces
        compressed_vx_grid = compressed_vx_grid + (dt / particle_mass) * ( -0                    - fx_grid_boundary_right    + fx_grid_boundary_left    - contact_pair_fx_collision_aggregated - contact_pair_fx_grid_damping_aggregated - fx_grid_left_damping    - fx_grid_right_damping    )
        compressed_vy_grid = compressed_vy_grid + (dt / particle_mass) * ( -0                    - fy_grid_boundary_top      + fy_grid_boundary_bottom  - contact_pair_fy_collision_aggregated - contact_pair_fy_grid_damping_aggregated - fy_grid_bottom_damping  - fy_grid_top_damping      )
        compressed_vz_grid = compressed_vz_grid + (dt / particle_mass) * ( -9.8 * particle_mass  - fz_grid_boundary_backward + fz_grid_boundary_forward - contact_pair_fz_collision_aggregated - contact_pair_fz_grid_damping_aggregated - fz_grid_forward_damping - fz_grid_backward_damping )

        # del fx_grid_collision, fy_grid_collision, fz_grid_collision, fx_grid_damping, fy_grid_damping, fz_grid_damping         
        del fx_grid_boundary_left, fx_grid_boundary_right, fy_grid_boundary_bottom, fy_grid_boundary_top, fz_grid_boundary_forward, fz_grid_boundary_backward
        del fx_grid_left_damping, fx_grid_right_damping, fy_grid_bottom_damping, fy_grid_top_damping, fz_grid_forward_damping, fz_grid_backward_damping
        
        # Update particle coordniates
        compressed_x_grid = compressed_x_grid + dt * compressed_vx_grid
        compressed_y_grid = compressed_y_grid + dt * compressed_vy_grid
        compressed_z_grid = compressed_z_grid + dt * compressed_vz_grid
        return compressed_x_grid, compressed_y_grid, compressed_z_grid, compressed_vx_grid, compressed_vy_grid, compressed_vz_grid, (fx_grid_collision+fx_grid_damping), (fy_grid_collision+fy_grid_damping), (fz_grid_collision+fz_grid_damping)

model = AI4DEM().to(device)
# model = torch.compile(model, mode="reduce-overhead")
# Module 2: Contact detection and force calculation
t = 0
dt = 0.001  # 0.0001
ntime = 20000000
# Convert np.array into torch.tensor and transfer it to GPU
filter_size = 5 

# Initialize tensors
eplis_global = torch.ones(input_shape_global, device=device) * 1e-04

# Main simulation loop
start = time.time()
with torch.no_grad():
    for itime in range(1, ntime + 1):
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            [compressed_x_grid, compressed_y_grid, compressed_z_grid, compressed_vx_grid, compressed_vy_grid, compressed_vz_grid, Fx, Fy, Fz] = model(compressed_x_grid, compressed_y_grid, compressed_z_grid, compressed_vx_grid, compressed_vy_grid, compressed_vz_grid, cell_size, kn, damping_coefficient_Eta, friction_coefficient, dt, input_shape_global, filter_size)
            print('Time step:', itime, 'Number of particles:', torch.count_nonzero(mask).item()) 
            if itime % 1000 == 0:
                xp = compressed_x_grid.cpu() 
                yp = compressed_y_grid.cpu() 
                zp = compressed_z_grid.cpu() 
                fig = plt.figure(figsize=(20, 10))
                ax = fig.add_subplot(111, projection="3d")
                sc = ax.scatter(xp,yp,zp)
                cbar = plt.colorbar(sc, orientation='horizontal', shrink=0.35)
                cbar.set_label('$AngularV_{x}$')
                ax = plt.gca()
                ax.set_xlim([0, domain_size_x*cell_size])
                ax.set_ylim([0, domain_size_y*cell_size])
                ax.set_zlim([0, domain_size_z*cell_size])
                ax.view_init(elev=45, azim=45)
                ax.set_xlabel('x')
                ax.set_ylabel('y')            
                ax.set_zlabel('z')    
                # Save visualization
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
end = time.time()
print('Elapsed time:', end - start)
