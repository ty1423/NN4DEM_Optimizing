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
kn = 600000#0#000  # Normal stiffness of the spring
dn = 0.5  # Normal damping coefficient
rho_p = 2700 
particle_mass = 4/3*3.1415*cell_size**3*rho_p #4188.7902
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


i, j ,k= np.meshgrid(np.arange(2, half_domain_size_x-2), np.arange(2, half_domain_size_y-2),np.arange(2, half_domain_size_z-2))
x_grid[0, 0, k*2, j*2, i*2] = i * cell_size * 2
y_grid[0, 0, k*2, j*2, i*2] = j * cell_size * 2
z_grid[0, 0, k*2, j*2, i*2] = k * cell_size * 2
vx_grid[0, 0, k*2, j*2, i*2] = -0.001
vy_grid[0, 0, k*2, j*2, i*2] = random.uniform(-1.0,1.0)*0.001
vz_grid[0, 0, k*2, j*2, i*2] = 0.001
mask[0, 0, k*2, j*2, i*2] = 1



print('Number of particles:', np.count_nonzero(mask))

mask = torch.from_numpy(mask).float().to(device)
x_grid = torch.from_numpy(x_grid).float().to(device)
y_grid = torch.from_numpy(y_grid).float().to(device)
z_grid = torch.from_numpy(z_grid).float().to(device)

vx_grid = torch.from_numpy(vx_grid).float().to(device)
vy_grid = torch.from_numpy(vy_grid).float().to(device)
vz_grid = torch.from_numpy(vz_grid).float().to(device)


compressed_x_grid = x_grid[mask!= 0]
compressed_y_grid = y_grid[mask!= 0]
compressed_z_grid = z_grid[mask!= 0]

compressed_vx_grid = vx_grid[mask!= 0]
compressed_vy_grid = vy_grid[mask!= 0]
compressed_vz_grid = vz_grid[mask!= 0]

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
    
    def forward(self, compressed_x_grid, compressed_y_grid, compressed_z_grid, compressed_vx_grid, compressed_vy_grid, compressed_vz_grid, d, kn, damping_coefficient_Eta, friction_coefficient, dt, input_shape, filter_size):
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
        
        mask[0,0,cell_zold,cell_yold,cell_xold]=1
        
        x_grid[0,0,cell_zold,cell_yold,cell_xold] = compressed_x_grid
        y_grid[0,0,cell_zold,cell_yold,cell_xold] = compressed_y_grid
        z_grid[0,0,cell_zold,cell_yold,cell_xold] = compressed_z_grid

        vx_grid[0,0,cell_zold,cell_yold,cell_xold] = compressed_vx_grid
        vy_grid[0,0,cell_zold,cell_yold,cell_xold] = compressed_vy_grid
        vz_grid[0,0,cell_zold,cell_yold,cell_xold] = compressed_vz_grid
        
        particle_number = len(mask[mask!=0])        
        fx_grid_collision = torch.zeros(particle_number, device=device)
        fy_grid_collision = torch.zeros(particle_number, device=device)
        fz_grid_collision = torch.zeros(particle_number, device=device)

        fx_grid_damping = torch.zeros(particle_number, device=device)
        fy_grid_damping = torch.zeros(particle_number, device=device)
        fz_grid_damping = torch.zeros(particle_number, device=device)
     
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
                    dist = torch.sqrt(diffx**2 + diffy**2 + diffz**2)  
                    
                    # calculate nodal velocity difference between the two particles
                    diffvx = self.detector(vx_grid, i, j, k) # individual
                    diffvy = self.detector(vy_grid, i, j, k) # individual
                    diffvz = self.detector(vz_grid, i, j, k) # individual
                    
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
                    
                    compressed_diffv_Vn_x =  compressed_diffv_Vn * compressed_diffx /  torch.maximum(eplis, compressed_dist)
                    compressed_diffv_Vn_y =  compressed_diffv_Vn * compressed_diffy /  torch.maximum(eplis, compressed_dist)
                    compressed_diffv_Vn_z =  compressed_diffv_Vn * compressed_diffz /  torch.maximum(eplis, compressed_dist)                     

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
                    
                    compressed_diffv_Vt_x = (compressed_diffvx) * (compressed_diffy**2 + compressed_diffz**2) / torch.maximum(eplis, compressed_dist**2) # + compressed_vx_angular
                    compressed_diffv_Vt_y = (compressed_diffvy) * (compressed_diffx**2 + compressed_diffz**2) / torch.maximum(eplis, compressed_dist**2) # + compressed_vy_angular
                    compressed_diffv_Vt_z = (compressed_diffvz) * (compressed_diffx**2 + compressed_diffy**2) / torch.maximum(eplis, compressed_dist**2) # + compressed_vz_angular
                    
                    compressed_diffv_Vt = torch.sqrt (compressed_diffv_Vt_x**2 + compressed_diffv_Vt_y**2 + compressed_diffv_Vt_z**2)

        del diffx, diffy, diffz, compressed_diffvx_Vn, compressed_diffvy_Vn, compressed_diffvz_Vn, compressed_diffv_Vn, compressed_diffv_Vn_x, compressed_diffv_Vn_y, compressed_diffv_Vn_z

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
        compressed_vx_grid = compressed_vx_grid + (dt / particle_mass) * ( -0                    - fx_grid_boundary_right    + fx_grid_boundary_left    - fx_grid_collision - fx_grid_damping - fx_grid_left_damping    - fx_grid_right_damping    )
        compressed_vy_grid = compressed_vy_grid + (dt / particle_mass) * ( -0                    - fy_grid_boundary_top      + fy_grid_boundary_bottom  - fy_grid_collision - fy_grid_damping - fy_grid_bottom_damping  - fy_grid_top_damping      )
        compressed_vz_grid = compressed_vz_grid + (dt / particle_mass) * ( -9.8 * particle_mass  - fz_grid_boundary_backward + fz_grid_boundary_forward - fz_grid_collision - fz_grid_damping - fz_grid_forward_damping - fz_grid_backward_damping )

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
ntime = 100000
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
            if itime % 100== 0:
                xp = compressed_x_grid.cpu() 
                yp = compressed_y_grid.cpu() 
                zp = compressed_z_grid.cpu() 
                
            # # Visualize particles
                fig = plt.figure(figsize=(20, 10))
                ax = fig.add_subplot(111, projection="3d")
                sc = ax.scatter(xp,yp,zp, c=compressed_x_grid.cpu(), cmap="turbo", s=S_graph, vmin=-10, vmax=10)
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
end = time.time()
print('Elapsed time:', end - start)
