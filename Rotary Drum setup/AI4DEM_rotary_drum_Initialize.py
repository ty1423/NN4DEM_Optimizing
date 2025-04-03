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
Mixer_anuglar_velocity_z = 0
domain_size_x = 100
domain_size_y = 100
domain_size_z = 18

half_domain_size_x = int(domain_size_x/2)
half_domain_size_y = int(domain_size_y/2)
half_domain_size_z = int(domain_size_z/2)

cell_size =0.00054  # Cell size and particle radius
K_graph = 160*10000000000*1
S_graph = K_graph * (cell_size / domain_size_x) ** 2
k_values   = np.array([1,8])
k_values_2 = np.array([3,5])

# coefficient of damping calculation 
# Discrete particle simulation of two-dimensional fluidized bed
# Module 1: Domain discretisation and initial particle insertion
# Create grid
input_shape_global = (1, 1, domain_size_z, domain_size_y, domain_size_x)

# Generate particles

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
    torch.lt(distance_between_particle_and_circle_center,(Mixer_radius-10)*cell_size)&
    torch.gt(distance_between_particle_and_circle_center,10*cell_size),    
    torch.tensor(1, device=device), 
    torch.tensor(0.0, device=device))

mask_particle_group_1 = mask.copy()

i= np.meshgrid(np.arange(1, domain_size_x*2))
i = i[0]
boundary_x = cell_size*(half_domain_size_x-1)*(1+np.sin(2*3.1415926*i/domain_size_x/2))
boundary_y = cell_size*(half_domain_size_y-1)*(1+np.cos(2*3.1415926*i/domain_size_y/2))

cell_x = np.round(boundary_x/cell_size).astype(int)
cell_y = np.round(boundary_y/cell_size).astype(int)

for k in range(1, 9):
    x_grid[0, 0, k*2, cell_y, cell_x] = boundary_x
    y_grid[0, 0, k*2, cell_y, cell_x] = boundary_y
    z_grid[0, 0, k*2, cell_y, cell_x] = k * cell_size * 2
    mask  [0, 0, k*2, cell_y, cell_x] = 1
    mask_particle_group_2[0, 0, k*2, cell_y, cell_x] = 1
    mask_particle_group_3[0, 0, k*2, cell_y, cell_x] = 1

mask_particle_group_1[mask_particle_group_2 == 1] = 0
'''
angular_velocity_z[mask_particle_group_2 == 1] = Mixer_anuglar_velocity_z * Mixer_radius
'''

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

save_name = "initialization_results/"
torch.save(x_grid, save_name + 'x_grid.pt')
torch.save(y_grid, save_name + 'y_grid.pt')
torch.save(z_grid, save_name + 'z_grid.pt')
torch.save(vx_grid, save_name + 'vx_grid.pt')
torch.save(vy_grid, save_name + 'vy_grid.pt')
torch.save(vz_grid, save_name + 'vz_grid.pt')
torch.save(angular_velocity_x, save_name + 'angular_velocity_x.pt')
torch.save(angular_velocity_y, save_name + 'angular_velocity_y.pt')
torch.save(angular_velocity_z, save_name + 'angular_velocity_z.pt')
torch.save(mask, save_name + 'mask.pt')
torch.save(mask_particle_group_1, save_name + 'mask_particle_group_1.pt')
torch.save(mask_particle_group_2, save_name + 'mask_particle_group_2.pt')
torch.save(mask_particle_group_3, save_name + 'mask_particle_group_3.pt')

# 创建散点图
plt.figure(figsize=(8, 6))  # 设置图形大小
xp=x_grid[mask!=0]
yp=y_grid[mask!=0]
c_color=x_grid[mask!=0]
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
save_name = "2D_new.jpg"
plt.savefig(save_name, dpi=1000, bbox_inches='tight')


mask_plot = torch.where((x_grid*mask)!=0, 1, 0)
print(mask_plot.shape)

zp, yp, xp = torch.where(mask_plot[0, 0,:,:,:] == 1)
xp = xp.cpu().numpy()
yp = yp.cpu().numpy()
zp = zp.cpu().numpy()



print(mask.shape)
print(mask_plot.shape[2])
print(mask_plot.shape[1])
print(mask_plot.shape[0])

x_lim = mask_plot.shape[4] * cell_size
y_lim = mask_plot.shape[3] * cell_size
z_lim = mask_plot.shape[2] * cell_size
print(x_lim)
print(y_lim)
print(z_lim)


ratio_x =x_lim/z_lim
ratio_y =y_lim/z_lim
print(ratio_x,ratio_y)
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
sc = ax.scatter(xp,yp,zp, facecolors='#C0C0C0', edgecolors='black', s=S_graph, linewidths=0.2)
ax = plt.gca()
ax.set_xlim([0, domain_size_x])
ax.set_ylim([0, domain_size_y])
ax.set_zlim([0, domain_size_z])
ax.view_init(elev=30, azim=45)
ax.set_box_aspect([ratio_x, ratio_y, 1])
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
save_name = "3D_new.jpg"

plt.savefig(save_name, dpi=1000, bbox_inches='tight')

