import glm
import random


def generate_voxel_positions(width, height, depth):
    block_size = 1.0
    data = []
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                if random.randint(0, 1000) < 5:
                    data.append([x*block_size - width/2, y*block_size, z*block_size - depth/2])
    return data


def generate_grid(width, depth):
    block_size = 1.0
    data = []
    for x in range(width):
        for z in range(depth):
            data.append([x*block_size - width/2, -block_size, z*block_size - depth/2])
    return data


def get_cam_pos():
    return [[-64, 64, -64], [64, 64, -64], [64, 64, 64], [-64, 64, 64]]
