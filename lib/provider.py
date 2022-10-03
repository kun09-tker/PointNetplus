import numpy as np

def shift_point_cloud(batch_data, shift_range=0.025):
    try:
        idx = batch_data.tolist().index([0.0,0.0,0.0])
    except:
        idx = batch_data.shape[0]
    random = np.random.uniform(-shift_range, shift_range, idx)
    shifts = []
    for r in random:
        shifts.append([r,r,r]) 
    for batch_index in range(idx):
        batch_data[batch_index,:] += shifts[batch_index]
    return batch_data

def rotation_point_cloud(batch_data):
    rotation_angle = np.random.uniform(-10,10) * np.pi / 180
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[1, 0, 0],
                                [0, cosval, -sinval],
                                [0, sinval, cosval]])
    rotated_data = np.dot(batch_data, rotation_matrix)
    return rotated_data

def random_noise_point_cloud(batch_data, scale = 0.015):
    try:
        idx = batch_data.tolist().index([0.0,0.0,0.0])
    except:
        idx = batch_data.shape[0]
    noise = np.zeros(batch_data.shape)
    noise[:idx] = np.random.normal(0, scale, ((idx, 3)))
    
    noisy_pointcloud = batch_data + noise
    return  noisy_pointcloud