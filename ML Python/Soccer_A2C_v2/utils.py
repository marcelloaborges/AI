import torch

def to_tensor(np_array):
    tensor = torch.from_numpy(np_array)

    return tensor