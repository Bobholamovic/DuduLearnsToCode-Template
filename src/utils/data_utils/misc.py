import torch
import numpy as np
from scipy.io import loadmat
from skimage.io import imread


def default_loader(path_):
    return imread(path_)


def mat_loader(path_):
    return loadmat(path_)


def make_onehot(index_map, n):
    # Only deals with tensors with no batch dim
    old_size = index_map.size()
    z = torch.zeros(n, *old_size[-2:]).type_as(index_map)
    z.scatter_(0, index_map, 1)
    return z
    

def to_tensor(arr):
    if any(s < 0 for s in arr.strides):
        # Enforce contiguousness since currently torch.from_numpy doesn't support negative strides.
        arr = np.ascontiguousarray(arr)
    if arr.ndim < 3:
        return torch.from_numpy(arr)
    elif arr.ndim == 3:
        return torch.from_numpy(np.transpose(arr, (2,0,1)))
    else:
        raise ValueError


def to_array(tensor):
    if tensor.ndimension() <= 4:
        arr = tensor.data.cpu().numpy()
        if tensor.ndimension() in (3, 4):
            arr = np.moveaxis(arr, -3, -1)
        return arr
    else:
        raise ValueError