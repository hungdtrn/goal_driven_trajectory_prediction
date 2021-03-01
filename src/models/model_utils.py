import torch
import torch.nn as nn
import math

# ----------------------------------Model-related utilities ------------------------
def make_mlp(dim_list, activation='leakyrelu', batch_norm=True, dropout=0.0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        elif activation == 'prelu':
            layers.append(nn.PReLU())
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)

def torch_dest_repeat(seq_destinations, batch_size):
    """ Repeate destination match with the number of pedestrians

    Arguments:
        seq_destinations {Torch Tensor} -- Size of (num_dest, dest_dim)
        batch_size {Number} -- Specify the number of sequence in batch

    Returns:
        Torch Tensor -- The result of size (num_dest, batch_size, dest_dim)
    """
    # from (n_dest x 8) to (n_dest x batch x 8)
    dest_rep = seq_destinations.unsqueeze(dim=1).repeat(1, batch_size, 1)

    return dest_rep

def compute_rel_dest(obs_trajectory, seq_destinations):
    """Compute the new location of each destination based on the perspective of the pedestrian

    Arguments:
        obs_trajectory -- Torch Tensor of size (batch_size, 2)
        seq_destinations -- Torch Tensor of size (num_dest, batch_size, 8)

    Returns:
        rel_destinations -- Torch tensor of size (num_dest, batch_size, 8)
    """

    # (batch_size, 2)
    last_obs = obs_trajectory[-1]
    num_dest, batch_size, dest_dim = seq_destinations.shape

    # ----------------- Duplciate destinations-----------------------

    # From (num_dest, batch_size, 8) to (num_dest, batch_size, 4, 2)
    dup_dest = seq_destinations.reshape(num_dest, batch_size, -1, 2)

    # num_points_in_dest = 4
    num_points_in_dest = dup_dest.size(-2)

    # ------------------ Duplicate last observations ----------------------
    # from (batch_size, 2) to (batch_size, 4, 2)
    dup_obs = last_obs.unsqueeze(dim=1).repeat(1, num_points_in_dest, 1)

    # from (batch_size, 4, 2) to (num_dest, batch_size, 4, 2)
    dup_obs = dup_obs.unsqueeze(dim=0).repeat(num_dest, 1, 1, 1)

    # ------------------ Compute relative destination ----------------------
    rel_dest = dup_dest - dup_obs

    # from (num_dest, batch_size, 4, 2) to (num_dest, batch_size, 8)
    rel_dest = rel_dest.reshape(num_dest, batch_size, -1)

    return rel_dest

def compute_2point_goal(seq_destinations):
    """Compute the new location of each destination based on the perspective of the pedestrian

    Arguments:
        obs_trajectory -- Torch Tensor of size (batch_size, 2)
        seq_destinations -- Torch Tensor of size (num_dest, batch_size, 8)

    Returns:
        rel_destinations -- Torch tensor of size (num_dest, batch_size, 4)
    """

    # (batch_size, 2)
    num_dest, batch_size, dest_dim = seq_destinations.shape

    # ----------------- Duplciate destinations-----------------------

    # From (num_dest, batch_size, 8) to (num_dest, batch_size, 4, 2)
    tmp = seq_destinations.reshape(num_dest, batch_size, -1, 2)

    # From (num_dest, batch_size, 4, 2) to (num_dest, batch_size)
    xmin, _ = torch.min(tmp[:, :, :, 0], dim=2)
    xmax, _ = torch.max(tmp[:, :, :, 0], dim=2)
    ymin, _ = torch.min(tmp[:, :, :, 1], dim=2)
    ymax, _ = torch.max(tmp[:, :, :, 1], dim=2)

    # From (num_dest, batch_size) to (num_dest, batch_size, 4)
    new_dest = torch.stack([xmin, ymin, xmax, ymax], dim=-1)
    
    return new_dest


def compute_angle_dest(seq_destinations):
    """
    Arguments:
        seq_destinations -- Torch Tensor of size (num_dest, batch_size, 4)

    Returns:
        angle_destinations -- Torch tensor of size (num_dest, batch_size, 2)
    """
    # (batch_size, 2)
    num_dest, batch_size, dest_dim = seq_destinations.shape
    

    # From (num_dest, batch_size, 4) to (num_dest, batch_size, 2, 2)
    tmp = seq_destinations.reshape(num_dest, batch_size, -1, 2)
    
    # ------------------- Compute angle    
    feature_first = torch.atan2(tmp[:, :, 0, 1], tmp[:, :, 0, 0])
    feature_last = torch.atan2(tmp[:, :, 1, 1], tmp[:, :, 1, 0])
    
    dup_dest = torch.stack([feature_first, feature_last], dim=2)
    return dup_dest 