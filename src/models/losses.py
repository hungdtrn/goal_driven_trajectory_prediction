import torch
import torch.nn as nn
import torch.nn.functional as F
import random

bce = nn.BCELoss(reduction='none')
mse = nn.MSELoss()
kl_loss = nn.KLDivLoss(reduction='none')

def bce_loss(score, label, mode='sum'):
    raw_loss = bce(score, label)

    if mode == 'sum':
        return torch.sum(raw_loss)
    elif mode == 'average':
        return torch.mean(raw_loss)
    elif mode == 'raw':
        return torch.mean(raw_loss, dim=1)

    return raw_loss


def mse_loss(src, target):
    return mse(src, target)

def attention_loss(src, target):
    """
    
    Args:
        src: The attention score
        target: The output of Goal channel, goal score
        
    """
    
    # compute log score
    log_attention = torch.log(src)
    
    return kl_loss(log_attention, target)


def vae_loss(pred_traj, pred_traj_gt, mu_m, logvar_m, loss_mask, mode='average'):
    """

    Args:
        pred_traj: see l2_loss
        pred_traj_gt: see l2_loss
        mu_m: Mean of the distribution from which z is sampled (batch, zm_dim)
        logvar_m: Logarit of variance from which z is sampled (batch, zm_dim)
        loss_mask: see l2_loss
        mode: see l2_loss

    Returns:
        l2_loss: reconstruction loss (batch,)
        dl_loss: regularization loss (batch,)

    """

    # Compute reconstruction loss
    reconstruction_loss = l2_loss(pred_traj, pred_traj_gt, loss_mask, mode)

    # KL loss
    dl_loss = -0.5 * torch.sum(1 + logvar_m -
                               mu_m.pow(2) - logvar_m.exp(), dim=1)

    return reconstruction_loss, dl_loss


def l2_loss(pred_traj, pred_traj_gt, loss_mask, mode='raw'):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    predictions.
    - loss_mask: Tensor of shape (batch, seq_len)
    - mode: Can be one of sum, average, raw
    Output:
    - loss: l2 loss depending on mode
    """
    seq_len, batch, _ = pred_traj.size()
    loss = (loss_mask.unsqueeze(dim=2) *
            (pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2))**2)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'average':
        return loss.sum(dim=2).sum(dim=1) / torch.sum(loss_mask, dim=1)
    elif mode == 'raw':
        return loss.sum(dim=2).sum(dim=1)
    elif mode == 'sqrt':
        return torch.sqrt(loss.sum(dim=2))


def displacement_error(pred_traj, pred_traj_gt, loss_mask, consider_ped=None, mode='sum'):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth
    predictions.
    - consider_ped: Tensor of shape (batch)
    - mode: Can be one of sum, raw
    Output:
    - loss: gives the eculidian displacement error
    """
    seq_len, _, _ = pred_traj.size()

    # sum of all loss of all timesteps of all ped
    loss = l2_loss(pred_traj, pred_traj_gt, loss_mask, mode='sqrt')

    # print(loss)
    # average over sequence
    loss = torch.sum(loss, dim=1) / torch.sum(loss_mask, dim=1)

    if consider_ped is not None:
        loss = loss * consider_ped

    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'average':
        return torch.mean(loss)
    elif mode == 'raw':
        return loss


def final_displacement_error(
    pred_pos, pred_pos_gt, loss_mask, consider_ped=None, mode='sum'
):
    """
    Input:
    - pred_pos: Tensor of shape (seq_len, batch, 2). Predicted pos.
    - pred_pos_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    last pos
    - consider_ped: Tensor of shape (batch)
    Output:
    - loss: gives the eculidian displacement error
    """
    loss = pred_pos_gt[-1] - pred_pos[-1]
    loss = loss**2

    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=1)) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=1))

    if mode == 'raw':
        return loss
    else:
        return torch.sum(loss)
