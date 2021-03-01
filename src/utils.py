import re
import sys
import os
from pathlib import Path
import datetime
import json

import cv2
import torch
from torch.utils.data import DataLoader
import numpy as np
import logging
from torch.utils.tensorboard import SummaryWriter

from src.features.trajectory import TrajectoryDataset, seq_collate
from .shared import LOG_PATH, TENSORBOARD_PATH, CHECKPOINT_PATH, DATA_PATH

FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)


# Dataset related
def get_trajectory_path():
    return os.path.join(DATA_PATH, "text")

def get_segmentation_path(data_type):
    return os.path.join(DATA_PATH, "segmentation")

# --------------------------------- Train-related utilities ------------------------
def prepare_full_checkpoint_name(args, now=None):
    # Get current time for distinguishing between files
    if now is None:
        now = datetime.datetime.now()
    file_suffix = now.strftime("%m-%d-%Y-%H-%M-%S")

    return "obs{}_pred{}_{}_{}_{}".format(args.obs_len,
                                          args.pred_len,
                                          args.data_name,
                                          args.checkpoint_name,
                                          file_suffix)

def prepare_model_report(args, checkpoint_name):
    # Check if report folder is exist
    # If not, create them
    if not os.path.isdir(LOG_PATH):
        os.mkdir(LOG_PATH)

    if not os.path.isdir(TENSORBOARD_PATH):
        os.mkdir(TENSORBOARD_PATH)

    # prepare logger file
    logger_filename = "{}.txt".format(checkpoint_name)
    logger_path = os.path.join(LOG_PATH, logger_filename)
    logger = logging.getLogger(logger_path)

    if args.save_log:
        fh = logging.FileHandler(logger_path)
        logger.addHandler(fh)

    # prepare tensorboard
    tensorboard_writer = None
    if args.save_log:
        tensorboard_path = os.path.join(
            TENSORBOARD_PATH, checkpoint_name)
        if not os.path.isdir(tensorboard_path):
            os.mkdir(tensorboard_path)

        tensorboard_writer = SummaryWriter(log_dir=tensorboard_path)

    return logger, tensorboard_writer


def set_require_grad(model, flag):
    for p in model.parameters():
        p.requires_grad = flag


def to_cuda(batch):
    dataset = batch[-1]
    destination = batch[-2]

    if destination is not None:
        destination = [tensor.cuda() for tensor in destination]

    batch = batch[:-2]

    batch = [tensor.cuda() for tensor in batch]
    batch.append(destination)
    batch.append(dataset)

    return batch


def get_sub_batch(batch, idx):
    dataset = batch[-1]
    destination = batch[-2]

    sub_batch_1 = batch[:4]
    sub_batch_2 = batch[:-2]

    sub_batch_1 = [tensor[:, idx] for tensor in sub_batch_1]
    sub_batch_2 = [tensor[idx, :] for tensor in sub_batch_2]
    sub_dataset = dataset[idx.item()]

    if len(destination.shape) == 3:
        sub_destination = destination[:, idx]
    else:
        sub_destination = destination

    sub_batch = sub_batch_1 + sub_batch_2 + [sub_destination] + [sub_dataset]


def get_trajectories_from_batch(batch):
    (all_obs_traj, all_gt_traj, all_obs_traj_rel, all_gt_traj_rel,
     all_loss_mask, all_seq_start_end) = batch[:6]

    return (all_obs_traj, all_gt_traj, all_obs_traj_rel, all_gt_traj_rel,
            all_loss_mask, all_seq_start_end)


def get_additional_info_from_batch(batch):

    (all_frame_idx, all_ped_idx, all_labels,
     all_destination, dataset) = batch[6:]

    return all_frame_idx, all_ped_idx, all_labels, all_destination, dataset


def relative_to_abs(rel_traj, start_pos):
    """
    Inputs:
    - rel_traj: pytorch tensor of shape (seq_len, batch, 2)
    - start_pos: pytorch tensor of shape (batch, 2)
    Outputs:
    - abs_traj: pytorch tensor of shape (seq_len, batch, 2)
    """
    # batch, seq_len, 2
    rel_traj = rel_traj.permute(1, 0, 2)
    displacement = torch.cumsum(rel_traj, dim=1)
    start_pos = torch.unsqueeze(start_pos, dim=1)
    abs_traj = displacement + start_pos
    return abs_traj.permute(1, 0, 2)


def seq_from_batch_generator(batch):
    """Return a generator over sequences in a batch
    As a batch contains multiple sequences from multiple dataset, this function
    return a generator that allows models to operate on each sequence.

    Arguments:
        batch

    Yields:
        seq_x_origin
        seq_y_origin
        seq_x_rel
        seq_y_rel
        seq_loss_mask
        seq_dataset,
        seq_destinations
        seq_labels
        start: start index of the sequence in the batch
        end: end index of the sequence in the batch
        seq_start_frame
        seq_ped_idx 
    """

    (x_origin, y_origin, x_rel, y_rel,
     loss_mask, seq_start_end) = get_trajectories_from_batch(batch)

    (all_frame_idx, all_ped_idx, all_labels, all_destination,
     dataset) = get_additional_info_from_batch(batch)

    for i, (start, end) in enumerate(seq_start_end.data):
        (seq_x_origin,
         seq_y_origin,
         seq_x_rel,
         seq_y_rel) = (x_origin[:, start:end].contiguous(),
                       y_origin[:, start:end].contiguous(),
                       x_rel[:, start:end].contiguous(),
                       y_rel[:, start:end].contiguous())

        seq_dest = None
        if all_destination is not None:
            seq_dest = all_destination[i]

        (seq_dataset,
         seq_label) = (dataset[i],
                       all_labels[start:end])

        seq_loss_mask = loss_mask[start:end]
        seq_ped_idx = all_ped_idx[start:end]
        seq_start_frame = all_frame_idx[start:end, 0]

        yield (seq_x_origin, seq_y_origin,
               seq_x_rel, seq_y_rel,
               seq_loss_mask, seq_dataset,
               seq_dest, seq_label,
               start, end, seq_start_frame,
               seq_ped_idx)


def prepare_datasets(batch_size, data_name,
                     rotate=False, shift=False, scale=False,
                     shuffle=False, test=False, **kwargs):
    
    trajectory_path = get_trajectory_path()

    if not test:
        # Train loader
        train_path = os.path.join(trajectory_path, data_name, "train")
        train_dataset = TrajectoryDataset(data_dir=train_path,
                                        rotate=rotate, shift=shift, scale=scale,
                                        training=True, **kwargs)
        train_loader = DataLoader(train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                collate_fn=seq_collate)

        # Validation loader
        val_path = os.path.join(trajectory_path, data_name, "val")
        val_dataset = TrajectoryDataset(data_dir=val_path,
                                        rotate=False, shift=False, scale=False,
                                        training=False, **kwargs)
        val_loader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                collate_fn=seq_collate)

        # Test loader
        test_dataset, test_loader = None, None
    else:
        train_dataset, train_loader, val_dataset, val_loader = None, None, None, None
        
        test_path = os.path.join(trajectory_path, data_name, "test")
        test_dataset = TrajectoryDataset(data_dir=test_path,
                                         rotate=False, shift=False, scale=False,
                                         training=False, **kwargs)
        test_loader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 collate_fn=seq_collate)

    return (train_dataset, train_loader), (val_dataset, val_loader), (test_dataset, test_loader)


def info(args, logger, train_dataset, val_dataset, test_dataset, model):
        # Log info about the arguments
    logger.info("Here is the arguments")
    logger.info(json.dumps(args, indent=2))

    # Log info about the dataset
    logger.info('There are {} seq, {} trajectories in the train dataset'.format(
        len(train_dataset), train_dataset.get_num_traj()))
    logger.info('There are {} seq, {} trajectories in the validation dataset'.format(
        len(val_dataset), val_dataset.get_num_traj()))

    if test_dataset is not None:
        logger.info('There are {} seq, {} trajectories in the test dataset'.format(
            len(test_dataset), test_dataset.get_num_traj()))

    # Log info about the training process
    iterations_per_epoch = len(train_dataset) / args.batch_size
    logger.info('There are {} iterations per epoch'.format(
        iterations_per_epoch))

    # Log model
    logger.info("Here is the model")
    logger.info(model)
    num_params = sum(p.numel()
                     for p in model.parameters() if p.requires_grad)
    logger.info("Number of parameters: {}".format(num_params),)


def number_to_one_hot(label, num_classes):
    one_hot_label = torch.zeros((label.size(0), num_classes))

    for i, current_label in enumerate(label):
        one_hot_label[i, current_label] = 1

    return one_hot_label.to(label.device)


def get_checkpoint(checkpoint_name):
    return os.path.join(CHECKPOINT_PATH, "{}.pt".format(checkpoint_name))
