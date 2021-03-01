# Inspired by sgan project from https://github.com/agrimgupta92/sgan

import os
import re
import math

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import read_file, rotate, shift, scale, change_perspective, prepare_goal, add_goal_to_trajectories, add_dummy_goal


def seq_collate(data):
    """ Transform data format to LSTM tensor format

    Args:
        See TrajectoryDataset

    Returns:
        obs_traj: trajectories of pedestrians in the selected sequence
                (obs_length, batch_size * num_peds_in_seq, 2)
        gt_traj: future trajectories of pedestrians in the selected sequence
                (pred_length, batch_size * num_peds_in_seq, 2)
        obs_traj_rel: relative trajectories of pedestrians in the selected sequence
                (obs_length, batch_size * num_peds_in_seq, 2)
        gt_traj_rel:  future relative trajectories of pedestrians in the selected sequence
                (pred_length, batch_size * num_peds_in_seq, 2)
        loss_mask: specify whether the pedestrians appear in each frame of the sequence
                (batch_size * num_peds_in_seq, seq_length)
        seq_start_end:  specify the position of each sequence in the flattened matrix
                (batch_size, 2)
        frame_idx: specify the frame corresponding to the pedestrian in the sequence
                (seq_len, batch_size * num_peds_in_seq)
        cluster_label:
                (batch_size * num_peds_in_seq
        datasets:
                (batch_size

    """
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list,
     loss_mask_list, frame_idx, ped_idx,
     cluster_label, destinations, datasets) = zip(*data)

    _len = [seq.size(1) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size

    obs_traj = torch.cat(obs_seq_list, dim=1)
    pred_traj = torch.cat(pred_seq_list, dim=1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=1)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    frame_idx = torch.cat(frame_idx, dim=0)
    ped_idx = torch.cat(ped_idx, dim=0)
    cluster_label = torch.cat(cluster_label, dim=0)

    # datasets = [x for x in datasets]
    if destinations[0] is None or len(destinations[0]) == 0:
        destinations = None

    seq_start_end = torch.LongTensor(seq_start_end)
    out = [
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel,
        loss_mask, seq_start_end, frame_idx, ped_idx,
        cluster_label, destinations, datasets
    ]

    return tuple(out)

def dest_4point_to_2point(dest):
    num_dest = len(dest)
    tmp = dest.reshape(num_dest, -1, 2)
    xmin, _ = torch.min(tmp[:, :, 0], dim=1)
    xmax, _ = torch.max(tmp[:, :, 0], dim=1)
    ymin, _ = torch.min(tmp[:, :, 1], dim=1)
    ymax, _ = torch.max(tmp[:, :, 1], dim=1)
    
    return torch.stack([xmin, ymin, xmax, ymax], dim=-1)
    


# TODO: Add support for returning frameId and video name
# TODO: Add support for reading frame
class TrajectoryDataset(Dataset):
    """ Dataloader for the trajectory dataset
    """

    def __init__(self, data_dir, obs_len=8, pred_len=12, skip=1,
                 min_ped=1, drop=0, use_goal=False,
                 noise_thresh=None, dummy_goal=True,
                 training=True, exclude=None,
                 rotate=False, shift=False, scale=False, perspective=False,
                 get_perpsective_matrix=False,
                 **kwargs):
        """
        Args:
            data_dir: Directory containing dataset files in the format
                    <frame_id> <ped_id> <x> <y>
            obs_len: Number of time-steps in input trajectories
            pred_len: Number of time-steps in output trajectories
            skip: Number of frames to skip while making the dataset (1 = no skip)
            min_ped: Minimum number of pedestrian that should be in a sequence

        Return:
            obs_traj: trajectories of pedestrians in the selected sequence
                    (num_peds_in_seq, 2, obs_length)
            gt_traj: future trajectories of pedestrians in the selected sequence
                    (num_peds_in_seq, 2, pred_length)
            obs_traj_rel: relative trajectories of pedestrians in the selected sequence
                    (num_peds_in_seq, 2, obs_length)
            gt_traj_rel:  future relative trajectories of pedestrians in the selected sequence
                    (num_peds_in_seq, 2, pred_length)
            loss_mask: specify whether the pedestrians appear in each frame of the sequence
                    (num_peds_in_seq, seq_length)
            frame_idx: specify the frame corresponding to the pedestrian in the sequence
                    (num_peds_in_seq, seq_length
            cluster_label:
                    (num_peds_in_seq)
            dataset:
                    (num_peds_in_seq)

        """
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.min_pred_len = 8
        self.pred_len = pred_len
        self.skip = skip

        self.output_len = self.obs_len + self.pred_len

        self.drop = drop
        self.ped_idx_start_end = {}
        self.total_num_ped = 0
        self.destination = {}

        # Setting
        self.rotate = rotate
        self.shift = shift
        self.scale = scale
        self.perspective = perspective
        self.dummy_goal = dummy_goal
        self.get_perpsective_matrix = get_perpsective_matrix

        if dummy_goal and noise_thresh is None:
            noise_thresh = 1

        self.noise_thresh = noise_thresh

        # Load the files
        # all_files = sorted(os.listdir(data_dir))
        all_files = os.listdir(data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]

        # Prepare goal if specified
        self.use_goal = use_goal
        if use_goal:
            all_goals = prepare_goal()

        dataset_in_seq = []

        cluster_label_in_seq = []
        frame_idx_in_seq = []
        ped_idx_in_seq = []
        num_peds_in_seq = []

        # (num_seq * num_peds_in_each_seq, 2, seq_length)
        seq_list = []
        seq_list_rel = []

        # (num_seq * num_peds_in_each_seq, seq_length)
        loss_mask_list = []

        exclude_pattern = None
        if exclude is not None:
            exclude_pattern = exclude.split(",")
            exclude_pattern = ["({})".format(n.strip())
                               for n in exclude_pattern if n != ""]
            exclude_pattern = "|".join(exclude_pattern)

        for path in all_files:
            dataset = path.split("/")[-1].replace(".txt", "").replace(
                "_train", "").replace("_test", "").replace("_val", "")

            if (training is True) and (exclude_pattern is not None):
                if re.search(exclude_pattern, dataset) is not None:
                    continue

            # data is of shape (n_frames * n_peds_per_frame, 4)
            data = read_file(path)

            if use_goal:
                destination = all_goals[dataset]
                data = add_goal_to_trajectories(data, destination)
                num_dest = destination.shape[0]
                # self.destination[dataset] = dest_4point_to_2point(torch.Tensor(destination))
                self.destination[dataset] = torch.Tensor(destination)

            # if use_goal and training:
            #     data = data[data[:, -1] != -1]

            # Drop some data if drop is specify
            new_len = int(len(data) * (1 - self.drop))
            start_seq_idx = len(data) - new_len
            if start_seq_idx != 0:
                start_seq_idx = np.random.randint(0, start_seq_idx)

            data = data[start_seq_idx:(start_seq_idx + new_len)]

            frames = np.unique(data[:, 0]).tolist()

            # frame_data is of shape (n_frames, num_peds_in_frame, 4)
            frame_data = []

            for frame in frames:
                frame_data.append(data[data[:, 0] == frame, :])

            num_sequences = int(
                math.ceil((len(frames) - self.output_len + 1) / skip))

            dataset_local_ped_idx = []
            total_num_frame = len(frames)

            local_considered_idx = []

            for idx in range(0, num_sequences * skip + 1, skip):
                # curr_seq_data is of shape (num_frames_in_seq * num_peds_each_frame, 4)
                start_frame = idx
                end_frame = idx + self.output_len

                if end_frame > total_num_frame:
                    end_frame = total_num_frame

                curr_seq_data = np.concatenate(
                    frame_data[start_frame:end_frame], axis=0)

                # Find number of pedestrians in current sequence
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])

                # curr_seq_rel store the relative coordinates of each pedestrian in current sequence
                curr_seq_rel = np.zeros(
                    (len(peds_in_curr_seq), 2, self.output_len))
                curr_seq = np.zeros(
                    (len(peds_in_curr_seq), 2, self.output_len))

                # store the frame idx in which the pedestrian appear

                # curr loss mask store the boolean value for computing losses in the whole dataset
                curr_loss_mask = np.zeros(
                    (len(peds_in_curr_seq), self.output_len))

                curr_frame_idx = np.zeros(
                    (len(peds_in_curr_seq), self.output_len)) - 1
                curr_ped_idx = np.zeros(len(peds_in_curr_seq)) - 1

                curr_cluster_label = np.zeros(len(peds_in_curr_seq))

                curr_dataset = []

                num_peds_considered = 0

                for _, ped_id in enumerate(peds_in_curr_seq):
                    # curr_ped_seq has shape (num_seqs_has_ped, 4)

                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1]
                                                 == ped_id, :]

                    # Round float to have 4 decimal point
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)

                    curr_label = curr_ped_seq[:, -1][-1]

                    curr_ped_seq = curr_ped_seq[:, :4]

                    if len(curr_ped_seq) != self.output_len:
                        continue

                    # compute how many time the person appear in the sequence
                    if (noise_thresh is not None):
                        # if (training is True) and (noise_thresh is not None):
                        start_obs = curr_ped_seq[0, 2:4]
                        end_obs = curr_ped_seq[obs_len - 1, 2:4]
                        dis = np.sqrt(np.sum((end_obs - start_obs)**2))

                        if dis < noise_thresh and not dummy_goal:
                            continue
                        elif dis < noise_thresh and dummy_goal and use_goal:
                            curr_label = num_dest

                    local_considered_idx.append(ped_id)

                    # (2, num_seqs_has_ped)
                    curr_ped_coord_seq = np.transpose(curr_ped_seq[:, 2:])

                    # Make coordinates relative (to their previous coordinates)
                    rel_curr_ped_seq = np.zeros(curr_ped_coord_seq.shape)
                    rel_curr_ped_seq[:, 1:] = curr_ped_coord_seq[:,
                                                                 1:] - curr_ped_coord_seq[:, :-1]

                    # Store the coordinate of current pedestrian
                    _idx = num_peds_considered
                    curr_seq[_idx, :] = curr_ped_coord_seq
                    curr_seq_rel[_idx, :] = rel_curr_ped_seq

                    curr_loss_mask[_idx] = 1

                    curr_frame_idx[_idx] = np.transpose(
                        curr_ped_seq[:, 0])
                    curr_ped_idx[_idx] = ped_id

                    curr_cluster_label[_idx] = int(curr_label)

                    curr_dataset.append(dataset)

                    num_peds_considered += 1

                if num_peds_considered >= min_ped:
                    num_peds_in_seq.append(num_peds_considered)

                    # WARNING: These line of code reduce the number of sequence by half
                    frame_idx_in_seq.append(
                        curr_frame_idx[:num_peds_considered])
                    ped_idx_in_seq.append(curr_ped_idx[:num_peds_considered])
                    dataset_local_ped_idx.append(
                        curr_ped_idx[:num_peds_considered])

                    cluster_label_in_seq.append(
                        curr_cluster_label[:num_peds_considered])
                    dataset_in_seq.append(dataset)

                    # TODO: Find what does this variable mean
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])

                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])

            if len(dataset_local_ped_idx) > 0:
                dataset_local_ped_idx = np.concatenate(dataset_local_ped_idx)
                self.ped_idx_start_end[dataset] = np.min(
                    dataset_local_ped_idx).item(), np.max(dataset_local_ped_idx).item()

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)

        # WARNING: currently loss mask list is a long 1 sequence since the "current_loss_mask" only accept
        # pedestrian that appear in the whole sequence
        # TODO: reconsider this
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)

        frame_idx_in_seq = np.concatenate(frame_idx_in_seq, axis=0)
        ped_idx_in_seq = np.concatenate(ped_idx_in_seq, axis=0)
        cluster_label_in_seq = np.concatenate(cluster_label_in_seq, axis=0)

        # Convert Numpy array to Torch tensor

        self.mean = torch.from_numpy(
            np.mean(seq_list, axis=(0, 2))).type(torch.float)
        self.std = torch.from_numpy(
            np.std(seq_list, axis=(0, 2))).type(torch.float)

        # (num_seq * num_peds_in_each_seq, 2, obs_length)
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.num_trajectory = len(self.obs_traj)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)

        # (num_seq * num_peds_in_each_seq, 2, pred_length)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)

        # (num_seq * num_peds_in_each_seq, seq_length)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)

        self.frame_idx = torch.from_numpy(frame_idx_in_seq).type(torch.float)
        self.ped_idx = torch.from_numpy(ped_idx_in_seq)
        self.cluster_label = torch.from_numpy(
            cluster_label_in_seq).type(torch.float)
        self.datasets = dataset_in_seq

        # Specify the position of each sequence in the flattened matrix of sequences
        # The index is the id of the sequence, start and stop are the position of
        # each sequence in each the flattened array
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]

        self.computed_ped_idx_start_end()
        self.ped_count = self.compute_ped_count()
        
        dests = []
        obs_traj = []
        pred_traj = []
        obs_traj_rel = []
        pred_traj_rel = []
        transform_matrix = []
        for i in range(self.num_seq):
            start, end = self.seq_start_end[i]
            new_x_traj, new_y_traj, new_x_rel, new_y_rel, new_dest, m = self.transform(i)
            
            obs_traj.append(new_x_traj)
            pred_traj.append(new_y_traj)
            obs_traj_rel.append(new_x_rel)
            pred_traj_rel.append(new_y_rel)
            dests.append(new_dest)
            transform_matrix.append(m)
            
        self.obs_traj = obs_traj
        self.pred_traj = pred_traj
        self.obs_traj_rel = obs_traj_rel
        self.pred_traj_rel = pred_traj_rel
        self.dests = dests   
        self.transform_matrix = transform_matrix         
        

    def __len__(self):
        return self.num_seq
    
    def transform(self, index):
        start, end = self.seq_start_end[index]
        current_dataset = self.datasets[index]

        current_destination = None

        if self.use_goal:
            current_destination = self.destination[current_dataset]
        (x_traj, y_traj,
         x_rel, y_rel) = (self.obs_traj[start:end].permute(2, 0, 1),
                          self.pred_traj[start:end].permute(2, 0, 1),
                          self.obs_traj_rel[start:end].permute(2, 0, 1),
                          self.pred_traj_rel[start:end].permute(2, 0, 1))

        if ((not self.rotate) and
            (not self.shift) and
            (not self.scale) and
            (not self.perspective) and
                (not self.dummy_goal)):

            return (
                x_traj, y_traj, x_rel, y_rel, current_destination, None
            )
        else:
            x_traj = x_traj.clone()
            y_traj = y_traj.clone()

            dest = None
            
            if self.use_goal:
                dest = current_destination.clone()

            if self.rotate:
                x_traj, dest, angle = rotate(x_traj, dest)
                y_traj, _, _ = rotate(y_traj, None, angle)

            if self.shift:
                x_traj, dest, offset_x, offset_y = shift(x_traj, dest)
                y_traj, _, _, _ = shift(y_rel, None, offset_x, offset_y)

            if self.scale:
                x_traj, alpha = scale(x_traj)
                dest, _ = scale(dest, alpha)
                y_traj, _ = scale(y_traj, alpha)

            if self.perspective:
                if not self.get_perpsective_matrix:
                    m = None
                    x_traj, y_traj, dest = change_perspective(x_traj, y_traj, dest)
                else:
                    x_traj, y_traj, dest, m = change_perspective(x_traj, y_traj, dest, get_m=self.get_perpsective_matrix)
            # Add dummy goal if specified
            if self.dummy_goal:
                dest = add_dummy_goal(x_traj[-1], self.noise_thresh / 2, dest)

            x_rel = torch.zeros_like(x_traj)
            y_rel = torch.zeros_like(y_traj)

            x_rel[1:] = x_traj[1:] - x_traj[:-1]
            y_rel[1:] = y_traj[1:] - y_traj[:-1]
            y_rel[0] = y_traj[0] - x_traj[-1]

            return (
                x_traj, y_traj, x_rel, y_rel, dest, m
            )


    def get_num_traj(self):
        return self.num_trajectory

    def computed_ped_idx_start_end(self):
        ped_idx_offset = 0
        for dataset_name, (ped_idx_start, ped_idx_end) in self.ped_idx_start_end.items():
            start = ped_idx_offset
            end = ped_idx_offset + ped_idx_end + 1
            self.ped_idx_start_end[dataset_name] = (int(start), int(end))
            ped_idx_offset = end

        self.total_num_ped = int(ped_idx_offset)

    def compute_ped_count(self):
        ped_count = torch.zeros((self.total_num_ped)).cuda()
        for index, (start, end) in enumerate(self.seq_start_end):
            dataset_name = self.datasets[index]
            seq_ped_idx = self.ped_idx[start:end]
            ped_idx_offset, _ = self.ped_idx_start_end[dataset_name]
            updated_seq_ped_idx = seq_ped_idx + ped_idx_offset

            ped_count[updated_seq_ped_idx.long()] += 1

        return ped_count

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        current_dataset = self.datasets[index]

        x_traj = self.obs_traj[index]
        y_traj = self.pred_traj[index]
        x_rel = self.obs_traj_rel[index]
        y_rel = self.pred_traj_rel[index]
        dest = self.dests[index]

        return (
            x_traj, y_traj, x_rel, y_rel,
            self.loss_mask[start:end], self.frame_idx[start:end],
            self.ped_idx[start:end], self.cluster_label[start:end],
            dest, current_dataset
        )


    def get_raw_trajectory(self, current_dataset, selected_frame_idx, selelcted_ped_idx):
        index = np.array(self.datasets) == current_dataset
        current_seq_start_end = np.array(self.seq_start_end)[index]
        start, end = np.min(current_seq_start_end), np.max(
            current_seq_start_end)
        current_index = (self.frame_idx[start:end, 0] == selected_frame_idx) & (
            self.ped_idx[start:end] == selelcted_ped_idx)
        current_x_traj, current_y_traj = self.obs_traj[start:
                                                       end][current_index], self.pred_traj[start:end][current_index]
        current_x_rel, current_y_rel = self.obs_traj_rel[start:end][
            current_index], self.pred_traj_rel[start:end][current_index]
        current_loss_mask = self.loss_mask[start:end][current_index]
        current_frame_idx = self.frame_idx[start:end][current_index]
        current_ped_idx = self.ped_idx[start:end][current_index]
        current_label = self.cluster_label[start:end][current_index]
        current_destination = self.destination[current_dataset]

        current_x_traj, current_y_traj = current_x_traj.permute(
            2, 0, 1), current_y_traj.permute(2, 0, 1)
        current_x_rel, current_y_rel = current_x_rel.permute(
            2, 0, 1), current_y_rel.permute(2, 0, 1)

        return (
            current_x_traj, current_y_traj,
            current_x_rel, current_y_rel,
            current_loss_mask, current_frame_idx,
            current_ped_idx, current_label,
            current_destination, current_dataset
        )
