

import os
import re

import cv2
import torch
import pickle
import numpy as np

# from src.utils import get_goal_path, get_segmentation_path
from src.shared import DATA_PATH
import src.tools.segmentation.goal_from_image as goal_extractor_tool

border = {
    "biwi_eth": [0, 120, 480, 460],
    "biwi_hotel": [0, 0, 576, 720],
    "crowds_zara01": [0, 0, 576, 720],
    "crowds_zara02": [0, 0, 576, 720],
    "students003": [0, 0, 576, 720],
}

def read_file(_path):
    data = []

    with open(_path, 'r') as f:
        for line in f:
            data.append([float(x) for x in re.split(r"\s+", line.strip())])

    return np.asarray(data)


def pixel_to_real(x, y, dataset_type):
    frame_height = 576

    if dataset_type == "biwi_eth":
        frame_height = 480
        h = np.array([[2.8128700e-02, 2.0091900e-03, -4.6693600e+00],
                      [8.0625700e-04, 2.5195500e-02, -5.0608800e+00],
                      [3.4555400e-04, 9.2512200e-05, 4.6255300e-01]])

        real_coord = np.dot(h, np.array([y, x, 1]).T)
        real_coord = (real_coord / real_coord[2])
    elif dataset_type == "biwi_hotel":
        h = np.array([[1.1048200e-02, 6.6958900e-04, -3.3295300e+00],
                      [-1.5966000e-03, 1.1632400e-02, -5.3951400e+00],
                      [1.1190700e-04, 1.3617400e-05, 5.4276600e-01]])
        real_coord = np.dot(h, np.array([y, x, 1]).T)
        real_coord = (real_coord / real_coord[2])
    else:
        pts_img = np.array([[476, 117], [562, 117], [562, 311], [476, 311]])
        pts_wrd = np.array([[0, 0], [1.81, 0], [1.81, 4.63], [0, 4.63]])
        h, status = cv2.findHomography(pts_img, pts_wrd)

        y = frame_height - y

        real_coord = np.dot(h, np.array([x, y, 0]).T)

    x, y = real_coord[0], real_coord[1]

    return x, y

def real_to_pixel(x, y, dataset_type):
    frame_height = 576
    
    if dataset_type == "biwi_eth":
        frame_height = 480
        h = np.linalg.inv([[2.8128700e-02, 2.0091900e-03, -4.6693600e+00],
                           [8.0625700e-04, 2.5195500e-02, -5.0608800e+00],
                           [3.4555400e-04, 9.2512200e-05, 4.6255300e-01]])

        pix_coord = np.dot(h, np.array([x, y, 1]).T)
        pix_coord = (pix_coord / pix_coord[2])
    elif dataset_type == "biwi_hotel":
        h = np.linalg.inv([[1.1048200e-02, 6.6958900e-04, -3.3295300e+00],
                           [-1.5966000e-03, 1.1632400e-02, -5.3951400e+00],
                           [1.1190700e-04, 1.3617400e-05, 5.4276600e-01]])
        pix_coord = np.dot(h, np.array([x, y, 1]).T)
        pix_coord = (pix_coord / pix_coord[2])
    else:
        pts_img = np.array([[476, 117], [562, 117], [562, 311], [476, 311]])
        pts_wrd = np.array([[0, 0], [1.81, 0], [1.81, 4.63], [0, 4.63]])
        h, status = cv2.findHomography(pts_img, pts_wrd)
        h = np.linalg.inv(h)

        pix_coord = np.dot(h, np.array([x, y, 0]).T)
        pix_coord = pix_coord

    x, y = pix_coord[0], pix_coord[1]
    if dataset_type == "biwi_eth" or dataset_type == "biwi_hotel":
        x, y = y, x
    else:
        y = frame_height - y


    return x, y

def add_goal_to_trajectories(trajectories, goals):
    num_goal = len(goals)

    reshaped_goal = np.reshape(goals, (num_goal, -1, 2))

    # Initialize goal label
    labels = np.zeros((len(trajectories), 1)) - 1

    offset = 0

    # Label the points that are in the boxes
    for i, c in enumerate(reshaped_goal):
        max_w, max_h = np.max(c[:, 0]), np.max(c[:, 1])
        min_w, min_h = np.min(c[:, 0]), np.min(c[:, 1])

        offset_h = offset * np.min((np.abs(min_h), np.abs(max_h)))
        offset_w = offset * np.min((np.abs(min_w), np.abs(max_w)))

        selector_h = (trajectories[:, 3] + offset_h >
                      min_h) & (trajectories[:, 3] - offset_h < max_h)
        selector_w = (trajectories[:, 2] + offset_w >
                      min_w) & (trajectories[:, 2] - offset_w < max_w)

        labels[selector_h & selector_w] = i

    result = np.concatenate([trajectories, labels], 1)

    # label sequences that lead to the point in the box
    unique_ped_idx = np.unique(result[:, 1])
    for ped_idx in unique_ped_idx:
        current_sequence = result[result[:, 1] == ped_idx]
        labeled_points = current_sequence[current_sequence[:, -1] != -1]

        if len(labeled_points) == 0:
            continue

        labeled_points = labeled_points[np.argsort(labeled_points[:, 0])]
        for lp in labeled_points:
            labeling_indexes = (current_sequence[:, 0] < lp[0]) & (
                current_sequence[:, -1] == -1)
            current_sequence[labeling_indexes, -1] = int(lp[-1].item())

        result[result[:, 1] == ped_idx] = current_sequence

    return result


def prepare_goal(to_real=True):
    goals = {}

    segmentation_path = os.path.join(DATA_PATH, "segmentation")

    for seg_file in os.listdir(segmentation_path):
        dataname = seg_file.split(".")[0]
        data_border = border[dataname]
        ymin, xmin, ymax, xmax = data_border
            
        with open(os.path.join(segmentation_path, seg_file), "rb") as f:
            seg_data = pickle.load(f)
                
        seg_data = [data[ymin:ymax, xmin:xmax] for data in seg_data]
            
        pixel_goals, _ = goal_extractor_tool.detect_goal(
            seg_data, feature_type="segmentation")
            
        if to_real:
            real_g = []

            for g in pixel_goals:
                g = g.reshape(-1, 2)
                
                g[:, 0] = g[:, 0] + xmin
                g[:, 1] = g[:, 1] + ymin
                    
                rg = [pixel_to_real(p[0], p[1], dataname) for p in g]
                rg = np.array(rg).reshape(-1)
                real_g.append(rg)

            goals[dataname] = np.array(real_g)

        else:
            goals[dataname] = pixel_goals

    return goals


def rotate(seq_trajectories, seq_destinations=None, ref_angle=None):
    seq_len, batch = seq_trajectories.shape[:2]
    flat_trajectories = seq_trajectories.reshape(-1, 2)

    angle = ref_angle
    if angle is None:
        angle = torch.rand(1)[0] * 2 * np.pi

    real_dests = None
    if seq_destinations is not None:
        destinations_shape = seq_destinations.shape
        flat_dest = seq_destinations.reshape(-1, 2)

    # angle = flip_type * np.pi / 8
    rotation_matrix = torch.Tensor([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]])

    flat_trajectories = torch.matmul(flat_trajectories, rotation_matrix)
    real_trajectories = flat_trajectories.reshape(seq_trajectories.shape)

    if seq_destinations is not None:
        flat_dest = torch.matmul(flat_dest, rotation_matrix)
        real_dests = flat_dest.reshape(seq_destinations.shape)

    return real_trajectories, real_dests, angle


def shift(seq_trajectories, seq_destinations=None, ref_offset_x=None, ref_offset_y=None):
    min_offset = -2
    max_offset = 2

    offset_x, offset_y = ref_offset_x, ref_offset_y

    if offset_x is None:
        offset_x, offset_y = torch.rand(
            2) + (max_offset - min_offset) + min_offset

    flat_trajectories = seq_trajectories.reshape(-1, 2)
    flat_trajectories[:, 0] = flat_trajectories[:, 0] + offset_x
    flat_trajectories[:, 1] = flat_trajectories[:, 1] + offset_y
    rs_trajectories = flat_trajectories.reshape(seq_trajectories.shape)

    rs_destinations = None
    if seq_destinations is not None:
        flat_destinations = seq_destinations.reshape(-1, 2)
        flat_destinations[:, 0] = flat_destinations[:, 0] + offset_x
        flat_destinations[:, 1] = flat_destinations[:, 1] + offset_y
        rs_destinations = flat_destinations.reshape(seq_destinations.shape)

    return rs_trajectories, rs_destinations, offset_x, offset_y


def scale(trajectories, ref_alpha=None):
    alpha = ref_alpha
    if alpha is None:
        alpha = torch.rand(1)[0]
        alpha = alpha * (1.25 - 0.75) + 0.75

    return alpha * trajectories, alpha


def change_perspective(obs, future=None, destinations=None, array_m=None, get_m=False, rotate=False):
    # destinations (num_dest, 8)
    # obs (seq_len, batch_size, 2)

    batch_size = obs.size(1)

    # (num_dest, 4, 2)
    point_destinations = None
    if destinations is not None:
        num_dest = destinations.size(0)
        point_destinations = destinations.reshape(num_dest, -1, 2)

    #(num_dest, batch_size, 4, 2)
    new_dests = []
    new_obs = []
    new_future = []

    inverse = array_m is not None
    if not inverse:
        array_m = []

    for seq_id in range(batch_size):
        obs_np = obs[:, seq_id, :].cpu().numpy()

        if future is not None:
            future_np = future[:, seq_id, :].cpu().numpy()

        if not inverse:
            B = obs_np[0, :]
            A = obs_np[-1, :]

            vect = B - A + 1e-4
            C = np.array([A[0] - vect[1], A[1] + vect[0]])
            perpend = C - A + 1e-4

            vect = vect / np.linalg.norm(vect)
            perpend = perpend / np.linalg.norm(perpend)

            src = np.array([A, vect + A, perpend + A])
            if not rotate:
                dst = np.array([[0, 0], [-1, 0], [0, -1]])
            else:
                random = np.random.rand() * 2 * np.pi
                dst = np.array([[0, 0], 
                               [np.cos(random), np.sin(random)],
                               [np.cos(random + np.pi / 2), np.sin(random + np.pi / 2)]])

            m, _ = cv2.estimateAffinePartial2D(
                np.float32(src), np.float32(dst))

            if get_m:
                array_m.append(m)

        else:
            m = np.float32(array_m[seq_id])
            m = cv2.invertAffineTransform(m)

        new_obs_np = cv2.transform(np.float32(obs_np[np.newaxis]), m)[0]

        if future is not None:
            new_future_np = cv2.transform(
                np.float32(future_np[np.newaxis]), m)[0]
            new_future.append(new_future_np)

        new_obs.append(new_obs_np)

        if destinations is not None:
            new_dest_np = point_destinations.cpu().numpy()
            new_dest_np = cv2.transform(np.float32(new_dest_np), m)
            new_dests.append(new_dest_np)

    new_obs = torch.Tensor(new_obs).permute(1, 0, 2).to(obs.device)

    if future is not None:
        new_future = torch.Tensor(new_future).permute(1, 0, 2).to(obs.device)

    if destinations is not None:
        new_dests = torch.Tensor(new_dests).to(destinations.device).permute(
            1, 0, 2, 3).reshape(num_dest, batch_size, -1)

    if not get_m:
        return new_obs, new_future, new_dests
    else:
        return new_obs, new_future, new_dests, array_m


def add_dummy_goal(last_obs, offset, dest):
                # (batch_size, 2)
    goal_point1 = [last_obs[:, 0] -
                   offset, last_obs[:, 1] - offset]
    goal_point2 = [last_obs[:, 0] +
                   offset, last_obs[:, 1] - offset]
    goal_point3 = [last_obs[:, 0] +
                   offset, last_obs[:, 1] + offset]
    goal_point4 = [last_obs[:, 0] -
                   offset, last_obs[:, 1] + offset]

    additional_goal = [p.unsqueeze(1) for g in [
        goal_point1, goal_point2, goal_point3, goal_point4] for p in g]

    # (1, batch_size, 8)
    additional_goal = torch.cat(
        additional_goal, dim=1).unsqueeze(0)
    # (num_dest + 1, batch_size, 8)
    dest = torch.cat([dest, additional_goal], dim=0)

    return dest
