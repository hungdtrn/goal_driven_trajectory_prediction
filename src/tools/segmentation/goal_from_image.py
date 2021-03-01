import os
import re
from collections import OrderedDict
import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
from scipy.io import loadmat

from .lib.nn import user_scattered_collate, async_copy_to
from .dataset import TestDataset
from .config import cfg
from .models import ModelBuilder, SegmentationModule
from src.shared import PROJECT_PATH, TOOLPATH


def init_segmentation_tool():
    cfg_with_models = {
        "resnet50dilated": os.path.join(TOOLPATH, "config",
                                        "ade20k-resnet50dilated-ppm_deepsup.yaml")
    }

    MODEL_PATH = os.path.join(TOOLPATH, "pretrained",
                              "ade20k-resnet50dilated-ppm_deepsup")

    opts = ["DIR", MODEL_PATH, "TEST.checkpoint", "epoch_20.pth"]
    cfg.merge_from_file(cfg_with_models["resnet50dilated"])
    cfg.merge_from_list(opts)

    cfg.MODEL.arch_encoder = cfg.MODEL.arch_encoder.lower()
    cfg.MODEL.arch_decoder = cfg.MODEL.arch_decoder.lower()
    cfg.MODEL.weights_encoder = os.path.join(
        cfg.DIR, 'encoder_' + cfg.TEST.checkpoint)
    cfg.MODEL.weights_decoder = os.path.join(
        cfg.DIR, 'decoder_' + cfg.TEST.checkpoint)

    assert os.path.exists(cfg.MODEL.weights_encoder) and \
        os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"\

    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder,
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder,
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder,
        use_softmax=True)

    crit = nn.NLLLoss(ignore_index=-1)
    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
    segmentation_module.cuda()
    segmentation_module.eval()

    gpu = 0

    def run(image_path_dict):
        seg_maps = {}
        seg_features = {}

        reverse_dict = {}

        data_path = []
        for dataname, image_path in image_path_dict.items():
            data_path.append({'fpath_img': image_path})

            reverse_dict[os.path.basename(image_path)] = dataname

        dataset_test = TestDataset(
            data_path,
            cfg.DATASET
        )

        loader_test = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=cfg.TEST.batch_size,
            shuffle=False,
            collate_fn=user_scattered_collate,
            num_workers=5,
            drop_last=True)

        for batch_data in loader_test:
            batch_data = batch_data[0]
            segSize = (batch_data['img_ori'].shape[0],
                       batch_data['img_ori'].shape[1])
            img_resized_list = batch_data['img_data']
            name = os.path.basename(batch_data["info"])

            with torch.no_grad():
                scores = torch.zeros(
                    1, cfg.DATASET.num_class, segSize[0], segSize[1])

                features = torch.zeros(1, 512, segSize[0], segSize[1])

                for img in img_resized_list:
                    feed_dict = batch_data.copy()
                    feed_dict['img_data'] = img
                    del feed_dict['img_ori']
                    del feed_dict['info']
                    feed_dict = async_copy_to(feed_dict, gpu)

                    pred_tmp, pred_features = segmentation_module(
                        feed_dict, segSize=segSize)

                    pred_tmp_cpu = pred_tmp.cpu()
                    pred_features_cpu = pred_features.cpu()

                    scores = scores + pred_tmp_cpu / len(cfg.DATASET.imgSizes)
                    features = features + pred_features_cpu / \
                        len(cfg.DATASET.imgSizes)

                    del pred_tmp
                    del pred_features
                    torch.cuda.empty_cache()

                features = features.squeeze(0).permute(1, 2, 0)
                scores = scores.squeeze(0).permute(1, 2, 0)

                seg_maps[reverse_dict[name]] = scores.cpu().numpy()
                seg_features[reverse_dict[name]] = features.cpu().numpy()

        return seg_maps, seg_features

    return {
        "run": run
    }

# ----------------- Extract goals from featers ----------------------


def get_box(seg, n, iw, ih, box_w, box_h, frame_w, frame_h):
    w_start = iw * box_w
    w_end = (iw + 1) * box_w

    h_start = ih * box_h
    h_end = (ih + 1) * box_h

    if ih == n:
        h_end = frame_h

    if iw == n:
        w_end = frame_w

    return seg[h_start:h_end, w_start:w_end]


def get_features_in_box(box, feature_type="label"):
    if feature_type == "label":
        uniq_label = np.unique(box)

        max_count = None
        max_label = None
        for l in uniq_label:
            count = np.count_nonzero(box == l)
            if max_count is None or count > max_count:
                max_count = count
                max_label = l

        return int(max_label)
    else:
        mean_box = box.mean(axis=(0, 1))

        return mean_box


def determine_same(feature1, feature2, feature_type="label"):
    if feature_type == "label":
        return feature1 == feature2
    else:
        threshold = 4
        return np.sqrt(np.sum((feature1 - feature2)**2)) < threshold


def is_mergable(src_xmin, src_ymin, src_xmax, src_ymax, dst_xmin, dst_ymin, dst_xmax, dst_ymax, n, boundary):
    src_height = src_ymax - src_ymin
    src_width = src_xmax - src_xmin

    dst_height = dst_ymax - dst_ymin
    dst_width = dst_xmax - dst_xmin

    src_mergable = ((src_xmax <= boundary and src_height > boundary and dst_xmax > boundary) or
                    (src_xmin >= n - boundary and src_height > boundary and dst_xmin < n - boundary) or
                    (src_ymax <= boundary and src_width > boundary and dst_ymax > boundary) or
                    (src_ymin >= n - boundary and src_width > boundary and dst_ymin < n - boundary))

    dst_mergable = ((dst_xmax <= boundary and dst_height > boundary and src_xmax > boundary) or
                    (dst_xmin >= n - boundary and dst_height > boundary and src_xmin < n - boundary) or
                    (dst_ymax <= boundary and dst_width > boundary and src_ymax > boundary) or
                    (dst_ymin >= n - boundary and dst_width > boundary and src_ymin < n - boundary))

    return ((not src_mergable) and (not dst_mergable))


def update_labels(seg, labels, labels_info, n, boundary, iw, ih, box_w, box_h, frame_w, frame_h, feature_type):
    current_box = get_box(seg, n, iw, ih, box_w, box_h, frame_w, frame_h)

    max_label = np.max(labels)

    def merge(near_iw, near_ih):
        near_box = get_box(seg, n, near_iw, near_ih,
                           box_w, box_h, frame_w, frame_h)
        near_label = labels[near_ih, near_iw]
        is_same = determine_same(get_features_in_box(current_box, feature_type),
                                 get_features_in_box(near_box, feature_type),
                                 feature_type)

        if is_same and near_label != -1:
            xmin, ymin = labels_info[near_label]["xmin"], labels_info[near_label]["ymin"]
            xmax, ymax = labels_info[near_label]["xmax"], labels_info[near_label]["ymax"]
            width = xmax - xmin
            height = ymax - ymin

            if is_mergable(xmin, ymin, xmax, ymax, iw, ih, iw + 1, ih + 1, n, boundary):

                xmin = np.min((xmin, iw))
                xmax = np.max((xmax, iw + 1))
                ymin = np.min((ymin, ih))
                ymax = np.max((ymax, ih + 1))

                labels_info[near_label] = {
                    "xmin": xmin,
                    "xmax": xmax,
                    "ymin": ymin,
                    "ymax": ymax
                }

                return True, near_label, labels_info

        return False, None, None

    # check left
    if iw > 0:
        near_iw, near_ih = iw - 1, ih

        merged, near_label, updated_info = merge(near_iw, near_ih)

        if merged:
            return near_label, updated_info

    # check right
    if iw < n - 1:
        near_iw, near_ih = iw + 1, ih

        merged, near_label, updated_info = merge(near_iw, near_ih)

        if merged:
            return near_label, updated_info

    # check top
    if ih > 0:
        near_iw, near_ih = iw, ih - 1

        merged, near_label, updated_info = merge(near_iw, near_ih)

        if merged:
            return near_label, updated_info

    current_label = max_label + 1
    labels_info[current_label] = {
        "xmin": iw,
        "xmax": iw + 1,
        "ymin": ih,
        "ymax": ih + 1
    }
    return current_label, labels_info


def find_possible_goal(goals, xmin, ymin, xmax, ymax, n, boundary, exclude):
    for goal_label, goal_dict in goals.items():
        if goal_label in exclude:
            continue

        src_xmin, src_ymin = goal_dict["xmin"], goal_dict["ymin"]
        src_xmax, src_ymax = goal_dict["xmax"], goal_dict["ymax"]

        if ((xmin >= src_xmin and xmin <= src_xmax and ymin >= src_ymin and ymin <= src_ymax) or
                (xmax >= src_xmin and xmax <= src_xmax and ymax >= src_ymin and ymax <= src_ymax)):

            if is_mergable(src_xmin, src_ymin, src_xmax, src_ymax, xmin, ymin, xmax, ymax, n, boundary):
                return goal_label

    return exclude[-1]


def group_goals(goal_info, n, boundary, min_area=1):
    exclude = []
    for goal_label, goal_dict in goal_info.items():
        xmin, ymin = goal_dict["xmin"], goal_dict["ymin"]
        xmax, ymax = goal_dict["xmax"], goal_dict["ymax"]
        width = xmax - xmin
        height = ymax - ymin

#         if ((((xmax <= boundary) or (xmin >= n - boundary)) and (width < boundary)) or
#             (((ymax <= boundary) or (ymin >= n - boundary)) and (height < boundary))):

        if width < boundary or height < boundary or width * height < min_area:
            exclude.append(goal_label)
            new_goal_label = find_possible_goal(
                goal_info, xmin, ymin, xmax, ymax, n, boundary, exclude)

            goal_info[new_goal_label]["xmin"] = np.min(
                (goal_info[new_goal_label]["xmin"], xmin))
            goal_info[new_goal_label]["xmax"] = np.max(
                (goal_info[new_goal_label]["xmax"], xmax))
            goal_info[new_goal_label]["ymin"] = np.min(
                (goal_info[new_goal_label]["ymin"], ymin))
            goal_info[new_goal_label]["ymax"] = np.max(
                (goal_info[new_goal_label]["ymax"], ymax))

    for excluded in exclude:
        del goal_info[excluded]
    return goal_info


def merge_goal(goals, n, boundary):
    num_goals = len(goals)
    labels = np.ones((n, n)) * 200

    for goal_label, goal_dict in goals.items():
        xmin, ymin = goal_dict["xmin"], goal_dict["ymin"]
        xmax, ymax = goal_dict["xmax"], goal_dict["ymax"]

        labels[ymin:ymax, xmin:xmax] = np.minimum(labels[ymin:ymax, xmin:xmax],
                                                  np.ones((ymax - ymin, xmax - xmin)) * goal_label)

    new_goals = {}
    for ih in range(n):
        for iw in range(n):
            if (ih >= boundary and ih < n - boundary) and (iw >= boundary and iw < n - boundary):
                continue

            current_label = labels[ih, iw]
            if current_label not in new_goals:
                new_goals[current_label] = {
                    "xmin": iw,
                    "ymin": ih,
                    "xmax": iw + 1,
                    "ymax": ih + 1
                }
            else:
                xmin, ymin = new_goals[current_label]["xmin"], new_goals[current_label]["ymin"]
                xmax, ymax = new_goals[current_label]["xmax"], new_goals[current_label]["ymax"]

                new_goals[current_label]["xmin"] = np.min((xmin, iw))
                new_goals[current_label]["xmax"] = np.max((xmax, iw + 1))
                new_goals[current_label]["ymin"] = np.min((ymin, ih))
                new_goals[current_label]["ymax"] = np.max((ymax, ih + 1))

    return new_goals


def split_goal(goals):
    max_goal = np.max(list(goals.keys()))
    thresh = 6

    new_goals = {}

    splited = False
    for goal_label, goal_dict in goals.items():
        xmin, ymin = goal_dict["xmin"], goal_dict["ymin"]
        xmax, ymax = goal_dict["xmax"], goal_dict["ymax"]

        width = xmax - xmin
        height = ymax - ymin

        if width > thresh or height > thresh:
            xmin1, xmin2 = xmin, xmin
            xmax1, xmax2 = xmax, xmax
            ymin1, ymin2 = ymin, ymin
            ymax1, ymax2 = ymax, ymax

            splited = True

            if width > thresh:
                width1 = width // 2
                xmin1, xmin2 = xmin, xmin + width1
                xmax1, xmax2 = xmin + width1, xmax

            if height > thresh:
                height1 = height // 2
                ymin1, ymin2 = ymin, ymin + height1
                ymax1, ymax2 = ymin + height1, ymax

            new_goals[goal_label] = {
                "xmin": xmin1,
                "xmax": xmax1,
                "ymin": ymin1,
                "ymax": ymax1
            }

            new_goals[max_goal + 1] = {
                "xmin": xmin2,
                "xmax": xmax2,
                "ymin": ymin2,
                "ymax": ymax2
            }

            max_goal = max_goal + 1
        else:
            new_goals[goal_label] = goal_dict

    return splited, new_goals


def filter_boxes(goals, seg_scores, n, box_w, box_h, frame_w, frame_h):
    base_walkable_labels = [x - 1 for x in [4, 7, 12, 10]]
    base_threshold = 0.04

    def run(threshold, walkable_labels):
        walkable_scores = seg_scores[:, :, walkable_labels]

        count = 0
        new_goals = {}

        splited = False
        for goal_label, goal_dict in goals.items():
            xmin, ymin = goal_dict["xmin"], goal_dict["ymin"]
            xmax, ymax = goal_dict["xmax"], goal_dict["ymax"]

            box_xmin = xmin * box_w
            box_xmax = xmax * box_w
            box_ymin = ymin * box_h
            box_ymax = ymax * box_h

            if xmax == n:
                box_xmax = frame_w

            if ymax == n:
                box_ymax = frame_h

            box_walkable = walkable_scores[box_ymin:box_ymax,
                                           box_xmin:box_xmax]
            mean_score = box_walkable.mean(axis=(0, 1))
            max_score = np.max(mean_score)

            tmp = np.mean(seg_scores[box_ymin:box_ymax, box_xmin:box_xmax], axis=(0, 1))
            tmp = np.argsort(tmp)[::-1][:5] + 1

            if max_score > threshold:
                new_goals[goal_label] = {
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax
                }
                count += 1

        return count, new_goals

    # Run to get filter the goals
    count, new_goals = run(base_threshold, base_walkable_labels)

    # If almost all of the goals are filtered, add more scene label (ceilling)
    if count < 6:
        base_walkable_labels.append(5)
        base_threshold = 0.3

        count, new_goals = run(base_threshold, base_walkable_labels)

    if count < 6:
        base_threshold = 0.01
        count, new_goals = run(base_threshold, base_walkable_labels)

    return new_goals


def detect_goal(seg_results, feature_type="label", bg_image=None):
    n = 16
    boundary = 2

    frame = None

    seg_scores, seg_features = seg_results

    is_visualize = False
    if bg_image is not None:
        is_visualize = True

    if feature_type == "label":
        seg = np.argmax(seg_scores, axis=2)
    else:
        seg = seg_features

    h, w = seg.shape[:2]
    box_h = h // n
    box_w = w // n

    labels = np.zeros((n, n)) - 1
    labels_info = {}

    # Get image if visualize is true
    if is_visualize:
        frame = bg_image.copy()

    # Divide the image into several boxes
    # Assign label for each box
    for ih in range(n):
        h_start = ih * box_h
        h_end = (ih + 1) * box_h

        if ih == n:
            h_end = h

        for iw in range(n):
            w_start = iw * box_w
            w_end = (iw + 1) * box_w

            if iw == n:
                w_end = w

            if (ih >= boundary and ih < n - boundary) and (iw >= boundary and iw < n - boundary):
                continue

            box_label, labels_info = update_labels(seg, labels, labels_info, n, boundary,
                                                   iw, ih, box_w, box_h, w, h, feature_type)
            labels[ih, iw] = box_label

            if is_visualize:
                cv2.rectangle(frame, (int(w_start), int(h_start)), (int(
                    w_end), int(h_end)), (255, 0, 0), thickness=1)

                str_value = "{}".format(box_label)

    goals = merge_goal(labels_info, n, boundary)
    splited, goals = split_goal(goals)

    goals = group_goals(goals, n, boundary)

    while splited:
        splited, goals = split_goal(goals)

    goals = filter_boxes(goals, seg_scores, n, box_w, box_h, w, h)
    goals = group_goals(goals, n, boundary, 7)
    for goal_label, goal_dict in goals.items():
        xmin, ymin = goal_dict["xmin"], goal_dict["ymin"]
        xmax, ymax = goal_dict["xmax"], goal_dict["ymax"]

        w_start = xmin * box_w
        w_end = xmax * box_w
        h_start = ymin * box_h
        h_end = ymax * box_h

        if xmax == n:
            w_end = w

        if ymax == n:
            h_end = h

        cv2.rectangle(frame, (int(w_start), int(h_start)), (int(
            w_end), int(h_end)), (0, 255, 0), thickness=2)

        cv2.putText(frame, str(goal_label),
                    (int((w_start + w_end) / 2),
                     int((h_start + h_end) / 2)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    color=(0, 255, 0),
                    fontScale=0.5,
                    thickness=2)

    result = []
    for _, goal_dict in goals.items():
        xmin, ymin = goal_dict["xmin"], goal_dict["ymin"]
        xmax, ymax = goal_dict["xmax"], goal_dict["ymax"]

        min_w = xmin * box_w
        max_w = xmax * box_w
        min_h = ymin * box_h
        max_h = ymax * box_h

        if xmax == n:
            max_w = w

        if ymax == n:
            max_h = h

        result.append([min_w, min_h, max_w, min_h, max_w, max_h, min_w, max_h])

    result = np.array(result)
    return result, frame
