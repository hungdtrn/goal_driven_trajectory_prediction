from attrdict import AttrDict

import torch
import cv2
import numpy as np

from .model import Model as GoalGRU
import src.utils as global_utils
from .local_utils import check_accuracy, check_accuracy_predicted, \
    predict_trajectory_from_batch, predict_goal_from_batch, \
    get_ade_fde_batch, predict_trajectory_from_pedestrian, \
    predict_trajectory_from_sequence, predict_goal_from_pedestrian, \
    top_N_goal_acc
    
from src.features.utils import change_perspective


def inverse_transform(m_array, predicted, data, is_batch=False, is_seq=False, is_ped=False):
    if is_batch:
        (x_origin, y_origin,
         x_rel, y_rel,
         loss_mask, seq_start_end,
         frame_idx, ped_idx,
         cluster_label, destinations,
         datasets) = data

        batch_inverse_x = []
        batch_inverse_y = []
        
        for i, (start, end) in enumerate(seq_start_end.data):
            pred_y = global_utils.relative_to_abs(
                predicted[:, start:end], x_origin[-1, start:end])
            inverse_x, inverse_y, _ = change_perspective(
                x_origin[:, start:end], pred_y, array_m=m_array[i])

            batch_inverse_x.append(inverse_x)
            batch_inverse_y.append(inverse_y)

        inverse_x = torch.cat(batch_inverse_x, dim=1)
        inverse_y = torch.cat(batch_inverse_y, dim=1)
    else:
        x_origin = data[0]
        pred_y = global_utils.relative_to_abs(predicted, x_origin[-1])
        inverse_x, inverse_y, _ = change_perspective(
            x_origin, pred_y, array_m=m_array[0])

    inverse_y_rel = torch.zeros_like(inverse_y)
    inverse_y_rel[1:] = inverse_y[1:] - inverse_y[:-1]
    inverse_y_rel[0] = inverse_y[0] - inverse_x[-1]

    return inverse_y_rel


def transform_perspective(data, is_batch=False, is_seq=False, is_ped=False):
    m_array = []

    if is_batch:
        (x_origin, y_origin,
         x_rel, y_rel,
         loss_mask, seq_start_end,
         frame_idx, ped_idx,
         cluster_label, destinations,
         datasets) = data

        new_x_origin = []
        new_y_origin = []
        new_destinations = []
        for i, (start, end) in enumerate(seq_start_end.data):
            seq_x_origin = x_origin[:, start:end].contiguous()
            seq_y_origin = y_origin[:, start:end].contiguous()
            seq_destinations = destinations[i]

            (new_seq_x_origin, new_seq_y_origin,
             new_seq_destinations, m) = change_perspective(seq_x_origin,
                                                           seq_y_origin,
                                                           seq_destinations, get_m=True)
            new_x_origin.append(new_seq_x_origin)
            new_y_origin.append(new_seq_y_origin)

            new_destinations.append(new_seq_destinations.contiguous())
            m_array.append(m)

        new_x_origin = torch.cat(new_x_origin, dim=1)
        new_y_origin = torch.cat(new_y_origin, dim=1)
        new_x_rel = torch.zeros_like(new_x_origin)
        new_y_rel = torch.zeros_like(new_y_origin)

        new_x_rel[1:] = new_x_origin[1:] - new_x_origin[:-1]
        new_y_rel[1:] = new_y_origin[1:] - new_y_origin[:-1]
        new_y_rel[0] = new_y_origin[0] - new_x_origin[-1]

        new_batch = (new_x_origin, new_y_origin,
                     new_x_rel, new_y_rel,
                     loss_mask, seq_start_end,
                     frame_idx, ped_idx,
                     cluster_label, new_destinations,
                     datasets)

        return global_utils.to_cuda(new_batch), m_array

    elif is_seq:
        (seq_x_origin, seq_y_origin,
         seq_x_rel, seq_y_rel,
         seq_loss_mask, seq_dataset,
         seq_dest, seq_label,
         start, end, seq_start_frame,
         seq_ped_idx) = data

        (new_seq_x_origin, new_seq_y_origin,
         new_seq_destinations, m) = change_perspective(seq_x_origin,
                                                       seq_y_origin,
                                                       seq_destinations, get_m=True)
        m_array.append(m)
        new_seq_x_origin = new_seq_x_origin.cuda()
        new_seq_y_origin = new_seq_y_origin.cuda()
        new_seq_destinations = new_seq_destinations.cuda()

        new_seq_x_rel = torch.zeros_like(new_seq_x_origin)
        new_seq_y_rel = torch.zeros_like(new_seq_y_origin)

        new_seq_x_rel[1:] = new_seq_x_origin[1:] - new_seq_x_origin[:-1]
        new_seq_y_rel[1:] = new_seq_y_origin[1:] - new_seq_y_origin[:-1]
        new_seq_y_rel[0] = new_seq_y_origin[0] - new_seq_x_origin[-1]

        return (new_seq_x_origin, new_seq_y_origin,
                new_seq_x_rel, new_seq_y_rel,
                seq_loss_mask, seq_dataset,
                new_seq_destinations, seq_label,
                start, end, seq_start_frame,
                seq_ped_idx), m_array

    else:
        (ped_x_origin, ped_x_rel, ped_loss_mask, ped_dest) = data
        new_ped_x_origin, _, new_ped_dest, m = change_perspective(ped_x_origin, None,
                                                                  ped_dest, get_m=True)

        m_array.append(m)
        new_ped_x_origin = new_ped_x_origin.cuda()
        new_ped_dest = new_ped_dest.cuda()

        new_ped_x_rel = torch.zeros_like(new_ped_x_origin)
        new_ped_x_rel[1:] = new_ped_x_origin[1:] - new_ped_x_origin[:-1]

        return (new_ped_x_origin, new_ped_x_rel, ped_loss_mask, new_ped_dest), m_array


def create_runner(checkpoint_path, data_args, load_best=False, cuda=True):
    checkpoint = torch.load(checkpoint_path)

    is_change_perspective = (checkpoint["args"]["perspective"]) and (
        not data_args["perspective"])

    args = AttrDict(dict(checkpoint["args"], **data_args))
    args["perspective"] = checkpoint["args"]["perspective"]

    model = GoalGRU(obs_len=args.obs_len, pred_len=args.pred_len,
                    x_embedding_dim=args.x_embedding_dim, dest_dim=args.dest_dim,
                    dest_embedding_dim=args.dest_embedding_dim,
                    zg_dim=args.zg_dim, zo_dim=args.zo_dim,
                    attend_goal=args.attend_goal,
                    score_concat_dim=args.score_concat_dim,
                    dropout=args.dropout)

    loaded_epoch = None
    if not load_best:
        model.load_state_dict(checkpoint["model_state"])
        loaded_epoch = checkpoint["current_epoch"]
    else:
        model.load_state_dict(checkpoint["best_model_state"])
        loaded_epoch = checkpoint["best_epoch"]
    
    if cuda:
        model = model.cuda()

    model.eval()
    
    def batch_predict_trajectory(batch):
        if is_change_perspective:
            batch, m_array = transform_perspective(batch, is_batch=True)
            predicted = predict_trajectory_from_batch(args, batch, model)
            return inverse_transform(m_array, predicted.detach(), batch, is_batch=True)
        else:
            return predict_trajectory_from_batch(args, batch, model)

    def sequence_predict_trajectory(seq_data):
        if is_change_perspective:
            seq_data, m_array = transform_perspective(seq_data, is_seq=True)
            predicted = predict_trajectory_from_sequence(args, seq_data, model)
            return inverse_transform(m_array, predicted.detach(), seq_data, is_seq=True)
        else:
            return predict_trajectory_from_sequence(args, seq_data, model)

    def pedestrian_predict_trajectory(ped_data):
        if is_change_perspective:
            ped_data, m_array = transform_perspective(ped_data, is_ped=True)
            predicted = predict_trajectory_from_pedestrian(
                args, ped_data, model)
            return inverse_transform(m_array, predicted.detach(), ped_data, is_ped=True)
        else:
            return predict_trajectory_from_pedestrian(args, ped_data, model)

    def pedestrian_goal_predict(ped_data):
        if is_change_perspective:
            ped_data, _ = transform_perspective(ped_data, is_ped=True)

        return predict_goal_from_pedestrian(args, ped_data, model)

    def batch_get_ade_fde(batch):
        if is_change_perspective:
            batch, _ = transform_perspective(batch, is_batch=True)

        return get_ade_fde_batch(args, batch, model)

    # def loader_get_accuracy(data_loader):
    #     if is_change_perspective:
    #         transformed_loader = []
    #         for batch in data_loader:
    #             new_batch, _ = transform_perspective(batch, is_batch=True)
    #             transformed_loader.append(new_batch)
    #         return check_accuracy(args, transformed_loader, model)
    #     else:
    #         return check_accuracy(args, data_loader, model)

    def loader_get_accuracy(data_loader):
        if is_change_perspective:
            batch_array = []
            predicted_traj_array = []
            score_array = []
            label_array = []
            
            for batch in data_loader:
                new_batch, m_array = transform_perspective(batch, is_batch=True)
                
                new_batch = global_utils.to_cuda(new_batch)
                (obs_traj, gt_traj, obs_traj_rel, gt_traj_rel,
                loss_mask, seq_start_end) = global_utils.get_trajectories_from_batch(new_batch)
                
                new_pred_traj_rel = predict_trajectory_from_batch(args, new_batch, model)
                                
                pred_traj_rel = inverse_transform(m_array, new_pred_traj_rel.detach(), 
                                              new_batch, is_batch=True)
                
                pred_scores, labels = predict_goal_from_batch(args, new_batch, model)
                
                batch_array.append(global_utils.to_cuda(batch))
                predicted_traj_array.append(pred_traj_rel.cuda())
                score_array.append(pred_scores)
                label_array.append(labels)
                
            return check_accuracy_predicted(args, batch_array, predicted_traj_array, 
                                            score_array, label_array)
        
        else:
            return check_accuracy(args, data_loader, model)
                



    def loader_get_goal_accuracy(data_loader):
        if is_change_perspective:
            transformed_loader = []
            top1 = top_N_goal_acc(args, data_loader, model, 1)
            top3 = top_N_goal_acc(args, data_loader, model, 3)
        else:
            top1 = top_N_goal_acc(args, data_loader, model, 1)
            top3 = top_N_goal_acc(args, data_loader, model, 3)

        return {
            "top1": top1,
            "top3": top3
        }

    return {
        "model": model,
        "args": args,
        "loader_get_accuracy": loader_get_accuracy,
        "loader_get_goal_accuracy": loader_get_goal_accuracy,
        "batch_predict_trajectory": batch_predict_trajectory,
        "batch_get_ade_fde": batch_get_ade_fde,
        "seq_predict_trajectory": sequence_predict_trajectory,
        "ped_predict_trajectory": pedestrian_predict_trajectory,
        "ped_predict_goal": pedestrian_goal_predict
    }
