import os

import torch

import src.utils as global_utils
from ..losses import displacement_error, final_displacement_error, bce_loss


def predict_trajectory_from_batch(args, batch, model):
    seq_gen = global_utils.seq_from_batch_generator(batch)
    y_rel_hat = []

    for seq_data in seq_gen:
        seq_y_rel_hat = predict_trajectory_from_sequence(args, seq_data, model)
        y_rel_hat.append(seq_y_rel_hat)

    y_rel_hat = torch.cat(y_rel_hat, dim=1)
    return y_rel_hat


def predict_goal_from_batch(args, batch, model):
    seq_gen = global_utils.seq_from_batch_generator(batch)
    scores = []
    labels = []

    for seq_data in seq_gen:
        (seq_x_origin, seq_y_origin,
         seq_x_rel, seq_y_rel,
         seq_loss_mask, seq_dataset,
         seq_dest, seq_label,
         start, end, seq_start_frame,
         seq_ped_idx) = seq_data

        seq_y_loss_mask = seq_loss_mask[:, args.obs_len:]
        seq_scores, seq_attention = model(seq_x_rel, seq_y_loss_mask,
                                          seq_dest, predict_trajectory=False, get_attention=True)

        scores.append(seq_scores)
        labels.append(seq_label)

    return scores, labels


def predict_trajectory_from_sequence(args, seq_data, model):
    (seq_x_origin, seq_y_origin,
     seq_x_rel, seq_y_rel,
     seq_loss_mask, seq_dataset,
     seq_dest, seq_label,
     start, end, seq_start_frame,
     seq_ped_idx) = seq_data

    seq_y_loss_mask = seq_loss_mask[:, args.obs_len:]
    seq_y_rel_hat, _ = model(seq_x_rel, seq_y_loss_mask,
                             seq_dest, predict_trajectory=True)

    return seq_y_rel_hat


def predict_trajectory_from_pedestrian(args, ped_data, model):
    (ped_x_origin, ped_x_rel, ped_loss_mask, ped_dest) = ped_data

    ped_y_loss_mask = ped_loss_mask[:, args.obs_len:]
    ped_y_rel_hat, _ = model(ped_x_rel, ped_y_loss_mask,
                             ped_dest, predict_trajectory=True)

    return ped_y_rel_hat


def predict_goal_from_pedestrian(args, ped_data, model):
    (ped_x_origin, ped_x_rel, ped_loss_mask, ped_dest) = ped_data

    ped_y_loss_mask = ped_loss_mask[:, args.obs_len:]
    ped_goal, ped_attention = model(ped_x_rel, ped_y_loss_mask,
                                    ped_dest, predict_trajectory=False, get_attention=True)

    return ped_goal, ped_attention


def get_ade_fde_batch(args, batch, model):
    model.eval()
    with torch.no_grad():
        batch = global_utils.to_cuda(batch)

        (obs_traj, gt_traj, obs_traj_rel, gt_traj_rel,
         loss_mask, seq_start_end) = global_utils.get_trajectories_from_batch(batch)

        batch_size = obs_traj.size(1)

        pred_loss_mask = loss_mask[:, args.obs_len:]

        pred_traj_rel = predict_trajectory_from_batch(
            args, batch, model)

        pred_traj = global_utils.relative_to_abs(
            pred_traj_rel, obs_traj[-1])

        ade = displacement_error(
            pred_traj, gt_traj, pred_loss_mask, mode='raw')
        fde = final_displacement_error(
            pred_traj, gt_traj, pred_loss_mask, mode='raw')

        return ade, fde


def top_N_goal_acc(args, data_loader, model, N):
    goal_acc = []
    total_traj = 0
    with torch.no_grad():
        for batch in data_loader:
            batch = global_utils.to_cuda(batch)

            (obs_traj, gt_traj, obs_traj_rel, gt_traj_rel,
             loss_mask, seq_start_end) = global_utils.get_trajectories_from_batch(batch)
            
            pred_scores, labels = predict_goal_from_batch(args, batch, model)
            top_n = [torch.argsort(tmp, dim=1)[:, -N:] for tmp in pred_scores]
            top_n = torch.cat(top_n, dim=0)

            labels = torch.cat(labels, dim=0)
            candiate_index = labels != -1
            labels = labels.unsqueeze(1).repeat(1, N)
            
            out = top_n[candiate_index] == labels[candiate_index]
            out = out.long()
            out = out.sum(dim=1)
            goal_acc.append(len(out[out > 0]))

            total_traj += len(out)
    
    return sum(goal_acc) * 1.0 / total_traj


def check_accuracy_predicted(args, batch_array, predicted_traj_rel, score_array, label_array, predict_trajectory=True):
    total_traj = 0
    metrics = {}
    disp_error = [0.0]
    f_disp_error = [0.0]
    goal_acc = []
    
    for batch, pred_traj_rel, pred_scores, labels in zip(batch_array, predicted_traj_rel, 
                                                     score_array, label_array):
        (obs_traj, gt_traj, obs_traj_rel, gt_traj_rel,
        loss_mask, seq_start_end) = global_utils.get_trajectories_from_batch(batch)

        pred_loss_mask = loss_mask[:, args.obs_len:]
        
        pred_traj = global_utils.relative_to_abs(pred_traj_rel, obs_traj[-1])
        
        # compute the sum of the average loss over each sequence
        ade = displacement_error(pred_traj, gt_traj, pred_loss_mask)

        # copmute the sum of the loss at the end of each sequence
        fde = final_displacement_error(
            pred_traj, gt_traj, pred_loss_mask)
        
        disp_error.append(ade.item())
        f_disp_error.append(fde.item())
        
        labels = torch.cat(labels, dim=0)
        pred_labels = torch.cat(
            [torch.argmax(sc, dim=1) for sc in pred_scores], dim=0)

        candiate_index = labels != -1

        goal_acc.append(
            torch.sum((pred_labels[candiate_index] == labels[candiate_index]).long()))

        total_traj += gt_traj.size(1)
        
    metrics["ade"] = sum(disp_error) / total_traj
    metrics["fde"] = sum(f_disp_error) / total_traj
    metrics["goal_acc"] = sum(goal_acc) / total_traj

    return metrics

def check_accuracy(args, data_loader, model, predict_trajectory=True, limit=False):
    disp_error = [0.0]
    f_disp_error = [0.0]
    goal_acc = [0.0]
    total_traj = 0
    metrics = {}

    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            batch = global_utils.to_cuda(batch)

            (obs_traj, gt_traj, obs_traj_rel, gt_traj_rel,
             loss_mask, seq_start_end) = global_utils.get_trajectories_from_batch(batch)

            pred_loss_mask = loss_mask[:, args.obs_len:]

            if predict_trajectory:
                pred_traj_rel = predict_trajectory_from_batch(
                    args, batch, model)

                pred_traj = global_utils.relative_to_abs(
                    pred_traj_rel, obs_traj[-1])

                # compute the sum of the average loss over each sequence
                ade = displacement_error(pred_traj, gt_traj, pred_loss_mask)

                # copmute the sum of the loss at the end of each sequence
                fde = final_displacement_error(
                    pred_traj, gt_traj, pred_loss_mask)

                disp_error.append(ade.item())
                f_disp_error.append(fde.item())

            pred_scores, labels = predict_goal_from_batch(args, batch, model)
            labels = torch.cat(labels, dim=0)
            pred_labels = torch.cat(
                [torch.argmax(sc, dim=1) for sc in pred_scores], dim=0)

            candiate_index = labels != -1

            goal_acc.append(
                torch.sum((pred_labels[candiate_index] == labels[candiate_index]).long()))

            total_traj += gt_traj.size(1)

            if limit and total_traj > args.num_samples_check:
                break

    metrics["ade"] = sum(disp_error) / total_traj
    metrics["fde"] = sum(f_disp_error) / total_traj
    metrics["goal_acc"] = sum(goal_acc) / total_traj

    model.train()

    return metrics
