from collections import OrderedDict

import torch
import torch.nn as nn

import src.models.model_utils as model_utils

class ScoreNetwork(nn.Module):
    def __init__(self, dest_dim=4, dest_embedding_dim=32,
                 concat_dim=64, zg_dim=16, dropout=0.0):
        super(ScoreNetwork, self).__init__()

        self.dest_dim = dest_dim
        self.dest_dim = dest_dim
        self.dest_embedding_dim = dest_embedding_dim

        self.dest_embedding_net = model_utils.make_mlp([dest_dim, dest_embedding_dim],
                                           batch_norm=False,
                                           activation=None)

        self.concat_embedding_net = model_utils.make_mlp([zg_dim + dest_embedding_dim,
                                              concat_dim],
                                             batch_norm=False,
                                             dropout=dropout,
                                             activation="tanh")

        self.pre_act_embedding = model_utils.make_mlp([concat_dim, 1],
                                          batch_norm=False,
                                          activation=None)

        self.score_act = nn.Sequential(OrderedDict([
            ('act1', nn.Softmax(dim=1)),
        ]))

    def forward(self, zg, candidate_dest, get_intermediate=False, debug_id=None):
        """
            Args:
                candidate_dest: destination of size(n_dest, batch_size, dest_dim)
        """
        batch_size = zg.size(0)

        # Embed dest
        num_dest = candidate_dest.size(0)
        dest_dim = candidate_dest.size(-1)

        # [a,a,a,a,..,b,b,b,b,b,...,w,...]
        dest_embedded_dup = candidate_dest.reshape(-1, dest_dim)
        dest_embedded_dup = self.dest_embedding_net(dest_embedded_dup)

        # Compute the score between each zg and possible dest (ranking)
        # Flatten the matrices

        # [a,b,c,d,..,w] --> [a,b,c,d,..,w,a,b,c,d,...,w,...]
        zg_dup = zg.repeat(num_dest, 1)

        cat_embedded = self.concat_embedding_net(torch.cat([zg_dup,
                                                            dest_embedded_dup], dim=1))

        # if debug:
        #     print("Concated")
        #     print(cat_embedded)

        pre_score = self.pre_act_embedding(cat_embedded)

        # From flatten --> matrix
        # from [a,b,c,..,a,b,c] to [[a,..,a], [b,..,b],...]
        scores = torch.chunk(pre_score, num_dest)
        scores = torch.cat(scores, dim=1)

        # if debug:
        #     print("Pre softmax Score")
        #     print(scores)

        scores = self.score_act(scores)

        intermediate = None
        if get_intermediate:
            intermediate = cat_embedded

        return zg, scores, intermediate


class GoalEncoder(nn.Module):
    def __init__(self,
                 x_embedding_dim=16,
                 h_dim=64, layer_count=1):
        super(GoalEncoder, self).__init__()

        # Setting
        self.x_embedding_dim = x_embedding_dim
        self.h_dim = h_dim
        self.layer_count = layer_count
        # self.zg_gru_net = nn.GRU(2, h_dim, layer_count)

        # GRU net
        self.zg_gru_net = nn.GRU(x_embedding_dim, h_dim, layer_count)

        # Embedding nets
        self.x_embedding_net = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(2, x_embedding_dim)),
        ]))

    def init_hidden(self, batch_size, device):
        return torch.zeros(self.layer_count, batch_size, self.h_dim, device=device)

    def forward(self, x_rel):
        """

        Args:
            x_rel:
            dest: features of destinations

        Returns:
            informative_latent: Latent variable that capture the destination information
            scores: the probability of each trajectory belong to each destination
        """
        batch_size = x_rel.size(1)
        h = self.init_hidden(batch_size, x_rel.device)

        # Embed x
        x_embedded = x_rel.reshape(-1, 2)
        x_embedded = self.x_embedding_net(x_embedded)
        x_embedded = x_embedded.reshape(-1, batch_size, self.x_embedding_dim)
        # x_embedded = x_rel.clone()

        # Encode zg
        output, h = self.zg_gru_net(x_embedded, h)
        # return zg

        return h[-1]


class DynamicEncoder(nn.Module):
    def __init__(self,
                 x_embedding_dim=16,
                 h_dim=64, layer_count=1):
        super(DynamicEncoder, self).__init__()

        # Setting
        self.h_dim = h_dim
        self.layer_count = layer_count
        self.x_embedding_dim = x_embedding_dim

        # GRU net
        self.zo_encoder_net = nn.GRU(x_embedding_dim, h_dim, layer_count)

        # Embedding nets
        self.x_embedding_net = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(2, x_embedding_dim)),
        ]))

    def init_hidden(self, batch_size, device):
        return torch.zeros(self.layer_count, batch_size, self.h_dim, device=device)

    def encode(self, x_rel):
        """ Learn h from observed trajectories

        Args:
            x_rel: (seq_len, batch_size, 2)

        Returns:
            zo: uninformative latent at every time step
                (seq_len, batch_size, zo_dim)
            h: hidden variable at the last time-step
        """
        batch_size = x_rel.size(1)
        h = self.init_hidden(batch_size, x_rel.device)

        # Embed x
        x_embedded = x_rel.reshape(-1, 2)
        x_embedded = self.x_embedding_net(x_embedded)
        x_embedded = x_embedded.reshape(-1, batch_size, self.x_embedding_dim)

        # Encode zo
        output, h = self.zo_encoder_net(x_embedded, h)
        return h

    def decode(self, y_rel, h):
        """ Learn zo at each predict time step

        Args:
            y_rel: (bach_size, 2)
            h:

        Returns:
            zo:
            h:

        """
        y_embedded = self.x_embedding_net(y_rel).unsqueeze(0)
        output, h = self.zo_decoder_net(y_embedded, h)

        return h


class Decoder(nn.Module):
    def __init__(self,
                 x_embedding_dim=16,
                 pred_len=8,
                 layer_count=1,
                 zo_dim=64, zg_dim=64,
                 attend_goal=False, goal_intermediate_dim=64,
                 dropout=0.0):

        super(Decoder, self).__init__()

        self.max_pred_len = pred_len
        self.attend_goal = attend_goal

        raw_input_dim = 2

        if self.attend_goal:
            if dropout == 0:
                self.compability_net = nn.Sequential(OrderedDict([
                    ('fc1', nn.Linear(goal_intermediate_dim + zo_dim, zo_dim)),
                    ('act1', nn.Tanh()),
                    ('fc2', nn.Linear(zo_dim, 1))
                ]))
            else:
                self.compability_net = nn.Sequential(OrderedDict([
                    ('fc1', nn.Linear(goal_intermediate_dim + zo_dim, zo_dim)),
                    ('act1', nn.Tanh()),
                    ('dropout1', nn.Dropout(p=dropout)),
                    ('fc2', nn.Linear(zo_dim, 1))
                ]))

            self.alpha_net = nn.Softmax(dim=1)
            raw_input_dim = 2 + goal_intermediate_dim

        self.decoder_net = nn.GRU(x_embedding_dim, zo_dim, layer_count)
        self.x_embedding_net = nn.Linear(raw_input_dim, x_embedding_dim)
        self.output_net = nn.Linear(zo_dim, 2)

    def decode_one_timestep(self, y_embedded, z):
        output, h = self.decoder_net(y_embedded.unsqueeze(0), z)

        return h

    def forward(self, pred_len, last_x_rel, zo, zg=None, intermediate=None, debug_id=None, get_attention=False):
        y_hat_rel_seq = []

        y_hat_rel = last_x_rel

        batch_size = last_x_rel.size(0)
        y_hat_rel_seq = torch.zeros(
            self.max_pred_len, batch_size, 2).to(last_x_rel.device)

        h = zo

        goal_signal = None
        if self.attend_goal:
            num_dest = len(intermediate) // len(zg)
            num_dim = intermediate.shape[-1]

            goal_signal = torch.chunk(intermediate, num_dest)
            goal_signal = torch.cat(goal_signal, dim=1)
            goal_signal = goal_signal.reshape(len(zg), -1, num_dim)

        all_alpha = []
        for seq_id in range(pred_len):
            if not self.attend_goal:
                y_embedded = self.x_embedding_net(y_hat_rel)
                h = self.decode_one_timestep(y_embedded, h)
                z = h[-1]
            else:
                num_dest = len(intermediate) // len(zg)
                # dup_zo = h[-1].repeat(num_dest, 1)
                dup_zo = h[-1].repeat(num_dest, 1)

                compab = self.compability_net(
                    torch.cat([intermediate, dup_zo], dim=1))

                # From [[a, b, c], [a, b, c]] to [[a, a,], [b, b], [c, c]]
                pre_alpha = torch.chunk(compab, num_dest)
                pre_alpha = torch.cat(pre_alpha, dim=1)

                # (batch, n_goal)
                alpha = self.alpha_net(pre_alpha)

                if get_attention:
                    all_alpha.append(alpha)

                num_dim = intermediate.shape[-1]

                if (debug_id is not None):
                    if seq_id == 0:
                        print("Inside decoder")
                        print("num dest", num_dest)
                        print("num dim", num_dim)
                        print("intermediate shape", intermediate.shape)
                    print('t: {}, alpha: {}'.format(
                        seq_id, alpha[debug_id].data))

                # (batch, n_goal, goal_dim)
                alpha = alpha.unsqueeze(-1).repeat(1, 1, num_dim)

                # weighted goal signal
                weighted_goal_signal = alpha * goal_signal

                # (batch, goal_dim)
                weighted_goal_signal = torch.sum(weighted_goal_signal, dim=1)

                # cat features
                cat_input = torch.cat([weighted_goal_signal, y_hat_rel], dim=1)
                cat_input = self.x_embedding_net(cat_input)

                h = self.decode_one_timestep(cat_input, h)
                z = h[-1]

            y_hat_rel = self.output_net(z)
            y_hat_rel_seq[seq_id, :, :] = y_hat_rel

        if not get_attention:
            return y_hat_rel_seq
        else:
            return y_hat_rel_seq, all_alpha


class Model(nn.Module):
    def __init__(self, obs_len=8, pred_len=8, x_embedding_dim=16,
                 dest_dim=4, dest_embedding_dim=32,
                 zg_dim=16, zg_layer_count=1,
                 zo_dim=64, zo_layer_count=1,
                 decoder_layer_count=1, attend_goal=False,
                 score_concat_dim=64, dropout=0.0):
        super(Model, self).__init__()

        # Config
        self.obs_len = obs_len
        self.max_pred_len = pred_len
        self.attend_goal = attend_goal

        # nets
        self.zo_net = DynamicEncoder(x_embedding_dim,
                                     h_dim=zo_dim, layer_count=zo_layer_count)

        self.zg_net = GoalEncoder(x_embedding_dim,
                                  h_dim=zg_dim, layer_count=zg_layer_count)

        self.score_net = ScoreNetwork(dest_dim, dest_embedding_dim,
                                      score_concat_dim, zg_dim, dropout)

        self.decoder_net = Decoder(x_embedding_dim,
                                   pred_len, decoder_layer_count,
                                   zo_dim, zg_dim,
                                   attend_goal, score_concat_dim,
                                   dropout)

    def encode(self, x_rel, dest, debug_id=None):
        zo = self.zo_net.encode(x_rel)
        zg = self.zg_net(x_rel)

        scores = None
        intermediate = None

        # Prepare destination, duplicate to match the batch size
        if len(dest.shape) == 2:
            raw_dest = model_utils.torch_dest_repeat(dest, x_rel.size(1))
        else:
            raw_dest = dest
            
        processed_dest = model_utils.compute_2point_goal(raw_dest)
        goal_dest = model_utils.compute_angle_dest(processed_dest)
        processed_dest = torch.cat([processed_dest, goal_dest], dim=-1)

        zg, scores, intermediate = self.score_net(zg, processed_dest,
                                                  get_intermediate=self.attend_goal,
                                                  debug_id=debug_id)

        return zg, zo, scores, intermediate

    def forward(self, x_rel,
                loss_mask, dest,
                predict_trajectory=True,
                debug_id=None,
                get_attention=False):

        zg = None
        scores = None
        intermediate = None

        # Encode
        zg, ho, scores, intermediate = self.encode(x_rel,
                                                   dest,
                                                   debug_id=debug_id)

        # Decode
        y_hat_rel = None

        actual_pred_len = torch.max(torch.sum(loss_mask, dim=1)).long()
        # print(actual_pred_len, torch.min(torch.sum(loss_mask, dim=1)).long())
        if predict_trajectory:
            y_hat_rel = self.decoder_net(actual_pred_len,
                                         x_rel[-1], ho,
                                         zg,
                                         intermediate,
                                         debug_id=debug_id)
        elif get_attention:
            _, alpha = self.decoder_net(actual_pred_len,
                                        x_rel[-1], ho,
                                        zg,
                                        intermediate,
                                        debug_id=debug_id,
                                        get_attention=True)

            return scores, alpha

        return y_hat_rel, scores
