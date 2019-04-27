import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_


def get_config():
    mean = 0
    stddev = 0.01
    return mean, stddev


mean, stddev = get_config()


def Normal(tensor, mean=mean, stddev=stddev):
    "Re initialize the tensor with normal weights and custom mean and stddev"
    normal_(tensor, mean=mean, std=stddev)
    return None


class vBPR(nn.Module):
    """
    Creates a vBPR module, which learns the latent factors over
    the user and item interactions.

    For more details refer to the paper: https://arxiv.org/pdf/1510.01784.pdf
    """

    def __init__(self,
                 num_latent_factors,
                 num_visual_factors,
                 num_embedding_factors,
                 num_users,
                 num_items,
                 visual_features,
                 dropout=0.1):
        "Creates the weights matrices for storing factors"
        super(vBPR, self).__init__()
        self.K = num_latent_factors
        self.D = num_visual_factors
        self.F = num_embedding_factors

        self.n_u = num_users
        self.n_i = num_items

        mean, stddev = get_config()
        # declare latent factor matrices for users and items
        self.U_latent_factors = nn.Parameter(torch.randn(self.n_u, self.K))
        self.I_latent_factors = nn.Parameter(torch.randn(self.n_i, self.K))
        Normal(self.U_latent_factors)
        Normal(self.I_latent_factors)

        # declare visual factor matrices for users
        self.U_visual_factors = nn.Parameter(torch.randn(self.n_u, self.D))
        Normal(self.U_visual_factors)

        # embedding linear layer for projecting embedding to visual factors
        self.embedding_projection = nn.Linear(self.F, self.D)
        Normal(self.embedding_projection.weight)
        Normal(self.embedding_projection.bias)

        self.dropout = nn.Dropout(dropout)

        # visual bias
        self.beta_dash = nn.Parameter(torch.randn(1, self.F))
        Normal(self.beta_dash)

        # user bias and item bias
        self.user_bias = nn.Parameter(torch.zeros(self.n_u))
        self.item_bias = nn.Parameter(torch.zeros(self.n_i))
        Normal(self.user_bias)
        Normal(self.item_bias)

        self.visual_features = visual_features

        # TODO: include regularization

    def get_xui(self, u_s, i_s):
        "Get x_ui value for a bunch of user indices and item indices"
        I_visual_factors = self.dropout(
            self.embedding_projection(
                self.visual_features[i_s]))

        return self.user_bias[u_s] + self.item_bias[i_s] + \
            torch.bmm(
                self.U_latent_factors[u_s].unsqueeze(1),
                self.I_latent_factors[i_s].unsqueeze(2)
        ).squeeze() + \
            torch.bmm(
                self.U_visual_factors[u_s].unsqueeze(1),
                I_visual_factors.unsqueeze(2)
        ).squeeze() + \
            self.beta_dash.mm(
                self.visual_features[i_s].transpose(0, 1)
        ).squeeze()

    def forward(self, trg_batch):
        """Calculate the preferences of user, i, j pairs.

            Args:
                trg_batch: [batch, 3]
            Returns:
                A Tensor of shape [batch, 1]
        """
        user_indices = trg_batch[:, 0]
        i_indices = trg_batch[:, 1]
        j_indices = trg_batch[:, 2]
        return self.get_xui(user_indices, i_indices) - \
            self.get_xui(user_indices, j_indices)


def BPRLoss(batch_xuij):
    "Return the loss for a batch of xuij predictions"
    return torch.log(torch.sigmoid(xuij)).sum()


def data_gen(train_data=data, batch_size=16):
    """
    Yields batches of training data with given batch size

    Args:
        train_data : Interaction, which contains user, i, count tuples

    Yields :
        batch of (u, i, j) tuples
    """
    interactions_dict = train_data.get_interaction_dict()
    num_items = train_data.num_items
    num_users = train_data.num_users

    for user in interactions_dict:
        for item in interactions_dict[user]:
            X = torch.LongTensor(batch_size, 3)
            X[:, 0] = user
            X[:, 1] = item
            js = [np.random.randint(num_items) for _ in range(2 * batch_size)]
            row_index = 0
            for j in js:
                if j not in interactions_dict[user]:
                    X[row_index, 2] = j
                    row_index += 1

            # just repeat the first row remaining times
            if row_index < batch_size:
                X[row_index, 2] = X[0, 2]

            yield X
