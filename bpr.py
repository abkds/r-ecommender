import torch
import torch.nn as nn
import torch.nn.functional as F


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
                 dropout=0.1):
        "Creates the weights matrices for storing factors"
        super(vBPR, self).__init__():
        self.K = num_latent_factors
        self.D = num_visual_factors
        self.F = num_embedding_factors

        self.n_u = num_users
        self.n_i = num_items

        # declare latent factor matrices for users and items
        self.U_latent_factors = nn.Parameter(torch.randn(self.n_u, self.K))
        self.I_latent_factors = nn.Parameter(torch.randn(self.n_i, self.K))

        # declare visual factor matrices for users
        self.U_visual_factors = nn.Parameter(torch.randn(self.n_u, self.D))

        # embedding linear layer for projecting embedding to visual factors
        self.embedding_projection = nn.Linear(self.F, self.D)

        self.dropout = nn.Dropout(dropout)

        # visual bias 
        self.beta_dash = nn.Parameter(torch.randn(1, self.F))

        # user bias and item bias 
        self.user_bias = nn.Parameter(torch.zeros(1, self.n_u))
        self.item_bias = nn.Parameter(torch.zeros(1, self.n_i))

        # TODO: include regularization 

    def forward(self, visual_features, trg_batch):
        pass