# Neural Collaborative Filter described in 'Neural Collaborative Filtering' by Xiangnan et al.
# Adapted from https://github.com/yihong-chen/neural-collaborative-filtering

import torch
import torch.nn as nn

class NCF(nn.Module):
    def __init__(self, num_users, num_items):
        super().__init__()

        self.embedding_user_mlp = torch.nn.Embedding(
            num_embeddings=num_users, embedding_dim=8
        )
        self.embedding_item_mlp = torch.nn.Embedding(
            num_embeddings=num_items, embedding_dim=8
        )
        self.embedding_user_mf = torch.nn.Embedding(
            num_embeddings=num_users, embedding_dim=8
        )
        self.embedding_item_mf = torch.nn.Embedding(
            num_embeddings=num_items, embedding_dim=8
        )

        self.net = nn.Sequential(
            nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8), nn.ReLU()
        )

        self.affine_output = torch.nn.Linear(in_features=8 + 8, out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_input, item_input):

        user_embedding_mlp = self.embedding_user_mlp(user_input)
        item_embedding_mlp = self.embedding_item_mlp(item_input)
        user_embedding_mf = self.embedding_user_mf(user_input)
        item_embedding_mf = self.embedding_item_mf(item_input)

        mlp_vector = torch.cat(
            [user_embedding_mlp, item_embedding_mlp], dim=-1
        )
        mf_vector = torch.mul(user_embedding_mf, item_embedding_mf)

        mlp_vector = self.net(mlp_vector)
        vector = torch.cat([mlp_vector, mf_vector], dim=-1)

        logits = self.affine_output(vector)
        rating = self.logistic(logits)

        return rating
