import pdb
from typing import Optional, Tuple, List
import argparse
import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
from modules.gnn_models import EdgeGraphConvLayer


class LayerNorm(nn.Module):
    """
    Layer normalization.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(th.ones(hidden_size))
        self.bias = nn.Parameter(th.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: th.Tensor) -> th.Tensor:
        if x.size(-1) == 1:
            return x
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / th.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class MLP(nn.Module):
    def __init__(self, hidden_size: int, out_features: Optional[int] = None) -> None:
        super(MLP, self).__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear = nn.Linear(hidden_size, out_features)
        self.layer_norm = LayerNorm(out_features)

    def forward(self, hidden_states: th.Tensor) -> th.Tensor:
        hidden_states = self.linear(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = F.relu(hidden_states)
        return hidden_states


class MlpGate(nn.Module):

    def __init__(self, hidden_size: int, out_features: Optional[int] = None) -> None:
        super(MlpGate, self).__init__()
        if out_features is None:
            out_features = hidden_size

        self.linear = nn.Linear(hidden_size, out_features)
        self.layer_norm = LayerNorm(out_features)

    def forward(self, hidden_states: th.Tensor) -> th.Tensor:
        hidden_states = self.linear(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = F.sigmoid(hidden_states)
        return hidden_states


class GatedFusion(nn.Module):

    def __init__(self, args: argparse.Namespace, hidden_size: Optional[int] = None) -> None:
        super(GatedFusion, self).__init__()
        self.hidden_size = hidden_size or args.hidden_size
        self.encoder_weight = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Sigmoid()
        )

    def forward(self, past_feat: th.Tensor, dest_feat: th.Tensor) -> th.Tensor:
        fuse = th.cat((past_feat, dest_feat), dim=-1)
        weight = self.encoder_weight(fuse)
        fused = past_feat * weight + dest_feat * (1 - weight)
        return fused


def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        fan_in = m.in_channels / m.groups
        fan_out = m.out_channels / m.groups
        bound = (6.0 / (fan_in + fan_out)) ** 0.5
        nn.init.uniform_(m.weight, -bound, bound)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.MultiheadAttention):
        if m.in_proj_weight is not None:
            fan_in = m.embed_dim
            fan_out = m.embed_dim
            bound = (6.0 / (fan_in + fan_out)) ** 0.5
            nn.init.uniform_(m.in_proj_weight, -bound, bound)
        else:
            nn.init.xavier_uniform_(m.q_proj_weight)
            nn.init.xavier_uniform_(m.k_proj_weight)
            nn.init.xavier_uniform_(m.v_proj_weight)
        if m.in_proj_bias is not None:
            nn.init.zeros_(m.in_proj_bias)
        nn.init.xavier_uniform_(m.out_proj.weight)
        if m.out_proj.bias is not None:
            nn.init.zeros_(m.out_proj.bias)
        if m.bias_k is not None:
            nn.init.normal_(m.bias_k, mean=0.0, std=0.02)
        if m.bias_v is not None:
            nn.init.normal_(m.bias_v, mean=0.0, std=0.02)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(4, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(4, 0):
                    nn.init.orthogonal_(hh)
            elif 'weight_hr' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)
                nn.init.ones_(param.chunk(4, 0)[1])
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(3, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(3, 0):
                    nn.init.orthogonal_(hh)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = th.zeros(max_len, d_model)
        position = th.arange(0, max_len, dtype=th.float).unsqueeze(1)
        div_term = th.exp(
            th.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = th.sin(position * div_term)
        pe[:, 1::2] = th.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class LinearPredictor(nn.Module):
    '''
    main func of the pmm-net
    '''
    def __init__(self, args: argparse.Namespace, device: th.device) -> None:
        super(LinearPredictor, self).__init__()
        self.device = device
        self.input_dim = args.input_size
        self.output_dim = args.output_size
        self.input_len = args.obs_length  # type: int
        self.output_len = args.pred_length  # type: int
        self.embedding_size = 64  # E
        self.num_tokens = self.input_len  # self.input_len - 2
        self.social_ctx_dim = 256  # S
        self.hidden_size = 1024  # C
        self.num_samples = 20  # K
        self.projector = nn.Sequential(
            nn.Linear(self.embedding_size * self.num_tokens, self.hidden_size * self.num_samples),
            # nn.Dropout(0.05),
            nn.LayerNorm(self.hidden_size * self.num_samples),
            nn.ReLU(inplace=True),
        )
        self.output_layer = nn.Linear(self.social_ctx_dim, self.output_len * self.output_dim)
        self.positional_encoding = PositionalEncoding(d_model=self.embedding_size, dropout=0.0, max_len=24)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_size,
            nhead=args.x_encoder_head,
            dim_feedforward=self.embedding_size,
            dropout=0.0,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.social_ctx_dim,
            nhead=8,
            dim_feedforward=self.social_ctx_dim,
            dropout=0.0,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.patch_mlp_x = nn.Sequential(
            nn.Linear(3, self.embedding_size),
            nn.ReLU(inplace=True),
        )
        self.patch_mlp_y = nn.Sequential(
            nn.Linear(3, self.embedding_size),
            nn.ReLU(inplace=True),
        )
        self.temporal_channel_fusion = GatedFusion(args, hidden_size=self.embedding_size)
        self.social_channel_fusion = GatedFusion(args, hidden_size=self.social_ctx_dim)
        self.gru = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.embedding_size,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )
        self.inverted_mlp = nn.Linear(self.input_len, self.social_ctx_dim)
        self.emb_mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.embedding_size),
            nn.ReLU(inplace=True),
        )
        self.gnn_layer = EdgeGraphConvLayer(edge_feat_dim=3, node_feat_dim=self.social_ctx_dim, num_heads=4)
        self.output_dim_reduction = nn.Sequential(
            nn.Linear(self.hidden_size, self.social_ctx_dim),
            nn.PReLU(),
        )

    def mlp_embedding(self, x: th.Tensor) -> th.Tensor:
        return self.emb_mlp(x)

    def patch_embedding(self, x: th.Tensor) -> th.Tensor:
        patches = x.permute(0, 2, 1).unfold(dimension=2, size=3, step=1)  # (B, 2, H - 2, 3)
        patch_emb_x = self.patch_mlp_x(patches[:, 0])  # (B, H - 2, E)
        patch_emb_y = self.patch_mlp_y(patches[:, 1])
        return self.temporal_channel_fusion(patch_emb_x, patch_emb_y)

    def inverted_embedding(self, x: th.Tensor) -> th.Tensor:
        x_inv_emb = self.inverted_mlp(x.permute(0, 2, 1))  # (B, H, 2) -> (B, 2, E)
        return self.social_channel_fusion(x_inv_emb[:, 0], x_inv_emb[:, 1])

    def forward(self, x: th.Tensor, vel, pos, nei_lists, batch_splits) -> th.Tensor:
        # x: (B, H, 2)
        batch_size = x.shape[0]
        # x_emb = self.patch_embedding(x)  # (B, H, E)
        x_emb = self.mlp_embedding(x)

        # c_0 = th.full((1, batch_size, self.embedding_size), 0, device=self.device).float()  # (B, 64), empty token
        # social_ctx = th.full((batch_size, self.embedding_size), 0, device=self.device).float()  # (B, 64), empty token
        nei_emb = self.inverted_embedding(x)
        social_ctx = self._get_social_context(vel, pos, nei_emb, nei_lists, batch_splits)  # (B, S)

        x_emb = self.positional_encoding(x_emb.permute(1, 0, 2)).permute(1, 0, 2)
        enc_out = self.transformer_encoder(x_emb)
        enc_out, _ = self.gru(enc_out)
        # dec_out = self.transformer_decoder(tgt=enc_out, memory=social_ctx.unsqueeze(1))

        latent = self.projector(enc_out.reshape(batch_size, -1))
        # latent = latent.view(batch_size, self.num_samples, self.hidden_size).permute(1, 0, 2)  # (K, B, C)
        latent = latent.view(batch_size, self.num_samples, self.hidden_size)  # (B, K, C)
        latent = self.output_dim_reduction(latent)  # (B, K, S)
        dec_out = self.transformer_decoder(tgt=latent, memory=social_ctx.unsqueeze(1))  # (B, K, S)

        output = self.output_layer(latent)
        # output = output.view(self.num_samples, batch_size, self.output_len, self.output_dim)  # (K, B, H, 2)
        output = output.view(batch_size, self.num_samples, self.output_len, self.output_dim)  # (B, K, H, 2)
        return output.transpose(0, 1)

    def _get_social_context(
        self,
        pseudo_vel_batch: th.Tensor,
        current_pos_batch: th.Tensor,
        node_feature_batch: th.Tensor,
        nei_lists: List[np.ndarray],
        batch_splits: List[List[int]],
    ) -> th.Tensor:
        # nei_lists: (B, H, N, N), current_pos_batch (B, 2)
        refined_feature_batch = th.zeros_like(node_feature_batch, device=self.device)
        batch_node_mask = th.full((node_feature_batch.shape[0],), False, dtype=th.bool, device=self.device)
        graphs = []

        for batch_idx in range(len(nei_lists)):
            # ids for select agents in the same scene
            left, right = batch_splits[batch_idx][0], batch_splits[batch_idx][1]
            node_features = node_feature_batch[left: right]  # (N, D)
            pseudo_vel = pseudo_vel_batch[left: right, :]  # velocity of the agent
            ped_num = node_features.shape[0]  # num_nodes in current scene
            nei_index = th.tensor(nei_lists[batch_idx][self.input_len - 1], device=self.device)  # (N, N)
            # build graph based on adjacency matrix
            scene_graph = dgl.graph(nei_index.nonzero().unbind(1), num_nodes=ped_num, device=self.device)

            if ped_num != 1:
                corr = current_pos_batch[left: right].repeat(ped_num, 1, 1)  # (N, N, 2)
                corr_index = corr.transpose(0, 1) - corr  # (N, N, 2)  d_{ij} in matrix form

                pseudo_vel_norm = th.norm(pseudo_vel, dim=1, keepdim=True)  # (N, 1)
                # \cos\theta_ij in matrix form
                angle_mat = (
                    th.einsum('ij,nij->ni', pseudo_vel, corr_index) /
                    (th.norm(corr_index, dim=2, keepdim=True).sum(-1) * pseudo_vel_norm.transpose(0, 1) + 1e-10)
                )
                zero_velocity_mask = pseudo_vel_norm.squeeze() == 0
                zero_velocity_indices = zero_velocity_mask.nonzero(as_tuple=False)
                if zero_velocity_indices.numel() > 0:
                    angle_mat[:, zero_velocity_mask] = -1

                angle_mat = angle_mat.reshape(-1, 1)
                corr_mat = corr_index.reshape((ped_num * ped_num, -1))  # (N * N, 2)
                edge_feature_mat = th.cat((corr_mat, angle_mat), dim=-1)  # (N * N, 3)
                edge_mask = nei_index.view(ped_num * ped_num) > 0  # (N * N)
                node_mask = th.sum(nei_index, dim=1) > 0  # at least one edge exist for this node
                batch_node_mask[left: right] = node_mask  # update batch node mask
                scene_graph.ndata["h"] = node_features
                if scene_graph.number_of_edges() > 0:
                    scene_graph.edata["feat"] = edge_feature_mat[edge_mask]
            else:  # one node, no edge
                scene_graph.ndata["h"] = node_features

            graphs.append(scene_graph)

        graph_batch = dgl.batch(graphs)
        try:
            refined_feature_batch[batch_node_mask] = self.gnn_layer(graph_batch)[batch_node_mask]
        except IndexError:
            pdb.set_trace()
        return refined_feature_batch
