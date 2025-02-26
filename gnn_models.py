import pdb
from typing import Optional

import dgl
import dgl.function as fn
import torch as th
from torch import nn
import torch.nn.functional as F


class EdgeGraphConvLayer(nn.Module):

    def __init__(self, edge_feat_dim: int, node_feat_dim: int, num_heads: int) -> None:
        super(EdgeGraphConvLayer, self).__init__()
        self.num_heads = num_heads
        self.node_feat_dim = node_feat_dim
        self.relative_layer = nn.Sequential(
            nn.Linear(edge_feat_dim, node_feat_dim),
            nn.LayerNorm(node_feat_dim),
            nn.ReLU(inplace=True),
        )
        self.attention_linear = nn.Sequential(
            nn.Linear(3 * node_feat_dim, num_heads),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.weight = nn.Linear(node_feat_dim, node_feat_dim * num_heads)
        self.output_activation = nn.PReLU()

    def message_func(self, edges):
        weighted_edge_feat = self.relative_layer(edges.data["feat"])
        tmp = th.cat((edges.dst["h"], edges.src["h"], weighted_edge_feat), dim=1)
        u = self.attention_linear(tmp)
        return {"u": u, "weighted_src": self.weight(edges.src["h"])}

    def reduce_func(self, nodes):
        prefix_shape = nodes.mailbox["u"].shape[:-1]
        u_mh = nodes.mailbox["u"].view(*prefix_shape, self.num_heads, 1)
        weighted_src_mh = nodes.mailbox["weighted_src"].view(*prefix_shape, self.num_heads, self.node_feat_dim)
        alpha = F.softmax(u_mh.permute(0, 2, 1, 3), dim=2)
        h_mh = th.sum(alpha * weighted_src_mh.permute(0, 2, 1, 3), dim=2)
        return {"h": self.output_activation(th.sum(h_mh, dim=1) / self.num_heads)}

    def forward(self, g: dgl.DGLGraph):
        if g.number_of_edges() > 0:
            g.update_all(self.message_func, self.reduce_func)
        return g.ndata["h"]
