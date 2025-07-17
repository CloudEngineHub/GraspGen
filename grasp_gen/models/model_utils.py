# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# Author: Wentao Yuan, Adithya Murali
"""
Utility functions for network.
"""
import collections
import math

import numpy as np
import torch
import torch.nn as nn

from grasp_gen.models.pointnet.pointnet2_modules import PointnetSAModule
from grasp_gen.utils.logging_config import get_logger

logger = get_logger(__name__)


def load_pretrained_checkpoint_to_dict(checkpoint_path, model_name):
    """Loads and processes a pretrained model checkpoint.

    This function loads a checkpoint file and extracts the weights for a specific model
    component, modifying the state dict keys to match the target model's structure.

    Args:
        checkpoint_path (str): Path to the checkpoint file
        model_name (str): Name of the model component to extract weights for

    Returns:
        collections.OrderedDict: Processed state dictionary containing model weights
    """
    ckpt = torch.load(checkpoint_path)["model"]
    logger.info(
        f"Loading pretrained {model_name} checkpoint from file {checkpoint_path}"
    )
    new_ckpt_dict = collections.OrderedDict()
    for key in ckpt:
        if key.find(model_name) >= 0:
            key_modified = key[key.find(".") + 1 :]
            new_ckpt_dict[key_modified] = ckpt[key]
    return new_ckpt_dict


def convert_to_ptv3_pc_format(point_cloud: torch.Tensor, grid_size: float = 0.01):
    device = point_cloud.device
    batch_size = point_cloud.shape[0]
    num_points = point_cloud.shape[1]
    data_dict = dict()
    data_dict["coord"] = point_cloud.reshape([-1, 3]).to(device)
    data_dict["grid_size"] = grid_size
    data_dict["feat"] = point_cloud.reshape([-1, 3]).to(device)
    data_dict["offset"] = (
        torch.tensor([num_points]).repeat(batch_size).cumsum(dim=0).to(device)
    )
    return data_dict


def repeat_new_axis(tensor, rep, dim):
    reps = [1] * len(tensor.shape)
    reps.insert(dim, rep)
    return tensor.unsqueeze(dim).repeat(*reps)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        batch_size = x.shape[0]
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        if len(x.shape) == 1:
            emb = x[:, None] * emb[None, :]
        else:
            emb = x[:, :, None] * emb[None, None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        emb = emb.reshape([batch_size, -1])
        return emb


def break_up_pc(pc):
    xyz = pc[..., 0:3].contiguous()
    features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
    return xyz, features


def offset2bincount(offset):
    return torch.diff(
        offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long)
    )


def offset2batch(offset):
    bincount = offset2bincount(offset)
    return torch.arange(
        len(bincount), device=offset.device, dtype=torch.long
    ).repeat_interleave(bincount)


def compute_grasp_loss(target_grasps, pred_grasps, ctr_pts):
    ctr_pts = ctr_pts.to(device=target_grasps.device)
    pred_pts = (pred_grasps @ ctr_pts).transpose(-2, -1)[..., :3]
    gt_pts = (target_grasps @ ctr_pts).transpose(-2, -1)[..., :3]

    loss = torch.sum(torch.abs(pred_pts - gt_pts), -1)  # Sum across xyz dimension
    loss = torch.mean(loss, -1)  # Mean over control points
    return torch.mean(loss)  # Mean over batch


def get_activation_fn(activation):
    return getattr(nn, activation)()


OBJ_NPOINTS = [256, 64, None]
OBJ_RADII = [0.02, 0.04, None]
OBJ_NSAMPLES = [64, 128, None]
OBJ_MLPS = [[0, 64, 128], [128, 128, 256], [256, 256, 512]]
SCENE_PT_MLP = [3, 128, 256]
SCENE_VOX_MLP = [256, 512, 1024, 512]
CLS_FC = [2057, 1024, 256]


class PointNetPlusPlus(nn.Module):
    def __init__(
        self, activation="relu", bn=False, output_embedding_dim=512, feature_dim=-1
    ):
        mlp = []
        for elem in OBJ_MLPS:
            mlp.append(elem.copy())
        if feature_dim > 0:
            mlp[0][0] = feature_dim
        super().__init__()
        self.activation = activation
        self.output_embedding_dim = output_embedding_dim
        self.obj_SA_modules = nn.ModuleList()
        for k in range(OBJ_NPOINTS.__len__()):
            self.obj_SA_modules.append(
                PointnetSAModule(
                    npoint=OBJ_NPOINTS[k],
                    radius=OBJ_RADII[k],
                    nsample=OBJ_NSAMPLES[k],
                    mlp=mlp[k],
                    use_xyz=True,
                )
            )

        self.prediction_head = nn.Sequential(
            nn.Linear(OBJ_MLPS[-1][-1], 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.output_embedding_dim),
        )

    def forward(self, pc):
        xyz, features = break_up_pc(pc)
        for i in range(len(self.obj_SA_modules)):
            # new_xyz, new_xyz_idx, features, sample_ids
            xyz, _, features, _ = self.obj_SA_modules[i](xyz, features)
        features = self.prediction_head(features.squeeze(axis=-1))
        return features


class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers,
        activation="ReLU",
        dropout=0.0,
    ):
        super().__init__()
        h = [hidden_dim] * (num_layers - 1)
        layers = []
        for m, n in zip([input_dim] + h[:-1], h):
            layers.extend([nn.Linear(m, n), get_activation_fn(activation)])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class AttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query, key, value, query_pos_enc, key_pos_enc, attn_mask=None):
        output, attn = self.attn(
            query + query_pos_enc, key + key_pos_enc, value, attn_mask=attn_mask
        )
        return self.norm(query + output)


class FFNLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, activation="ReLU"):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            get_activation_fn(activation),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        return self.norm(x + self.ff(x))


class PositionEncoding3D(nn.Module):
    """
    Generate sinusoidal positional encoding
    f(p) = (sin(2^0 pi p), cos(2^0 pi p), ..., sin(2^L pi pi), cos(2^L pi p))
    """

    def __init__(self, enc_dim, scale=np.pi, temperature=10000):
        super(PositionEncoding3D, self).__init__()
        self.enc_dim = enc_dim
        self.freq = np.ceil(enc_dim / 6)
        self.scale = scale
        self.temperature = temperature

    def forward(self, pos):
        pos_min = pos.min(dim=1, keepdim=True)[0]
        pos_max = pos.max(dim=1, keepdim=True)[0]
        pos = ((pos - pos_min) / (pos_max - pos_min) - 0.5) * 2 * np.pi
        dim_t = torch.arange(self.freq, dtype=torch.float32, device=pos.device)
        dim_t = self.temperature ** (dim_t / self.freq)
        pos = pos[..., None] * self.scale / dim_t  # (B, N, 3, F)
        pos = torch.stack([pos.sin(), pos.cos()], dim=-1).flatten(start_dim=2)
        pos = pos[..., : self.enc_dim].transpose(1, 2)
        return pos.detach()


class PositionEncodingOld3D(nn.Module):
    """
    Generate sinusoidal positional encoding
    f(p) = (sin(2^0 pi p), cos(2^0 pi p), ..., sin(2^L pi pi), cos(2^L pi p))
    """

    def __init__(self, enc_dim, scale=np.pi, temperature=10000):
        super(PositionEncodingOld3D, self).__init__()
        self.enc_dim = enc_dim
        self.freq = np.ceil(enc_dim / 6)
        self.scale = scale
        self.temperature = temperature

    def forward(self, pos):
        pos_min = pos.min(dim=-1, keepdim=True)[0]
        pos_max = pos.max(dim=-1, keepdim=True)[0]
        pos = ((pos - pos_min) / (pos_max - pos_min) - 0.5) * 2 * np.pi
        dim_t = torch.arange(self.freq, dtype=torch.float32, device=pos.device)
        dim_t = self.temperature ** (dim_t / self.freq)
        pos = pos[..., None] * self.scale / dim_t  # (B, N, 3, F)
        pos = torch.stack([pos.sin(), pos.cos()], dim=-1).flatten(start_dim=2)
        pos = pos[..., : self.enc_dim].transpose(1, 2)
        return pos.detach()
