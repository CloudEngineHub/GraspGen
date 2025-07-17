#!/usr/bin/env python3

# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import time

import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from scipy.spatial import KDTree

from grasp_gen.utils.math_utils import matrix_to_rt, rt_to_matrix
from grasp_gen.metrics import compute_metrics_given_two_sets_of_poses, compute_recall
from grasp_gen.models.model_utils import (
    PointNetPlusPlus,
    SinusoidalPosEmb,
    compute_grasp_loss,
    convert_to_ptv3_pc_format,
    load_pretrained_checkpoint_to_dict,
    offset2batch,
)
from grasp_gen.models.ptv3.ptv3 import PointTransformerV3
from grasp_gen.robot import get_gripper_info
from grasp_gen.utils.logging_config import get_logger

logger = get_logger(__name__)


class GraspGenGenerator(nn.Module):
    """GraspGen generator model for generating 6-DOF robotic grasps.

    This class implements a diffusion model that generates robotic grasping poses from point cloud observations.
    It uses a combination of object encoding and diffusion-based denoising to generate high-quality grasps.

    Args:
        num_embed_dim (int): Dimension of embedding vectors. Default: 256
        num_obs_dim (int): Dimension of observation features. Default: 512
        diffusion_embed_dim (int): Dimension of diffusion step embeddings. Default: 512
        image_size (int): Size of input images if using vision backbone. Default: 256
        num_diffusion_iters (int): Number of diffusion steps for training. Default: 100
        num_diffusion_iters_eval (int): Number of diffusion steps for evaluation. Default: 100
        obs_backbone (str): Type of observation encoder backbone ('vit', 'pointnet', 'ptv3'). Default: 'vit'
        compositional_schedular (bool): Whether to use separate schedulers for position and rotation. Default: False
        loss_pointmatching (bool): Whether to use point matching loss. Default: True
        loss_l1_pos (bool): Whether to use L1 loss on positions. Default: False
        loss_l1_rot (bool): Whether to use L1 loss on rotations. Default: False
        grasp_repr (str): Grasp representation type ('r3_6d', 'r3_so3', 'r3_euler'). Default: 'r3_6d'
        kappa (float): Scale factor for noise. Default: -1.0
        clip_sample (bool): Whether to clip samples in diffusion process. Default: True
        beta_schedule (str): Schedule for noise variance. Default: 'beta_schedule'
        attention (str): Type of attention mechanism. Default: 'cat'
        grid_size (float): Grid size for point cloud processing. Default: 0.02
        gripper_name (str): Name of the gripper model. Default: 'franka_panda'
        pose_repr (str): Type of pose representation. Default: 'mlp'
        num_grasps_per_object (int): Number of grasps to generate per object. Default: 20
        checkpoint_object_encoder_pretrained (str): Path to pretrained object encoder. Default: None
    """

    def __init__(
        self,
        num_embed_dim: int = 256,
        num_obs_dim: int = 512,
        diffusion_embed_dim: int = 512,
        image_size: int = 256,
        num_diffusion_iters: int = 100,
        num_diffusion_iters_eval: int = 100,
        obs_backbone: str = "pointnet",
        compositional_schedular: bool = False,
        loss_pointmatching: bool = True,
        loss_l1_pos: bool = False,
        loss_l1_rot: bool = False,
        grasp_repr: str = "r3_6d",
        kappa: float = -1.0,
        clip_sample: bool = True,
        beta_schedule: str = "beta_schedule",
        attention: str = "cat",
        grid_size: float = 0.02,
        gripper_name: str = "franka_panda",
        pose_repr: str = "mlp",
        num_grasps_per_object: int = 20,
        checkpoint_object_encoder_pretrained: str = None,
    ):
        super().__init__()

        self.num_embed_dim = num_embed_dim
        self.num_obs_dim = num_obs_dim
        self.diffusion_embed_dim = diffusion_embed_dim
        self.image_size = image_size
        self.num_diffusion_iters = num_diffusion_iters
        self.num_diffusion_iters_eval = num_diffusion_iters_eval
        self.obs_backbone = obs_backbone
        self.compositional_schedular = compositional_schedular
        self.loss_pointmatching = loss_pointmatching
        self.loss_l1_pos = loss_l1_pos
        self.loss_l1_rot = loss_l1_rot
        self.grasp_repr = grasp_repr
        self.kappa = None if kappa <= 0 else kappa
        self.clip_sample = clip_sample
        self.beta_schedule = beta_schedule
        self.attention = attention
        self.grid_size = grid_size
        self.gripper_name = gripper_name
        self.pose_repr = pose_repr
        self.num_grasps_per_object = num_grasps_per_object
        self.checkpoint_object_encoder_pretrained = checkpoint_object_encoder_pretrained

        if self.grasp_repr == "r3_6d":
            self.output_dim = 9
        elif self.grasp_repr in ["r3_so3", "r3_euler"]:
            self.output_dim = 6
        else:
            raise NotImplementedError(
                f"Rotation representation {grasp_repr} is not implemented!"
            )

        if obs_backbone == "vit":
            from grasp_gen.models.vit import VisionTransformer

            self.object_encoder = VisionTransformer(
                img_size=self.image_size,
                embed_dim=self.num_embed_dim,
                num_classes=self.num_obs_dim,
                patch_size=64,
                depth=12,
                num_heads=8,
            )
        elif obs_backbone == "pointnet":
            self.object_encoder = PointNetPlusPlus(
                output_embedding_dim=self.num_obs_dim,
                feature_dim=1 if self.pose_repr == "pc_feature" else -1,
            )
        elif obs_backbone == "ptv3":
            self.object_encoder = PointTransformerV3(
                in_channels=3,
                enable_flash=False,
                cls_mode=True,
            )
        else:
            raise NotImplementedError()

        self.diffusion_head = DiffusionNoisePredictionNet(
            diffusion_step_embed_dim=self.diffusion_embed_dim,
            observation_embed_dim=self.num_obs_dim,
            sample_embed_dim=self.diffusion_embed_dim,
            sample_dim=self.output_dim,
            attention=self.attention,
            moreparams=False,
            pose_repr=self.pose_repr,
        )

        if self.compositional_schedular:

            self.noise_scheduler_pos = DDPMScheduler(
                num_train_timesteps=self.num_diffusion_iters,
                beta_schedule="scaled_linear",
                clip_sample=True,
                prediction_type="epsilon",
            )

            self.noise_scheduler_rot = DDPMScheduler(
                num_train_timesteps=self.num_diffusion_iters,
                beta_schedule="squaredcos_cap_v2",
                clip_sample=True,
                prediction_type="epsilon",
            )

        else:
            logger.info(
                f"DDPM parameters, num_diffusion_iters: {self.num_diffusion_iters}, beta_schedule: {self.beta_schedule}, clip_sample: {self.clip_sample}"
            )
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.num_diffusion_iters,
                beta_schedule=self.beta_schedule,  # TODO: Check this
                clip_sample=self.clip_sample,
                prediction_type="epsilon",
            )

        self.gripper_info = get_gripper_info(self.gripper_name)
        self.gripper_mesh = self.gripper_info.collision_mesh
        self.ctr_pts = self.gripper_info.control_points

        if self.checkpoint_object_encoder_pretrained is not None:
            if os.path.exists(self.checkpoint_object_encoder_pretrained):
                model_state_dict_object_encoder = load_pretrained_checkpoint_to_dict(
                    self.checkpoint_object_encoder_pretrained, "object_encoder"
                )
                self.object_encoder.load_state_dict(model_state_dict_object_encoder)
                for param in self.object_encoder.parameters():
                    param.requires_grad = False
                logger.info("Using pretrained object encoder!")
            else:
                logger.info(
                    f"Object encoder checkpoints not found at location {self.checkpoint_object_encoder_pretrained}"
                )

    @classmethod
    def from_config(cls, cfg):
        """Creates a GraspGenGenerator instance from a configuration object.

        Args:
            cfg: Configuration object containing model parameters

        Returns:
            GraspGenGenerator: Instantiated model
        """
        args = {
            "num_embed_dim": cfg.num_embed_dim,
            "num_obs_dim": cfg.num_obs_dim,
            "diffusion_embed_dim": cfg.diffusion_embed_dim,
            "image_size": cfg.image_size,
            "num_diffusion_iters": cfg.num_diffusion_iters,
            "num_diffusion_iters_eval": cfg.num_diffusion_iters_eval,
            "obs_backbone": cfg.obs_backbone,
            "compositional_schedular": cfg.compositional_schedular,
            "loss_pointmatching": cfg.loss_pointmatching,
            "loss_l1_pos": cfg.loss_l1_pos,
            "loss_l1_rot": cfg.loss_l1_rot,
            "grasp_repr": cfg.grasp_repr,
            "kappa": cfg.kappa,
            "clip_sample": cfg.clip_sample,
            "beta_schedule": cfg.beta_schedule,
            "attention": cfg.attention,
            "grid_size": cfg.ptv3.grid_size,
            "gripper_name": cfg.gripper_name,
            "pose_repr": cfg.pose_repr,
            "num_grasps_per_object": cfg.num_grasps_per_object,
            "checkpoint_object_encoder_pretrained": cfg.checkpoint_object_encoder_pretrained,
        }
        return cls(**args)

    def forward(self, data, cfg=None, eval=False):
        """Forward pass of the model.

        Args:
            data: Input data dictionary containing point clouds and optionally ground truth grasps
            cfg: Optional configuration object
            eval (bool): Whether to run in evaluation mode

        Returns:
            tuple: (outputs, losses, stats) containing model predictions, losses and metrics
        """
        if eval:
            return self.forward_inference(data, return_metrics=True)
        else:
            return self.forward_train(data)

    def infer(self, data, return_metrics=False):
        """Inference method for generating grasps.

        Args:
            data: Input data dictionary containing point clouds
            return_metrics (bool): Whether to compute and return evaluation metrics

        Returns:
            tuple: (outputs, losses, stats) containing generated grasps and optional metrics
        """
        return self.forward_inference(data, return_metrics=return_metrics)

    def forward_train(self, data):
        """Training forward pass implementing the diffusion process.

        Args:
            data: Input data dictionary containing point clouds and ground truth grasps

        Returns:
            tuple: (outputs, losses, stats) containing predictions, training losses and metrics
        """
        device = data["points"].device
        num_objects_in_batch = len(data["points"])

        num_grasps_per_batch = data["grasps"][0].shape[0]
        batch_size = num_objects_in_batch * num_grasps_per_batch
        depth = data["points"]
        grasps = data["grasps"]

        num_points = depth.shape[-2]
        depth = depth.reshape([-1, num_points, 3])

        grasps_init_size = [num_objects_in_batch, num_grasps_per_batch, 4, 4]
        if type(grasps) == list:
            grasps = torch.cat(grasps)

        grasps = grasps.reshape([-1, 4, 4])

        if self.kappa is not None:
            depth = self.kappa * depth

        if self.obs_backbone == "ptv3":
            depth = convert_to_ptv3_pc_format(depth, grid_size=self.grid_size)

        grasps_gt = matrix_to_rt(grasps, self.grasp_repr, kappa=self.kappa)

        noise = torch.randn([batch_size, self.output_dim], device=device).float()

        timesteps = torch.randint(
            0, self.num_diffusion_iters, (batch_size,), device=device
        ).long()

        offset = (
            torch.tensor([num_grasps_per_batch])
            .repeat(num_objects_in_batch)
            .cumsum(dim=0)
            .to(device)
        )
        mask_batch = offset2batch(offset)

        if self.pose_repr in ["grasp_cloud", "grasp_cloud_pe", "pc_feature"]:
            ctrl_pts = self.ctr_pts.to(device=device)
            grasp_pc = (grasps @ ctrl_pts).transpose(-2, -1)[..., :3]

        if self.pose_repr == "pc_feature":
            depth_full = depth[mask_batch]
            depth_full = torch.cat([depth_full, grasp_pc], dim=1)
            pc_feature = torch.cat(
                [
                    torch.zeros(
                        [num_grasps_per_batch * num_objects_in_batch, num_points, 1]
                    ),
                    torch.ones(
                        [
                            num_grasps_per_batch * num_objects_in_batch,
                            grasp_pc.shape[1],
                            1,
                        ]
                    ),
                ],
                dim=1,
            ).to(device=device)

            object_embedding = torch.cat([depth_full, pc_feature], dim=-1)
            object_embedding = self.object_encoder(object_embedding)
        elif self.pose_repr == "mlp":
            object_embedding = self.object_encoder(
                depth
            )  # object_embedding size is [num_objects_in_batch, self.num_obs_dim]
            object_embedding = object_embedding[
                mask_batch
            ]  # Redistribute object embeddings to full batch, result is [batch_size, self.num_obs_dim]
        else:
            raise NotImplementedError(f"Pose repr {self.pose_repr} not implemented!")

        if self.compositional_schedular:
            noisy_grasps_pos = self.noise_scheduler_pos.add_noise(
                grasps_gt[..., :3], noise[..., :3], timesteps
            )
            noisy_grasps_rot = self.noise_scheduler_rot.add_noise(
                grasps_gt[..., 3 : self.output_dim],
                noise[..., 3 : self.output_dim],
                timesteps,
            )
            noisy_grasps = torch.hstack([noisy_grasps_pos, noisy_grasps_rot])
        else:
            noisy_grasps = self.noise_scheduler.add_noise(grasps_gt, noise, timesteps)

        samples = noisy_grasps if self.pose_repr == "mlp" else None
        noise_pred = self.diffusion_head(object_embedding, timesteps, samples)

        pred_noise_pts_mat = rt_to_matrix(noise_pred, self.grasp_repr, self.kappa)
        actual_noise_pts_mat = rt_to_matrix(noise, self.grasp_repr, self.kappa)
        noisy_grasps_mat = rt_to_matrix(noisy_grasps, self.grasp_repr, self.kappa)
        grasps_gt_mat = rt_to_matrix(grasps_gt, self.grasp_repr, self.kappa)

        stats = compute_metrics_given_two_sets_of_poses(
            actual_noise_pts_mat, pred_noise_pts_mat, self.gripper_info
        )

        losses = {}
        if self.loss_pointmatching:
            point_matching_loss = compute_grasp_loss(
                actual_noise_pts_mat, pred_noise_pts_mat, self.ctr_pts
            )
            losses["noise_pred"] = (2.0, point_matching_loss)

        if self.loss_l1_pos:
            position_loss = torch.linalg.norm(
                noise[..., :3] - noise_pred[..., :3], dim=-1
            )
            position_loss = torch.mean(position_loss)  # across the batch
            losses["position_loss"] = (1.0, position_loss)

        if self.loss_l1_rot:
            rotation_loss = torch.linalg.norm(
                noise[..., 3 : self.output_dim] - noise_pred[..., 3 : self.output_dim],
                dim=-1,
            )
            rotation_loss = torch.mean(rotation_loss)
            losses["rotation_loss"] = (1.0, rotation_loss)

        outputs = {}
        outputs["actual_noise_pts_mat"] = actual_noise_pts_mat.reshape(grasps_init_size)
        outputs["pred_noise_pts_mat"] = pred_noise_pts_mat.reshape(grasps_init_size)

        outputs["noisy_grasps_mat"] = noisy_grasps_mat.reshape(grasps_init_size)
        outputs["grasps_gt_mat"] = grasps_gt_mat.reshape(grasps_init_size)

        return outputs, losses, stats

    def forward_inference(self, data, return_metrics=False):
        """Inference forward pass implementing the reverse diffusion process.

        Args:
            data: Input data dictionary containing point clouds
            return_metrics (bool): Whether to compute evaluation metrics

        Returns:
            tuple: (outputs, losses, stats) containing generated grasps and optional metrics
        """

        device = data["points"].device

        num_objects_in_batch = len(data["points"])
        if "grasps" in data:
            if type(data["grasps"][0]) == list:
                data["grasps"][0] = np.array(data["grasps"][0])
            num_grasps_per_batch = data["grasps"][0].shape[0]
        else:
            num_grasps_per_batch = self.num_grasps_per_object
            return_metrics = False

        batch_size = data["points"].shape[0] * num_grasps_per_batch
        depth = data["points"]

        num_points = depth.shape[-2]

        depth = depth.reshape([-1, num_points, 3])
        depth = depth.to(device)

        grasps_init_size = [num_objects_in_batch, num_grasps_per_batch, 4, 4]

        if self.kappa is not None:
            depth = self.kappa * depth

        if self.obs_backbone == "ptv3":
            depth = convert_to_ptv3_pc_format(depth, grid_size=self.grid_size)
        # num_grasps_per_batch = data['grasps'].shape[1]
        grasps_per_iteration = torch.zeros(
            [
                num_objects_in_batch,
                self.num_diffusion_iters_eval,
                num_grasps_per_batch,
                4,
                4,
            ]
        )

        with torch.no_grad():
            noisy_init = torch.randn([batch_size, self.output_dim], device=device)
            noisy_grasps = noisy_init

            # Initialize likelihood scores
            likelihood = torch.zeros((batch_size, 1), device=device)

            offset = (
                torch.tensor([num_grasps_per_batch])
                .repeat(num_objects_in_batch)
                .cumsum(dim=0)
                .to(device)
            )
            mask_batch = offset2batch(offset)

            if self.pose_repr == "mlp":
                object_embedding = self.object_encoder(
                    depth
                )  # object_embedding size is [num_objects_in_batch, self.num_obs_dim]
                object_embedding = object_embedding[
                    mask_batch
                ]  # Redistribute object embeddings to full batch, result is [batch_size, self.num_obs_dim]

            if self.compositional_schedular:
                self.noise_scheduler_pos.set_timesteps(self.num_diffusion_iters_eval)
                timesteps = self.noise_scheduler_pos.timesteps
                self.noise_scheduler_rot.set_timesteps(self.num_diffusion_iters_eval)
            else:
                self.noise_scheduler.set_timesteps(self.num_diffusion_iters_eval)
                timesteps = self.noise_scheduler.timesteps

            for k in timesteps:
                samples = noisy_grasps if self.pose_repr == "mlp" else None

                if self.pose_repr in ["grasp_cloud", "grasp_cloud_pe", "pc_feature"]:
                    ctrl_pts = self.ctr_pts.to(device=device)

                    noisy_grasps_mat = rt_to_matrix(
                        noisy_grasps, self.grasp_repr, self.kappa
                    )
                    grasp_pc = (noisy_grasps_mat @ ctrl_pts).transpose(-2, -1)[..., :3]

                if self.pose_repr == "pc_feature":
                    depth_full = depth[mask_batch]
                    depth_full = torch.cat([depth_full, grasp_pc], dim=1)
                    pc_feature = torch.cat(
                        [
                            torch.zeros(
                                [
                                    num_grasps_per_batch * num_objects_in_batch,
                                    num_points,
                                    1,
                                ]
                            ),
                            torch.ones(
                                [
                                    num_grasps_per_batch * num_objects_in_batch,
                                    grasp_pc.shape[1],
                                    1,
                                ]
                            ),
                        ],
                        dim=1,
                    ).to(device=device)

                    object_embedding = torch.cat([depth_full, pc_feature], dim=-1)
                    object_embedding = self.object_encoder(object_embedding)

                # Forward: Predict noise
                noise_pred = self.diffusion_head(object_embedding, k, samples)

                if self.compositional_schedular:
                    # Handle compositional case
                    res_pos = self.noise_scheduler_pos.step(
                        model_output=noise_pred[..., :3],
                        timestep=k,
                        sample=noisy_grasps[..., :3],
                    )
                    res_rot = self.noise_scheduler_rot.step(
                        model_output=noise_pred[..., 3 : self.output_dim],
                        timestep=k,
                        sample=noisy_grasps[..., 3 : self.output_dim],
                    )

                    # Compute likelihood contributions
                    if k > 0:  # Skip first step
                        alpha_pos = self.noise_scheduler_pos.alphas[k]
                        beta_pos = self.noise_scheduler_pos.betas[k]
                        var_pos = beta_pos
                        likelihood_pos = (
                            torch.distributions.Normal(
                                res_pos.pred_original_sample,
                                torch.sqrt(torch.tensor(var_pos, device=device)),
                            )
                            .log_prob(noisy_grasps[..., :3])
                            .sum(-1, keepdim=True)
                        )

                        alpha_rot = self.noise_scheduler_rot.alphas[k]
                        beta_rot = self.noise_scheduler_rot.betas[k]
                        var_rot = beta_rot
                        likelihood_rot = (
                            torch.distributions.Normal(
                                res_rot.pred_original_sample,
                                torch.sqrt(torch.tensor(var_rot, device=device)),
                            )
                            .log_prob(noisy_grasps[..., 3 : self.output_dim])
                            .sum(-1, keepdim=True)
                        )

                        likelihood += likelihood_pos + likelihood_rot

                    noisy_grasps = torch.hstack(
                        [res_pos.prev_sample, res_rot.prev_sample]
                    )
                else:
                    # Handle standard case
                    res = self.noise_scheduler.step(
                        model_output=noise_pred, timestep=k, sample=noisy_grasps
                    )

                    # Compute likelihood contribution
                    if k > 0:  # Skip first step
                        alpha = self.noise_scheduler.alphas[k]
                        beta = self.noise_scheduler.betas[k]
                        var = beta
                        likelihood += (
                            torch.distributions.Normal(
                                res.pred_original_sample,
                                torch.sqrt(torch.tensor(var, device=device)),
                            )
                            .log_prob(noisy_grasps)
                            .sum(-1, keepdim=True)
                        )

                    noisy_grasps = res.prev_sample

                pred_grasps = rt_to_matrix(noisy_grasps, self.grasp_repr, self.kappa)

                grasps_pred = pred_grasps.reshape(grasps_init_size)

                grasps_per_iteration[:, k, :, ::] = grasps_pred

        grasps_pred = pred_grasps.reshape(grasps_init_size)
        grasps_pred[:, :, 3, 3] = 1  # To make proper homogeneous matrix

        stats_batch = []

        if return_metrics:
            all_stats = []
            for i in range(num_objects_in_batch):

                grasps_pred_i = grasps_pred[i].cpu().numpy()
                grasps_gt_i = data["grasps_highres"][i].cpu().numpy()

                tree = KDTree(grasps_gt_i[:, :3, 3])
                dist, idx = tree.query(grasps_pred_i[:, :3, 3])
                matched = dist < 4.0
                idx = idx[matched]

                grasps_pred_matched = grasps_pred_i[matched]
                grasps_gt_for_pred = grasps_gt_i[idx]

                grasps_pred_matched = torch.from_numpy(grasps_pred_matched)
                grasps_gt_for_pred = torch.from_numpy(grasps_gt_for_pred)
                stats = compute_metrics_given_two_sets_of_poses(
                    grasps_gt_for_pred,
                    grasps_pred_matched,
                    self.gripper_info,
                    consider_symmetry=True,
                )

                recall = compute_recall(grasps_gt_i, grasps_pred_i)
                precision = compute_recall(grasps_pred_i, grasps_gt_i)

                stats["recall"] = torch.tensor(recall).to(device)
                stats["precision"] = torch.tensor(precision).to(device)

                all_stats.append(stats)

            stats_keys = all_stats[0].keys()
            stats_batch = {}

            for key in stats_keys:
                stats_batch[key] = torch.mean(
                    torch.tensor([stats[key] for stats in all_stats]).to(device)
                )

        outputs = {
            "grasps_pred": grasps_pred,
            "grasps_per_iteration": grasps_per_iteration,
            "grasp_confidence": torch.zeros(grasps_pred.shape[0]),
            "grasping_masks": torch.zeros(grasps_pred.shape[0]),
            "grasp_contacts": torch.zeros(grasps_pred.shape[0]),
            "instance_masks": torch.zeros(grasps_pred.shape[0]),
            "likelihood": likelihood.reshape(
                num_objects_in_batch, num_grasps_per_batch, 1
            ),
        }
        return outputs, {}, stats_batch


class DiffusionNoisePredictionNet(nn.Module):
    """Neural network module implementing the diffusion model's denoising network.

    This network predicts the noise in the diffused samples given the current noisy sample,
    diffusion timestep, and object observation embedding.

    Args:
        diffusion_step_embed_dim (int): Dimension for diffusion step embeddings. Default: 512
        observation_embed_dim (int): Dimension of object observation embeddings. Default: 512
        sample_embed_dim (int): Dimension for sample embeddings. Default: 512
        sample_dim (int): Dimension of the grasp samples. Default: 9
        moreparams (bool): Whether to use additional parameters. Default: False
        attention (bool): Whether to use attention mechanisms. Default: False
        pose_repr (str): Type of pose representation. Default: 'mlp'
    """

    def __init__(
        self,
        diffusion_step_embed_dim=512,
        observation_embed_dim=512,
        sample_embed_dim=512,
        sample_dim=9,
        moreparams=False,
        attention=False,
        pose_repr="mlp",
    ):

        self.attention = attention
        self.pose_repr = pose_repr
        super().__init__()

        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )

        if self.pose_repr == "mlp":
            self.sample_encoder = nn.Sequential(
                nn.Linear(sample_dim, sample_embed_dim),
                nn.ReLU(),
                nn.Linear(sample_embed_dim, sample_embed_dim),
            )

            total_input_dim = (
                sample_embed_dim + diffusion_step_embed_dim + observation_embed_dim
            )
        else:
            total_input_dim = diffusion_step_embed_dim + observation_embed_dim

        self.prediction_head = nn.Sequential(
            nn.Linear(total_input_dim, total_input_dim // 2),
            nn.ReLU(),
            nn.Linear(total_input_dim // 2, total_input_dim // 4),
            nn.ReLU(),
            nn.Linear(total_input_dim // 4, sample_dim),
        )

        if self.attention.find("attn") > 0:
            from grasp_gen.models.model_utils import AttentionLayer, FFNLayer

            # transformer decoder
            self.embed_dim = total_input_dim
            self.num_heads = 8
            self.num_layers = 3
            self.feedforward_dim = 512
            self.activation = "GELU"
            num_grasp_queries = 1

            if self.attention.find("cross") >= 0:
                self.obs_pos_enc = nn.Embedding(1, observation_embed_dim)
                self.sample_pos_enc = nn.Embedding(1, sample_embed_dim)

                self.time_pos_enc = nn.Embedding(1, diffusion_step_embed_dim)

                self.query_embed = nn.Embedding(1, self.embed_dim)
                self.query_pos_enc = nn.Embedding(1, self.embed_dim)
                self.self_attention_layers = nn.ModuleList()
                self.cross_attention_layers = nn.ModuleList()
                self.ffn_layers = nn.ModuleList()
                for _ in range(self.num_layers):
                    self.self_attention_layers.append(
                        AttentionLayer(self.embed_dim, self.num_heads)
                    )
                    self.cross_attention_layers.append(
                        AttentionLayer(self.embed_dim, self.num_heads)
                    )
                    self.ffn_layers.append(
                        FFNLayer(self.embed_dim, self.feedforward_dim, self.activation)
                    )

            else:
                self.query_pos_enc = nn.Embedding(num_grasp_queries, self.embed_dim)

                self.self_attention_layers = nn.ModuleList()
                self.ffn_layers = nn.ModuleList()
                for _ in range(self.num_layers):
                    self.self_attention_layers.append(
                        AttentionLayer(self.embed_dim, self.num_heads)
                    )
                    self.ffn_layers.append(
                        FFNLayer(self.embed_dim, self.feedforward_dim, self.activation)
                    )

    def forward(
        self,
        observation_embedding: torch.Tensor,
        timesteps: torch.Tensor,
        sample: torch.Tensor = None,
    ):
        """Forward pass of the diffusion denoising network.

        Args:
            observation_embedding (torch.Tensor): Object observation embeddings
            timesteps (torch.Tensor): Current diffusion timesteps
            sample (torch.Tensor, optional): Current noisy samples

        Returns:
            torch.Tensor: Predicted noise in the samples
        """

        device = observation_embedding.device

        if torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(device)
            timesteps = timesteps.expand(observation_embedding.shape[0])

        timestep_embedding = self.diffusion_step_encoder(timesteps)

        if self.pose_repr == "mlp":
            sample_embedding = self.sample_encoder(sample)

        if self.attention.find("attn") >= 0:

            if self.attention.find("cross") >= 0:
                from grasp_gen.models.model_utils import repeat_new_axis

                # t0 = time.time()

                batch_size = timestep_embedding.shape[0]

                embed = repeat_new_axis(self.query_embed.weight, batch_size, dim=1)
                query_pos_enc = repeat_new_axis(
                    self.query_pos_enc.weight, batch_size, dim=1
                )

                obs_pos_enc = repeat_new_axis(
                    self.obs_pos_enc.weight, batch_size, dim=1
                )

                time_pos_enc = repeat_new_axis(
                    self.time_pos_enc.weight, batch_size, dim=1
                )

                # TODO - Improve this...
                sample_pos_enc = repeat_new_axis(
                    self.sample_pos_enc.weight, batch_size, dim=1
                )

                cross_embed = torch.cat(
                    [timestep_embedding, observation_embedding, sample_embedding],
                    axis=1,
                ).unsqueeze(0)
                cross_embed_pos_enc = torch.cat(
                    [time_pos_enc, obs_pos_enc, sample_pos_enc], axis=2
                )

                for i in range(self.num_layers):

                    embed = self.cross_attention_layers[i](
                        embed,
                        cross_embed,
                        cross_embed + cross_embed_pos_enc,
                        query_pos_enc,
                        cross_embed_pos_enc,
                    )

                    embed = self.self_attention_layers[i](
                        embed,
                        embed,
                        embed + query_pos_enc,
                        query_pos_enc,
                        query_pos_enc,
                    )
                    embed = self.ffn_layers[i](embed)
            else:
                from grasp_gen.models.model_utils import repeat_new_axis

                t0 = time.time()

                if self.pose_repr == "mlp":
                    embed = torch.cat(
                        [sample_embedding, timestep_embedding, observation_embedding],
                        axis=-1,
                    )
                    # print(f"Concatenation took {time.time() - t0}s")
                else:
                    embed = torch.cat(
                        [timestep_embedding, observation_embedding], axis=-1
                    )
                embed = embed.unsqueeze(0)
                batch_size = embed.shape[1]

                query_pos_enc = repeat_new_axis(
                    self.query_pos_enc.weight, batch_size, dim=1
                )

                t0 = time.time()
                for i in range(self.num_layers):

                    embed = self.self_attention_layers[i](
                        embed,
                        embed,
                        embed + query_pos_enc,
                        query_pos_enc,
                        query_pos_enc,
                    )
                    embed = self.ffn_layers[i](embed)
                # print(f"Attention took {time.time() - t0}s")
            embed = embed.squeeze(0)
        else:
            t0 = time.time()
            if self.pose_repr == "mlp":
                embed = torch.cat(
                    [sample_embedding, timestep_embedding, observation_embedding],
                    axis=-1,
                )
            else:
                embed = torch.cat([timestep_embedding, observation_embedding], axis=-1)

        return self.prediction_head(embed)
