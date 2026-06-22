# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Visualize grasps on scene point clouds. Supports two input formats:

  1. **realworld** — ``<NN>/{depth.npy, rgb.png, seg.png, meta_data.json}``
     (back-projects depth via intrinsics + camera_pose, segments into
     ``obj_*`` per ``label_map``). Multiple objects per scene.

  2. **json** (legacy GraspGen format) — single-object-per-file JSON
     with ``object_info`` and ``scene_info`` keys.

The format is auto-detected from the directory layout.
"""

import argparse
import os

import numpy as np
import torch
import trimesh
import trimesh.transformations as tra

from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from grasp_gen.robot import get_gripper_info
from grasp_gen.samplers import run_graspmoe
from grasp_gen.utils.point_cloud_utils import (
    filter_colliding_grasps_fast,
    point_cloud_outlier_removal_with_color,
)
from grasp_gen.utils.scene_loaders import (
    build_scene_pc_excluding_object,
    collect_scene_items,
    load_scene,
)
from grasp_gen.utils.viser_utils import (
    create_visualizer,
    get_color_from_score,
    visualize_grasp,
    visualize_mesh,
    visualize_pointcloud,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample_data_dir",
        type=str,
        required=True,
        help="Either a directory of <NN>/{depth.npy,...} subdirs (realworld) "
        "or a directory of *.json files (json).",
    )
    parser.add_argument("--gripper_config", type=str, required=True)
    parser.add_argument("--grasp_threshold", type=float, default=0.80)
    parser.add_argument("--num_grasps", type=int, default=200)
    parser.add_argument("--return_topk", action="store_true")
    parser.add_argument(
        "--topk_num_grasps",
        type=int,
        default=-1,
        help="Top-k cap on the union of diffusion + OBB grasps.",
    )

    # --- collision filtering ---
    parser.add_argument(
        "--filter_collisions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Drop grasps whose gripper mesh would intersect the scene PC "
        "(target object's own pixels are excluded). Uses the GPU-vectorized "
        "filter (filter_colliding_grasps_fast).",
    )
    parser.add_argument(
        "--collision_threshold",
        type=float,
        default=0.02,
        help="Distance (meters) under which a gripper surface sample counts "
        "as colliding with the scene PC.",
    )
    parser.add_argument(
        "--max_scene_points",
        type=int,
        default=8192,
        help="Random-downsample the scene PC to at most this many points "
        "before collision check.",
    )
    parser.add_argument(
        "--num_collision_samples",
        type=int,
        default=2000,
        help="Surface samples drawn from the gripper collision mesh.",
    )

    # --- realworld format ---
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Realworld format only: restrict to a single scene <NN> (e.g. '00').",
    )
    parser.add_argument(
        "--min_obj_points",
        type=int,
        default=100,
        help="Realworld format only: skip objects whose segmented PC has fewer points.",
    )

    # --- GraspMoE / OBB planner flags ---
    parser.add_argument(
        "--planner",
        type=str,
        default="graspmoe",
        choices=["diffusion", "graspmoe"],
        help="Inference planner. The OBB is computed on the *object* PC, "
        "not the full scene.",
    )
    parser.add_argument("--moe_num_yaws", type=int, default=36)
    parser.add_argument("--moe_z_offsets_cm", type=str, default="-8,-6,-4,-2,0")
    parser.add_argument("--moe_outlier_threshold", type=float, default=0.014)
    parser.add_argument("--moe_outlier_k", type=int, default=20)
    parser.add_argument(
        "--moe_obb_mode", type=str, default="advanced", choices=["advanced", "pca"]
    )
    parser.add_argument(
        "--moe_skip_obb_rule",
        type=str,
        default="auto",
        choices=["auto", "never"],
    )
    parser.add_argument(
        "--moe_obb_density",
        type=str,
        default="dense-topandside",
        choices=["sparse", "dense", "dense-topandside"],
    )
    parser.add_argument("--moe_obb_position_spacing_cm", type=float, default=1.0)

    parser.add_argument(
        "--no-visualization",
        dest="no_visualization",
        action="store_true",
        help="Skip viser visualization (headless).",
    )
    return parser.parse_args()


def _visualize_grasps_for_object(
    vis,
    label: str,
    grasps_world: np.ndarray,
    scores: np.ndarray,
    branch_tags: list,
    gripper_name: str,
    filter_was_applied: bool,
):
    """Render grasps under `obj/<label>/grasps/...`."""
    if vis is None or len(grasps_world) == 0:
        return
    score_colors = get_color_from_score(scores, use_255_scale=True)
    ns_prefix = f"obj/{label}"
    for j, grasp in enumerate(grasps_world):
        tag = branch_tags[j] if j < len(branch_tags) else "diff"
        # When collision filter was applied, every grasp shown is collision-free
        # — color by branch (green for diff, blue for obb). Otherwise color by score.
        if filter_was_applied:
            color = [0, 185, 0] if tag == "diff" else [60, 130, 200]
        else:
            color = score_colors[j]
        visualize_grasp(
            vis,
            f"{ns_prefix}/grasps/{tag}/{j:03d}",
            grasp,
            color=color,
            gripper_name=gripper_name,
            linewidth=1.5,
        )


def _run_planner_on_object(args, obj_pc, grasp_sampler):
    """Run --planner on a single segmented object PC. Returns
    (grasps_world, scores, branch_tags)."""
    if args.planner == "graspmoe":
        moe = run_graspmoe(
            obj_pc,
            grasp_sampler,
            grasp_threshold=args.grasp_threshold,
            num_grasps=args.num_grasps,
            topk_num_grasps=args.topk_num_grasps,
            num_yaws=args.moe_num_yaws,
            z_offsets_cm=tuple(
                float(z) for z in args.moe_z_offsets_cm.split(",") if z.strip()
            ),
            outlier_threshold=args.moe_outlier_threshold,
            outlier_k=args.moe_outlier_k,
            obb_mode=args.moe_obb_mode,
            skip_obb_rule=args.moe_skip_obb_rule,
            obb_density=args.moe_obb_density,
            obb_position_spacing_m=args.moe_obb_position_spacing_cm / 100.0,
        )
        grasps = np.concatenate(
            [moe["grasps_diff"], moe["grasps_obb"]], axis=0
        ).astype(np.float32)
        scores = np.concatenate(
            [moe["scores_diff"], moe["scores_obb"]], axis=0
        ).astype(np.float32)
        tags = ["diff"] * len(moe["grasps_diff"]) + ["obb"] * len(moe["grasps_obb"])
        print(
            f"    [graspmoe] diffusion={len(moe['grasps_diff'])}, "
            f"OBB={len(moe['grasps_obb'])}, skipped_obb={moe['skipped_obb']}"
        )
    else:
        gt, ct = GraspGenSampler.run_inference(
            obj_pc,
            grasp_sampler,
            grasp_threshold=args.grasp_threshold,
            num_grasps=args.num_grasps,
            topk_num_grasps=args.topk_num_grasps,
        )
        grasps = (
            gt.cpu().numpy().astype(np.float32)
            if len(gt) > 0
            else np.zeros((0, 4, 4), dtype=np.float32)
        )
        scores = (
            ct.cpu().numpy().astype(np.float32)
            if len(ct) > 0
            else np.zeros((0,), dtype=np.float32)
        )
        tags = ["diff"] * len(grasps)
    if len(grasps) > 0:
        grasps[:, 3, 3] = 1.0
    return grasps, scores, tags


def main():
    args = parse_args()

    if not os.path.exists(args.sample_data_dir):
        raise FileNotFoundError(
            f"sample_data_dir {args.sample_data_dir} does not exist"
        )
    if args.return_topk and args.topk_num_grasps == -1:
        args.topk_num_grasps = 100

    items = collect_scene_items(args.sample_data_dir, scene_filter=args.scene)
    fmt = items[0][0] if items else "json"
    print(f"[scene_pc] format={fmt}, {len(items)} scene(s) to process")

    grasp_cfg = load_grasp_cfg(args.gripper_config)
    gripper_name = grasp_cfg.data.gripper_name
    print(f"[scene_pc] gripper={gripper_name}, planner={args.planner}")
    grasp_sampler = GraspGenSampler(grasp_cfg)

    # Load the gripper collision mesh ONCE and pre-sample surface points so the
    # collision filter doesn't re-sample for every grasp.
    gripper_surface_points = None
    gripper_collision_mesh = None
    if args.filter_collisions:
        gripper_info = get_gripper_info(gripper_name)
        gripper_collision_mesh = gripper_info.collision_mesh
        sampled, _ = trimesh.sample.sample_surface(
            gripper_collision_mesh, args.num_collision_samples
        )
        gripper_surface_points = np.asarray(sampled, dtype=np.float32)
        print(
            f"[scene_pc] pre-sampled {len(gripper_surface_points)} gripper "
            f"surface points for collision filtering"
        )

    vis = None if args.no_visualization else create_visualizer()

    for fmt_tag, path in items:
        print(f"\n========== {os.path.basename(path)} ==========")
        if vis is not None:
            vis.scene.reset()

        scene = load_scene(fmt_tag, path, min_obj_points=args.min_obj_points)

        # Render full scene (clip to a reasonable working volume for viz).
        VIZ_BOUNDS = [[-1.5, -1.25, -0.15], [1.5, 1.25, 2.0]]
        xyz_scene = scene["scene_xyz"]
        rgb_scene = scene["scene_rgb"]
        m = np.all(
            (xyz_scene > VIZ_BOUNDS[0]) & (xyz_scene < VIZ_BOUNDS[1]), axis=1
        )
        xyz_scene_viz = xyz_scene[m]
        rgb_scene_viz = rgb_scene[m]
        if vis is not None:
            visualize_pointcloud(
                vis, "pc_scene", xyz_scene_viz, rgb_scene_viz, size=0.0025
            )

        if not scene["objects"]:
            print("  No segmented objects found; skipping.")
            if vis is not None:
                input("Press Enter to continue to next scene...")
            continue

        for label, obj in scene["objects"].items():
            obj_pc_raw = np.asarray(obj["pc"], dtype=np.float32)
            obj_rgb = np.asarray(obj["rgb"], dtype=np.uint8)
            if len(obj_pc_raw) < args.min_obj_points:
                print(f"  [{label}] too few points ({len(obj_pc_raw)}); skipping.")
                continue
            print(f"\n  --- {label} ({len(obj_pc_raw)} pts) ---")

            # Outlier removal on the object PC (matches old behavior).
            obj_pc_t, _, obj_rgb_t, _ = point_cloud_outlier_removal_with_color(
                torch.from_numpy(obj_pc_raw), torch.from_numpy(obj_rgb)
            )
            obj_pc = obj_pc_t.cpu().numpy().astype(np.float32)
            obj_rgb_filt = obj_rgb_t.cpu().numpy().astype(np.uint8)

            if vis is not None:
                visualize_pointcloud(
                    vis, f"obj/{label}/pc", obj_pc, obj_rgb_filt, size=0.005
                )

            grasps_world, scores, tags = _run_planner_on_object(
                args, obj_pc, grasp_sampler
            )
            if len(grasps_world) == 0:
                print(f"  [{label}] no grasps generated; skipping viz")
                continue

            # Collision filtering against the scene with this object's points removed.
            filter_applied = False
            if args.filter_collisions:
                scene_pc_no_target = build_scene_pc_excluding_object(scene, label)
                if len(scene_pc_no_target) > args.max_scene_points:
                    idx = np.random.choice(
                        len(scene_pc_no_target),
                        args.max_scene_points,
                        replace=False,
                    )
                    scene_pc_no_target = scene_pc_no_target[idx]

                cf_mask = filter_colliding_grasps_fast(
                    scene_pc=scene_pc_no_target,
                    grasp_poses=grasps_world,
                    collision_threshold=args.collision_threshold,
                    gripper_surface_points=gripper_surface_points,
                )
                n_before = len(grasps_world)
                grasps_world = grasps_world[cf_mask]
                scores = scores[cf_mask]
                tags = [t for t, keep in zip(tags, cf_mask) if keep]
                filter_applied = True
                print(
                    f"  [{label}] collision filter: {len(grasps_world)} free / "
                    f"{n_before - len(grasps_world)} colliding "
                    f"(thr={args.collision_threshold:.3f}m, "
                    f"scene_pts={len(scene_pc_no_target)})"
                )
                if len(grasps_world) == 0:
                    print(f"  [{label}] all grasps colliding; nothing to viz")
                    continue

            print(
                f"  [{label}] {len(grasps_world)} grasps "
                f"(diff={tags.count('diff')}, obb={tags.count('obb')}); "
                f"score range {scores.min():.3f}..{scores.max():.3f}"
            )
            _visualize_grasps_for_object(
                vis,
                label,
                grasps_world,
                scores,
                tags,
                gripper_name,
                filter_applied,
            )

        if vis is not None:
            input("\nPress Enter to continue to next scene...")


if __name__ == "__main__":
    main()
