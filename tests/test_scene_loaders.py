#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Tests for scene_loaders + the fast collision filter.

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from grasp_gen.utils.point_cloud_utils import filter_colliding_grasps_fast
from grasp_gen.utils.scene_loaders import (
    build_scene_pc_excluding_object,
    collect_scene_items,
    depth_to_camera_xyz,
    detect_format,
    load_graspgen_json_scene,
    load_realworld_scene,
    load_scene,
    transform_xyz,
)


# ─── format detection ──────────────────────────────────────────────────────


def test_detect_format_realworld(tmp_path):
    sub = tmp_path / "00"
    sub.mkdir()
    (sub / "meta_data.json").write_text("{}")
    assert detect_format(str(tmp_path)) == "realworld"


def test_detect_format_json(tmp_path):
    (tmp_path / "scene_a.json").write_text("{}")
    assert detect_format(str(tmp_path)) == "json"


def test_detect_format_neither(tmp_path):
    with pytest.raises(FileNotFoundError):
        detect_format(str(tmp_path))


def test_collect_scene_items_realworld(tmp_path):
    for nm in ["00", "01", "02"]:
        (tmp_path / nm).mkdir()
        (tmp_path / nm / "meta_data.json").write_text("{}")
    items = collect_scene_items(str(tmp_path))
    assert len(items) == 3
    assert all(f == "realworld" for f, _ in items)
    assert sorted(os.path.basename(p) for _, p in items) == ["00", "01", "02"]


def test_collect_scene_items_filter(tmp_path):
    for nm in ["00", "01"]:
        (tmp_path / nm).mkdir()
        (tmp_path / nm / "meta_data.json").write_text("{}")
    items = collect_scene_items(str(tmp_path), scene_filter="01")
    assert len(items) == 1
    assert os.path.basename(items[0][1]) == "01"


# ─── geometry helpers ──────────────────────────────────────────────────────


def test_depth_to_camera_xyz_identity():
    K = np.array([[100.0, 0, 64], [0, 100.0, 48], [0, 0, 1]])
    depth = np.ones((96, 128), dtype=np.float32)
    xyz = depth_to_camera_xyz(depth, K)
    assert xyz.shape == (96, 128, 3)
    # Center pixel: u=cx, v=cy → x=y=0, z=1
    assert np.allclose(xyz[48, 64], [0.0, 0.0, 1.0], atol=1e-5)


def test_transform_xyz_identity():
    pts = np.random.RandomState(0).randn(10, 3).astype(np.float32)
    out = transform_xyz(pts, np.eye(4))
    assert np.allclose(out, pts, atol=1e-6)


def test_transform_xyz_translation():
    pts = np.zeros((4, 3))
    T = np.eye(4)
    T[:3, 3] = [1.0, 2.0, 3.0]
    out = transform_xyz(pts, T)
    assert np.allclose(out, [[1, 2, 3]] * 4)


# ─── realworld scene loader ────────────────────────────────────────────────


def _make_realworld_scene(tmp_path, scene_name="00"):
    """Create a minimal fake realworld scene: 64x48 depth at z=0.5 with two
    objects (label_ids 1 and 2) in the top-left and bottom-right quadrants."""
    sub = tmp_path / scene_name
    sub.mkdir()

    H, W = 48, 64
    depth = np.full((H, W), 0.5, dtype=np.float32)
    rgb = (np.random.RandomState(0).rand(H, W, 3) * 255).astype(np.uint8)
    seg = np.zeros((H, W), dtype=np.int32)
    seg[:24, :32] = 1  # obj_1
    seg[24:, 32:] = 2  # obj_2
    K = np.array([[200.0, 0, 32], [0, 200.0, 24], [0, 0, 1]])
    cam_pose = np.eye(4)

    np.save(sub / "depth.npy", depth)
    Image.fromarray(rgb).save(sub / "rgb.png")
    Image.fromarray(seg.astype(np.uint8), mode="L").save(sub / "seg.png")
    meta = {
        "intrinsics": K.tolist(),
        "camera_pose": cam_pose.tolist(),
        "label_map": {"obj_1": 1, "obj_2": 2, "background": 0},
    }
    with open(sub / "meta_data.json", "w") as f:
        json.dump(meta, f)
    return str(sub)


def test_load_realworld_scene_basic(tmp_path):
    sub = _make_realworld_scene(tmp_path)
    scene = load_realworld_scene(sub)
    assert scene["_format"] == "realworld"
    assert scene["scene_xyz"].shape[1] == 3
    assert "obj_1" in scene["objects"]
    assert "obj_2" in scene["objects"]
    # Each object should have ~quadrant points (24*32 = 768, less invalid).
    for nm in ("obj_1", "obj_2"):
        pc = scene["objects"][nm]["pc"]
        assert pc.shape[1] == 3
        assert len(pc) > 500  # ~quadrant


def test_load_realworld_scene_denylist(tmp_path):
    sub = _make_realworld_scene(tmp_path, "denied")
    # Patch meta to deny obj_2
    meta_path = os.path.join(sub, "meta_data.json")
    with open(meta_path) as f:
        meta = json.load(f)
    meta["obj_denylist"] = ["obj_2"]
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    scene = load_realworld_scene(sub)
    assert "obj_1" in scene["objects"]
    assert "obj_2" not in scene["objects"]


def test_build_scene_pc_excluding_object_realworld(tmp_path):
    sub = _make_realworld_scene(tmp_path)
    scene = load_realworld_scene(sub)
    n_total = len(scene["scene_xyz"])
    excluded = build_scene_pc_excluding_object(scene, "obj_1")
    # Excluding one object should leave fewer points than the full scene.
    assert len(excluded) < n_total
    # Object 1 had ~768 points, so excluded should still have ~rest.
    assert len(excluded) > 0


def test_build_scene_pc_excluding_unknown_object(tmp_path):
    sub = _make_realworld_scene(tmp_path)
    scene = load_realworld_scene(sub)
    # Unknown label → returns full scene.
    out = build_scene_pc_excluding_object(scene, "obj_999")
    assert len(out) == len(scene["scene_xyz"])


# ─── json scene loader ─────────────────────────────────────────────────────


def test_load_graspgen_json_scene(tmp_path):
    """Synthetic GraspGen-format JSON: object_info + scene_info."""
    obj_pc = np.random.RandomState(0).rand(50, 3).astype(np.float32).tolist()
    obj_color = np.zeros((50, 3), dtype=np.uint8).tolist()
    scene_pc = np.random.RandomState(1).rand(500, 3).astype(np.float32).tolist()
    scene_color = (np.ones((500, 3)) * 100).astype(np.uint8).tolist()
    obj_mask = ([1] * 50 + [0] * 450)  # first 50 belong to the object

    data = {
        "object_info": {"pc": obj_pc, "pc_color": obj_color},
        "scene_info": {
            "pc_color": [scene_pc],  # GraspGenX-key compatibility path
            "img_color": scene_color,
            "obj_mask": obj_mask,
        },
        "grasp_info": {"grasp_poses": [], "grasp_conf": []},
    }
    p = tmp_path / "scene.json"
    p.write_text(json.dumps(data))
    scene = load_graspgen_json_scene(str(p))
    assert scene["_format"] == "json"
    assert "obj_0" in scene["objects"]
    # obj_mask removes 50 → 450 left.
    assert len(scene["scene_xyz"]) == 450
    assert len(scene["objects"]["obj_0"]["pc"]) == 50


def test_load_scene_dispatch(tmp_path):
    sub = _make_realworld_scene(tmp_path)
    scene = load_scene("realworld", sub)
    assert scene["_format"] == "realworld"
    with pytest.raises(ValueError):
        load_scene("unknown", sub)


# ─── fast collision filter ─────────────────────────────────────────────────


def test_filter_colliding_grasps_fast_empty_inputs():
    out = filter_colliding_grasps_fast(
        scene_pc=np.zeros((0, 3)),
        grasp_poses=np.tile(np.eye(4), (5, 1, 1)),
        gripper_surface_points=np.zeros((10, 3), dtype=np.float32),
        collision_threshold=0.02,
    )
    # Empty scene → everything collision-free.
    assert out.shape == (5,)
    assert out.all()


def test_filter_colliding_grasps_fast_no_grasps():
    out = filter_colliding_grasps_fast(
        scene_pc=np.random.RandomState(0).rand(100, 3).astype(np.float32),
        grasp_poses=np.zeros((0, 4, 4)),
        gripper_surface_points=np.zeros((10, 3), dtype=np.float32),
    )
    assert out.shape == (0,)


def test_filter_colliding_grasps_fast_far_grasps_pass():
    """Grasps far from the scene PC should all be collision-free."""
    scene_pc = np.array(
        [[0.0, 0.0, 0.0]], dtype=np.float32
    )  # single point at origin
    # Grasps at z=5 m, far from origin.
    grasps = np.tile(np.eye(4), (4, 1, 1)).astype(np.float32)
    grasps[:, 2, 3] = 5.0
    # Gripper sample points near gripper origin.
    pts_local = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]], dtype=np.float32)
    out = filter_colliding_grasps_fast(
        scene_pc=scene_pc,
        grasp_poses=grasps,
        gripper_surface_points=pts_local,
        collision_threshold=0.02,
    )
    assert out.all()


def test_filter_colliding_grasps_fast_close_grasps_collide():
    """Grasps colocated with a scene point should be marked as colliding."""
    scene_pc = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    grasps = np.tile(np.eye(4), (4, 1, 1)).astype(np.float32)
    # All grasps at the same location as the scene point.
    pts_local = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    out = filter_colliding_grasps_fast(
        scene_pc=scene_pc,
        grasp_poses=grasps,
        gripper_surface_points=pts_local,
        collision_threshold=0.02,
    )
    assert not out.any()


def test_filter_colliding_grasps_fast_requires_mesh_or_pts():
    with pytest.raises(ValueError, match="gripper_collision_mesh"):
        filter_colliding_grasps_fast(
            scene_pc=np.random.rand(10, 3).astype(np.float32),
            grasp_poses=np.tile(np.eye(4), (2, 1, 1)),
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
