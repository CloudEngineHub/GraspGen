import numpy as np
import torch
import trimesh

from grasp_gen.utils.meshcat_utils import (
    create_visualizer,
    make_frame,
    visualize_grasp,
    visualize_mesh,
    visualize_pointcloud,
)

MAPPING_ID2NAME = {
    0: "pos_true",
    1: "neg_true",
    2: "neg_hncolliding",
    3: "neg_freespace",
    4: "neg_hnretract",
    5: "pos_true_onpolicy",
    6: "neg_true_onpolicy",
}

MAPPING_NAME2ID = {val: key for key, val in MAPPING_ID2NAME.items()}


def visualize_generator_dataset(
    obj_pose,
    grasps,
    gripper_name,
    load_contact,
    contacts,
    pointcloud,
    gripper_visual_mesh=None,
    obj_mesh=None,
):

    vis = create_visualizer()
    vis.delete()

    if obj_mesh is not None:
        visualize_mesh(
            vis, "object_mesh", obj_mesh, color=[192, 192, 192], transform=obj_pose
        )

    for i, g in enumerate(grasps):
        if torch.is_tensor(g):
            g = g.cpu().numpy()

        if i < 20:
            visualize_grasp(
                vis,
                f"grasps/{i:03d}",
                g.astype(np.float32),
                [0, 255, 0],
                gripper_name=gripper_name,
                linewidth=0.2,
            )

    if pointcloud is not None:
        visualize_pointcloud(
            vis, f"point_cloud", pointcloud, [10, 10, 250], size=0.0025
        )

    if load_contact:

        if len(contacts.shape) == 3:
            contacts[:, 0, :] = tra.transform_points(contacts[:, 0, :], obj_pose)
            contacts[:, 1, :] = tra.transform_points(contacts[:, 1, :], obj_pose)
        else:
            contacts = tra.transform_points(contacts, obj_pose)

        visualize_pointcloud(vis, f"contact_points", contacts, [0, 0, 255], size=0.007)
    input()


def visualize_discriminator_dataset(
    batch_data: dict,
    scene_mesh: trimesh.base.Trimesh,
    gripper_name: str,
    gripper_mesh: trimesh.base.Trimesh = None,
    pointcloud: torch.Tensor = None,
):

    grasps = batch_data["grasps"]
    grasp_ids = batch_data["grasp_ids"].squeeze(1)

    pos_grasps = grasps[grasp_ids == MAPPING_NAME2ID["pos_true"]].cpu().numpy()
    neg_grasps_tn = grasps[grasp_ids == MAPPING_NAME2ID["neg_true"]].cpu().numpy()
    neg_grasps_hn = (
        grasps[grasp_ids == MAPPING_NAME2ID["neg_hncolliding"]].cpu().numpy()
    )
    neg_grasps_fs = grasps[grasp_ids == MAPPING_NAME2ID["neg_freespace"]].cpu().numpy()
    neg_grasps_retract = (
        grasps[grasp_ids == MAPPING_NAME2ID["neg_hnretract"]].cpu().numpy()
    )

    pos_grasps_onpolicy = (
        grasps[grasp_ids == MAPPING_NAME2ID["pos_true_onpolicy"]].cpu().numpy()
    )
    neg_grasps_onpolicy = (
        grasps[grasp_ids == MAPPING_NAME2ID["neg_true_onpolicy"]].cpu().numpy()
    )

    vis = create_visualizer()
    vis.delete()

    visualize_mesh(vis, "scene_mesh", scene_mesh, color=[192, 192, 192])

    max_grasps_to_visualize = 15

    for j, grasp in enumerate(pos_grasps):
        # visualize_mesh(vis, f"gt/pos_mesh/all/grasp_{j:03d}", gripper_mesh, color=[0, 255, 0], transform=grasp.astype(np.float32))
        visualize_grasp(
            vis,
            f"gt-offline/pos/pos_true/grasp_{j:03d}",
            grasp,
            [0, 255, 0],
            gripper_name=gripper_name,
            linewidth=0.2,
        )

        if j > max_grasps_to_visualize:
            break

    for j, grasp in enumerate(neg_grasps_tn):
        visualize_grasp(
            vis,
            f"gt-offline/neg/neg_true/grasp_{j:03d}",
            grasp,
            [255, 0, 0],
            gripper_name=gripper_name,
            linewidth=0.2,
        )

        if j > max_grasps_to_visualize:
            break

    for j, grasp in enumerate(neg_grasps_hn):
        visualize_grasp(
            vis,
            f"gt-offline/neg/neg_hncolliding/grasp_{j:03d}",
            grasp,
            [255, 0, 0],
            gripper_name=gripper_name,
            linewidth=0.2,
        )

        if j > max_grasps_to_visualize:
            break

    for j, grasp in enumerate(neg_grasps_fs):
        visualize_grasp(
            vis,
            f"gt-offline/neg/neg_freespace/grasp_{j:03d}",
            grasp,
            [255, 0, 0],
            gripper_name=gripper_name,
            linewidth=0.2,
        )

        if j > max_grasps_to_visualize:
            break

    for j, grasp in enumerate(neg_grasps_retract):
        visualize_grasp(
            vis,
            f"gt-offline/neg/neg_hnretract/grasp_{j:03d}",
            grasp,
            [255, 0, 0],
            gripper_name=gripper_name,
            linewidth=0.2,
        )

        if j > max_grasps_to_visualize:
            break

    for j, grasp in enumerate(pos_grasps_onpolicy):
        # visualize_mesh(vis, f"gt/pos_mesh/all/grasp_{j:03d}", gripper_mesh, color=[0, 255, 0], transform=grasp.astype(np.float32))
        visualize_grasp(
            vis,
            f"gt-ongenerator/pos/grasp_{j:03d}",
            grasp,
            [0, 255, 0],
            gripper_name=gripper_name,
            linewidth=0.2,
        )

        if j > max_grasps_to_visualize:
            break

    for j, grasp in enumerate(neg_grasps_onpolicy):
        visualize_grasp(
            vis,
            f"gt-ongenerator/neg/grasp_{j:03d}",
            grasp,
            [255, 0, 0],
            gripper_name=gripper_name,
            linewidth=0.2,
        )

        if j > max_grasps_to_visualize:
            break

    if pointcloud is not None:
        if type(pointcloud) == torch.Tensor:
            pointcloud = pointcloud.cpu().numpy()
        visualize_pointcloud(
            vis, f"point_cloud", pointcloud, [10, 10, 250], size=0.0025
        )

    input()
