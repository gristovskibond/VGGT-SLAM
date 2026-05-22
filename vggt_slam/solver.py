import os
import time

import numpy as np
import open3d as o3d
import torch
from scipy.spatial.transform import Rotation as R

from vggt_omega.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

from vggt_slam.frame_overlap import FrameTracker
from vggt_slam.map import GraphMap
from vggt_slam.slam_utils import Accumulator, compute_image_embeddings
from vggt_slam.submap import Submap
from vggt_slam.viewer import Viewer


class Solver:
    """Single-shot VGGT-Omega reconstruction (no submaps or loop closure)."""

    def __init__(
        self,
        init_conf_threshold: float,
        vis_voxel_size: float | None = None,
    ):
        self.init_conf_threshold = init_conf_threshold
        self.vis_voxel_size = vis_voxel_size

        self.viewer = Viewer()
        self.flow_tracker = FrameTracker()
        self.map = GraphMap()
        self.submap: Submap | None = None
        self.cam_to_world: np.ndarray | None = None
        self.extrinsic: np.ndarray | None = None

        self.vggt_timer = Accumulator()
        self.clip_timer = Accumulator()

    def set_point_cloud(
        self,
        points_in_world_frame: np.ndarray,
        points_colors: np.ndarray,
        name: str,
        point_size: float,
    ) -> None:
        if self.vis_voxel_size is not None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_in_world_frame.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(points_colors.astype(np.float64) / 255.0)
            pcd = pcd.voxel_down_sample(self.vis_voxel_size)
            points_in_world_frame = np.asarray(pcd.points, dtype=np.float32)
            points_colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
        self.viewer.server.scene.add_point_cloud(
            name="pcd_" + name,
            points=points_in_world_frame,
            colors=points_colors,
            point_size=point_size,
            point_shape="circle",
        )

    def run_reconstruction(
        self,
        image_names: list[str],
        model: torch.nn.Module,
        clip_model,
        clip_preprocess,
        image_resolution: int = 512,
    ) -> dict:
        """Run VGGT-Omega on all frames in one forward pass."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        t0 = time.time()
        with self.vggt_timer:
            images = load_and_preprocess_images(
                image_names, image_resolution=image_resolution
            ).to(device)
        print(
            f"Loaded and preprocessed {len(image_names)} images in {time.time() - t0:.2f}s"
        )
        print(f"Preprocessed images shape: {images.shape}")

        with torch.no_grad():
            t0 = time.time()
            with self.vggt_timer:
                raw = model(images)
            print(f"VGGT-Omega inference took {time.time() - t0:.2f}s")

        print("Converting pose encoding to extrinsic and intrinsic matrices...")
        extrinsic, intrinsic = encoding_to_camera(raw["pose_enc"], images.shape[-2:])

        predictions: dict = {
            "images": images,
            "depth": raw["depth"],
            "depth_conf": raw["depth_conf"],
            "extrinsic": extrinsic,
            "intrinsic": intrinsic,
            "image_names": image_names,
        }

        for key, value in list(predictions.items()):
            if isinstance(value, torch.Tensor):
                arr = value.float().cpu().numpy()
                if arr.shape[0] == 1:
                    arr = arr[0]
                predictions[key] = arr

        if clip_model is not None and clip_preprocess is not None:
            with self.clip_timer:
                predictions["semantic_vectors"] = compute_image_embeddings(
                    clip_model, clip_preprocess, image_names
                )

        return predictions

    def build_map(self, pred_dict: dict) -> None:
        """Store single-shot predictions as one submap (model world frame, no pose graph)."""
        images = pred_dict["images"]
        extrinsics_cam = pred_dict["extrinsic"]
        intrinsics_cam = pred_dict["intrinsic"]
        depth_map = pred_dict["depth"]
        conf = pred_dict["depth_conf"]
        image_names = pred_dict["image_names"]

        world_points = unproject_depth_map_to_point_map(
            depth_map, extrinsics_cam, intrinsics_cam
        )
        colors = (images.transpose(0, 2, 3, 1) * 255).astype(np.uint8)
        cam_to_world = closed_form_inverse_se3(extrinsics_cam)
        world_to_cam = np.linalg.inv(cam_to_world)

        n_frames = cam_to_world.shape[0]
        k_4x4 = np.tile(np.eye(4), (n_frames, 1, 1))
        k_4x4[:, :3, :3] = intrinsics_cam

        submap = Submap(0)
        frame_tensor = (
            torch.from_numpy(images).float()
            if isinstance(images, np.ndarray)
            else images.float()
        )
        submap.add_all_frames(frame_tensor)
        submap.set_frame_ids(image_names)
        submap.set_img_names(image_names)
        submap.set_last_non_loop_frame_index(n_frames - 1)
        submap.add_all_poses(world_to_cam)
        submap.add_all_points(
            world_points, colors, conf, self.init_conf_threshold, k_4x4
        )
        submap.set_conf_masks(conf)

        if "semantic_vectors" in pred_dict:
            submap.set_all_semantic_vectors(pred_dict["semantic_vectors"])

        self.map.add_submap(submap)
        self.submap = submap
        self.cam_to_world = cam_to_world
        self.extrinsic = extrinsics_cam

    def get_merged_point_cloud(self) -> tuple[np.ndarray, np.ndarray]:
        assert self.submap is not None
        points_all = []
        colors_all = []
        for index in range(len(self.submap.pointclouds)):
            points = self.submap.pointclouds[index]
            mask = self.submap.conf_masks[index] > self.submap.conf_threshold
            points_all.append(points[mask])
            colors_all.append(self.submap.colors[index][mask])
        return np.vstack(points_all), np.vstack(colors_all)

    def visualize_map(self) -> None:
        assert self.submap is not None and self.cam_to_world is not None
        points, colors = self.get_merged_point_cloud()
        self.set_point_cloud(points, colors, "0", 0.001)
        self.viewer.visualize_frames(
            self.cam_to_world[:, :3, :4],
            self.submap.get_all_frames(),
            self.submap.get_id(),
        )

    def write_poses_to_file(self, file_name: str) -> None:
        assert self.submap is not None and self.cam_to_world is not None
        os.makedirs(os.path.dirname(file_name) or ".", exist_ok=True)
        with open(file_name, "w", encoding="utf-8") as f:
            for frame_index, frame_id in enumerate(self.submap.get_frame_ids()):
                c2w = self.cam_to_world[frame_index]
                translation = c2w[:3, 3]
                quaternion = R.from_matrix(c2w[:3, :3]).as_quat()
                row = np.array([float(frame_id), *translation, *quaternion])
                f.write(" ".join(f"{v:.8f}" for v in row) + "\n")

    def write_points_to_file(self, file_name: str) -> None:
        points, colors = self.get_merged_point_cloud()
        if colors.max() > 1.0:
            colors = colors / 255.0
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float64)))
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        o3d.io.write_point_cloud(file_name, pcd)

    def save_framewise_pointclouds(self, directory: str) -> None:
        assert self.submap is not None
        os.makedirs(directory, exist_ok=True)
        for frame_index, frame_id in enumerate(self.submap.get_frame_ids()):
            points = self.submap.pointclouds[frame_index]
            mask = self.submap.conf_masks[frame_index] > self.submap.conf_threshold
            np.savez(
                f"{directory}/{frame_id}.npz",
                pointcloud=points[mask],
                mask=mask,
            )

    def get_points_in_mask(self, frame_index: int, mask: np.ndarray) -> np.ndarray:
        """World-frame points for semantic masking (no pose-graph transform)."""
        assert self.submap is not None
        points = self.submap.pointclouds[frame_index]
        points_flat = points.reshape(-1, 3)
        mask_flat = mask.reshape(-1).astype(bool)
        conf_mask = (
            self.submap.conf_masks[frame_index].reshape(-1) > self.submap.conf_threshold
        )
        return points_flat[mask_flat & conf_mask]
