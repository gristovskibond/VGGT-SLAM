import os
import glob
import time
import argparse

import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot
from torchvision.transforms.functional import to_pil_image
from tqdm.auto import tqdm
import cv2
import open3d as o3d

import vggt_slam.slam_utils as utils
from vggt_slam.solver import Solver

from vggt_omega.models import VGGTOmega

from huggingface_hub import login

if not os.getenv("SAM_API_KEY"):
    print("ERROR: SAM_API_KEY environment variable is not set.")
    print("Please set SAM_API_KEY to use the SAM model.")

sam_api_key = os.getenv("SAM_API_KEY")
if sam_api_key:
    login(sam_api_key)

VGGT_OMEGA_CHECKPOINT = "/app/VGGT-SLAM/third_party/vggt-omega/vggt_omega_model/vggt_omega_1b_512.pt"

parser = argparse.ArgumentParser(
    description="VGGT-SLAM: single-shot VGGT-Omega reconstruction"
)
parser.add_argument(
    "--image_folder",
    type=str,
    default="examples/kitchen/images/",
    help="Path to folder containing images",
)
parser.add_argument(
    "--vis_map",
    action="store_true",
    help="Visualize point cloud in viser after reconstruction",
)
parser.add_argument(
    "--vis_voxel_size",
    type=float,
    default=None,
    help="Voxel size for downsampling the point cloud in the viewer (e.g. 0.05 for 5 cm). Default: no downsampling",
)
parser.add_argument(
    "--run_os",
    action="store_true",
    help="Enable open-set semantic search with Perception Encoder CLIP and SAM3",
)
parser.add_argument(
    "--vis_flow",
    action="store_true",
    help="Visualize optical flow for keyframe selection",
)
parser.add_argument(
    "--log_results",
    action="store_true",
    help="save txt file with results",
)
parser.add_argument(
    "--skip_dense_log",
    action="store_true",
    help="by default, logging poses and logs dense point clouds. If this flag is set, dense logging is skipped",
)
parser.add_argument(
    "--log_path",
    type=str,
    default="poses.txt",
    help="Path to save the log file",
)
parser.add_argument(
    "--min_disparity",
    type=float,
    default=50,
    help="Minimum optical-flow disparity to keep a keyframe (0 = use every frame)",
)
parser.add_argument(
    "--conf_threshold",
    type=float,
    default=25.0,
    help="Percentage of lowest-confidence depth points to filter out",
)
parser.add_argument(
    "--image_resolution",
    type=int,
    default=512,
    help="VGGT-Omega input resolution (must be divisible by 16). Default: 512.",
)
parser.add_argument(
    "--use_all_frames",
    action="store_true",
    help="Skip optical-flow keyframe selection and run on every image in the folder",
)


def collect_image_names(args, solver: Solver) -> list[str]:
    image_names = [
        f
        for f in glob.glob(os.path.join(args.image_folder, "*"))
        if "depth" not in os.path.basename(f).lower()
        and "txt" not in os.path.basename(f).lower()
        and "db" not in os.path.basename(f).lower()
    ]
    image_names = utils.sort_images_by_number(image_names)
    image_names = utils.downsample_images(image_names, 1)

    if args.use_all_frames or args.min_disparity <= 0:
        return image_names

    selected: list[str] = []
    for image_name in tqdm(image_names, desc="Keyframe selection"):
        img = cv2.imread(image_name)
        if solver.flow_tracker.compute_disparity(
            img, args.min_disparity, args.vis_flow
        ):
            selected.append(image_name)
    return selected


def main():
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if args.image_resolution % 16 != 0:
        parser.error("--image_resolution must be divisible by 16")

    solver = Solver(
        init_conf_threshold=args.conf_threshold,
        vis_voxel_size=args.vis_voxel_size,
    )

    if args.run_os:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        import core.vision_encoder.pe as pe
        import core.vision_encoder.transforms as transforms

        sam3_model = build_sam3_image_model()
        processor = Sam3Processor(sam3_model, confidence_threshold=0.50)

        clip_model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=True)
        clip_model = clip_model.cuda()
        clip_tokenizer = transforms.get_text_tokenizer(clip_model.context_length)
        clip_preprocess = transforms.get_image_transform(clip_model.image_size)
    else:
        clip_model, clip_preprocess = None, None
        clip_tokenizer = None
        processor = None

    print(f"Initializing and loading VGGT-Omega from {VGGT_OMEGA_CHECKPOINT}...")
    model = VGGTOmega().eval()
    state_dict = torch.load(VGGT_OMEGA_CHECKPOINT, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)

    print(f"Loading images from {args.image_folder}...")
    image_names = collect_image_names(args, solver)
    if not image_names:
        raise RuntimeError("No images selected for reconstruction")
    print(f"Running single-shot reconstruction on {len(image_names)} frames")

    total_time_start = time.time()
    predictions = solver.run_reconstruction(
        image_names,
        model,
        clip_model,
        clip_preprocess,
        image_resolution=args.image_resolution,
    )
    solver.build_map(predictions)

    total_time = time.time() - total_time_start
    n_frames = len(image_names)
    print(f"{n_frames} frames processed")
    print("Total time:", total_time)
    print(f"Total time for VGGT-Omega: {solver.vggt_timer.total_time:.4f}s")
    print("Average VGGT time per frame:", solver.vggt_timer.total_time / n_frames)
    print("Average semantic time per frame:", solver.clip_timer.total_time / n_frames)
    print("Average total time per frame:", total_time / n_frames)
    print("Average FPS:", n_frames / total_time)

    if args.vis_map:
        solver.visualize_map()

    queries = ["walls"]
    if args.run_os:
        all_submap_points = []
        obb_pose_lines = []
        wall_segment_lines = []
        point_cloud_offset = 0
        start_time = time.time()
        for query in queries:
            text_emb = utils.compute_text_embeddings(clip_model, clip_tokenizer, query)
            matches = solver.map.retrieve_all_semantic_frames(text_emb)
            if not matches:
                print("No frames with semantic score above 0.8.")
                print("Time taken for query:", time.time() - start_time)
                continue

            for m_idx, (sem_score, submap_id, frame_index) in enumerate(matches):
                found_submap = solver.map.get_submap(submap_id)
                frame_ids = found_submap.get_frame_ids()
                frame_number = frame_ids[frame_index]
                best_img = found_submap.get_frame_at_index(frame_index)
                print(
                    f"Match {m_idx + 1}/{len(matches)} — submap {submap_id}, frame {frame_index}, score: {sem_score:.4f}"
                )
                with torch.no_grad():
                    best_img = to_pil_image(best_img)
                    inference_state = processor.set_image(best_img)
                    output = processor.set_text_prompt(state=inference_state, prompt=query)
                    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
                    print(f"  SAM3 masks for this frame: {masks.shape[0]} for '{query}'")
                    print("  SAM3 scores:", scores.cpu().numpy())

                for i in range(masks.shape[0]):
                    mask = masks[i].cpu().numpy()
                    submap_points = solver.get_points_in_mask(frame_index, mask)
                    if submap_points.size:
                        n_pts = submap_points.shape[0]
                        p_start = point_cloud_offset
                        p_end = point_cloud_offset + n_pts - 1
                        point_cloud_offset += n_pts
                        wall_segment_lines.append(
                            f"{p_start} {p_end} {frame_number:g} {i}"
                        )
                        all_submap_points.append(submap_points)

                        obb_center, obb_extent, obb_rotation, smallest_eigval = (
                            utils.compute_obb_from_points(submap_points)
                        )
                        if obb_center is not None:
                            rotvec = Rot.from_matrix(obb_rotation).as_rotvec()
                            row = np.concatenate(
                                [
                                    obb_center.ravel(),
                                    obb_extent.ravel(),
                                    rotvec.ravel(),
                                    np.array([smallest_eigval]),
                                ]
                            )
                            obb_pose_lines.append(" ".join(f"{v:.18g}" for v in row))
                        else:
                            print("Point cloud is too small to compute OBB")

        if obb_pose_lines:
            obb_path = args.log_path.replace(".txt", "_walls_obbs.txt")
            with open(obb_path, "w", encoding="ascii") as f:
                f.write("\n".join(obb_pose_lines) + "\n")
            print(f"Wrote {len(obb_pose_lines)} OBB lines to {obb_path}")

        if all_submap_points:
            merged = np.vstack(all_submap_points)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(merged.astype(np.float64))
            out_ply = args.log_path.replace(".txt", "_walls.ply")
            o3d.io.write_point_cloud(out_ply, pcd)
            print(f"Saved {merged.shape[0]} world-frame points to {out_ply}")

            seg_path = args.log_path.replace(".txt", "_walls_segments.txt")
            with open(seg_path, "w", encoding="ascii") as f:
                f.write("\n".join(wall_segment_lines) + "\n")
            print(
                f"Wrote {len(wall_segment_lines)} segment lines to {seg_path}"
            )

        print("Time taken for query:", time.time() - start_time)

    if not args.vis_map:
        solver.visualize_map()

    if args.log_results:
        print(f"Logging results to {args.log_path}")
        solver.write_poses_to_file(args.log_path)

        pcd_path = args.log_path.replace(".txt", "_points.pcd")
        print(f"Logging full point cloud to {pcd_path}")
        solver.write_points_to_file(pcd_path)

        if not args.skip_dense_log:
            logs_dir = args.log_path.replace(".txt", "_logs")
            print(f"Logging dense point clouds to {logs_dir}")
            solver.save_framewise_pointclouds(logs_dir)
    
    #input("Press Enter to continue...")

if __name__ == "__main__":
    main()
