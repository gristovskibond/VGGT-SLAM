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
import matplotlib.pyplot as plt
import open3d as o3d
import os

import vggt_slam.slam_utils as utils
from vggt_slam.solver import Solver
from vggt_slam.submap import Submap

from vggt.models.vggt import VGGT

from huggingface_hub import login

if not os.getenv("SAM_API_KEY"):
    print("ERROR: SAM_API_KEY environment variable is not set.")
    print("Please set SAM_API_KEY to use the SAM model.")

sam_api_key = os.getenv("SAM_API_KEY")
login(sam_api_key)

parser = argparse.ArgumentParser(description="VGGT-SLAM demo")
parser.add_argument("--image_folder", type=str, default="examples/kitchen/images/", help="Path to folder containing images")
parser.add_argument("--vis_map", action="store_true", help="Visualize point cloud in viser as it is being build, otherwise only show the final map")
parser.add_argument("--vis_voxel_size", type=float, default=None, help="Voxel size for downsampling the point cloud in the viewer (e.g. 0.05 for 5 cm). Default: no downsampling")
parser.add_argument("--run_os", action="store_true", help="Enable open-set semantic search with Perception Encoder CLIP and SAM3")
parser.add_argument("--vis_flow", action="store_true", help="Visualize optical flow from RAFT for keyframe selection")
parser.add_argument("--log_results", action="store_true", help="save txt file with results")
parser.add_argument("--skip_dense_log", action="store_true", help="by default, logging poses and logs dense point clouds. If this flag is set, dense logging is skipped")
parser.add_argument("--log_path", type=str, default="poses.txt", help="Path to save the log file")
parser.add_argument("--submap_size", type=int, default=16, help="Number of new frames per submap, does not include overlapping frames or loop closure frames")
parser.add_argument("--overlapping_window_size", type=int, default=1, help="ONLY DEFAULT OF 1 SUPPORTED RIGHT NOW. Number of overlapping frames, which are used in SL(4) estimation")
parser.add_argument("--max_loops", type=int, default=1, help="ONLY DEFAULT OF 1 SUPPORTED RIGHT NOW or 0 to disable loop closures.")
parser.add_argument("--min_disparity", type=float, default=50, help="Minimum disparity to generate a new keyframe")
parser.add_argument("--conf_threshold", type=float, default=25.0, help="Initial percentage of low-confidence points to filter out")
parser.add_argument("--lc_thres", type=float, default=0.95, help="Threshold for image retrieval. Range: [0, 1.0]. Higher = more loop closures")


def main():
    """
    Main function that wraps the entire pipeline of VGGT-SLAM.
    """
    args = parser.parse_args()

    use_optical_flow_downsample = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    solver = Solver(
        init_conf_threshold=args.conf_threshold,
        lc_thres=args.lc_thres,
        vis_voxel_size=args.vis_voxel_size
    )

    print("Initializing and loading VGGT model...")


    if args.run_os:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        import core.vision_encoder.pe as pe
        import core.vision_encoder.transforms as transforms

        sam3_model = build_sam3_image_model()
        processor = Sam3Processor(sam3_model, confidence_threshold=0.50)

        clip_model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=True)  # Downloads from HF
        clip_model = clip_model.cuda()
        clip_tokenizer = transforms.get_text_tokenizer(clip_model.context_length)
        clip_preprocess = transforms.get_image_transform(clip_model.image_size)
    else:
        clip_model, clip_preprocess = None, None
        clip_tokenizer = None

    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))

    model.eval()
    model = model.to(torch.bfloat16)  # use half precision
    model = model.to(device)

    # Use the provided image folder path
    print(f"Loading images from {args.image_folder}...")
    image_names = [f for f in glob.glob(os.path.join(args.image_folder, "*")) 
               if "depth" not in os.path.basename(f).lower() and "txt" not in os.path.basename(f).lower() 
               and "db" not in os.path.basename(f).lower()]

    image_names = utils.sort_images_by_number(image_names)
    downsample_factor = 1
    image_names = utils.downsample_images(image_names, downsample_factor)
    print(f"Found {len(image_names)} images")

    image_names_subset = []
    count = 0
    image_count = 0
    total_time_start = time.time()
    keyframe_time = utils.Accumulator()
    backend_time = utils.Accumulator()
    for image_name in tqdm(image_names):
        if use_optical_flow_downsample:
            with keyframe_time:
                img = cv2.imread(image_name)
                enough_disparity = solver.flow_tracker.compute_disparity(img, args.min_disparity, args.vis_flow)
                if enough_disparity:
                    image_names_subset.append(image_name)
                    image_count += 1
        else:
            image_names_subset.append(image_name)

        # Run submap processing if enough images are collected or if it's the last group of images.
        if len(image_names_subset) == args.submap_size + args.overlapping_window_size or image_name == image_names[-1]:
            count += 1
            print(image_names_subset)
            t1 = time.time()
            predictions = solver.run_predictions(image_names_subset, model, args.max_loops, clip_model, clip_preprocess)
            print("Solver total time", time.time()-t1)
            print(count, "submaps processed")

            solver.add_points(predictions)

            with backend_time:
                solver.graph.optimize()

            loop_closure_detected = len(predictions["detected_loops"]) > 0
            if args.vis_map:
                if loop_closure_detected:
                    solver.update_all_submap_vis()
                else:
                    solver.update_latest_submap_vis()
            
            # Reset for next submap.
            image_names_subset = image_names_subset[-args.overlapping_window_size:]

    total_time = time.time() - total_time_start
    average_fps = total_time / image_count
    print(image_count, "frames processed")
    print("Total time:", total_time)
    print(f"Total time for VGGT calls: {solver.vggt_timer.total_time:.4f}s")
    print("Average VGGT time per frame:", solver.vggt_timer.total_time / image_count)
    print("Average loop closure time per frame:", solver.loop_closure_timer.total_time / image_count)
    print("Average keyframe selection time per frame:", keyframe_time.total_time / image_count)
    print("Average backend time per frame:", backend_time.total_time / image_count)
    print("Average semantic time per frame:", solver.clip_timer.total_time / image_count)
    print("Average total time per frame:", total_time / image_count)
    print("Average FPS:", 1 / average_fps)
        
    print("Total number of submaps in map", solver.map.get_num_submaps())
    print("Total number of loop closures in map", solver.graph.get_num_loops())


    queries = ["walls"]
    #queries = ["ceiling"]
    if args.run_os:
        all_submap_points = []
        obb_pose_lines = []
        start_time = time.time()
        for query in queries:
            '''
            while True:
                # Prompt user for text input
                query = input("\nEnter text query or q to quit: ").strip()
                if len(query) == 0:
                    print("Empty query. Exiting.")
                    return
                
                if query == "q":
                    print("Exiting.")
                    return
            '''
            text_emb = utils.compute_text_embeddings(clip_model, clip_tokenizer, query)
            #x, y, z = solver.map.retrieve_best_semantic_frame(text_emb)
            #print(f"Best semantic frame: {x}, {y}, {z}")
            
            matches = solver.map.retrieve_all_semantic_frames(text_emb)
            if not matches:
                print("No frames with semantic score above 0.8.")
                print("Time taken for query:", time.time() - start_time)
                continue

            time_str = time.strftime("%Y%m%d_%H%M%S")
            for m_idx, (sem_score, submap_id, frame_index) in enumerate(matches):
                found_submap = solver.map.get_submap(submap_id)
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

                
                #masked_img = utils.overlay_masks(best_img, masks)
                #masked_img.show()
                #masked_img.save(f"masked_img_{time_str}_{m_idx}.png")

                for i in range(masks.shape[0]):
                    mask = masks[i].cpu().numpy()
                    submap_points = found_submap.get_points_in_mask(frame_index, mask, solver.graph)
                    if submap_points.size:
                        all_submap_points.append(submap_points)
                        
                        obb_center, obb_extent, obb_rotation, smallest_eigval = utils.compute_obb_from_points(
                            submap_points
                        )
                        if obb_center is not None:
                            rotvec = Rot.from_matrix(obb_rotation).as_rotvec()
                            row = np.concatenate(
                                [obb_center.ravel(), obb_extent.ravel(), rotvec.ravel(), np.array([smallest_eigval])]
                            )
                            obb_pose_lines.append(" ".join(f"{v:.18g}" for v in row))
                            #print("obb_center", obb_center)
                            #print("obb_extent", obb_extent)
                            #print("obb_rotation", obb_rotation)
                            #print("smallest_eigval", smallest_eigval)
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
            #out_ply = f"query_points_{time_str}.ply"
            out_ply = args.log_path.replace(".txt", "_walls.ply")
            o3d.io.write_point_cloud(out_ply, pcd)
            print(f"Saved {merged.shape[0]} world-frame points to {out_ply}")

        print("Time taken for query:", time.time() - start_time)

    if not args.vis_map:
        # just show the map after all submaps have been processed
        solver.update_all_submap_vis()

    if args.log_results:
        print(f"Logging results to {args.log_path}")
        solver.map.write_poses_to_file(args.log_path, solver.graph, kitti_format=False)

        # Log the full point cloud as one file, used for visualization.
        print(f"Logging full point cloud to {args.log_path.replace('.txt', '_points.pcd')}")
        solver.map.write_points_to_file(solver.graph, args.log_path.replace(".txt", "_points.pcd"))

        if not args.skip_dense_log:
            # Log the dense point cloud for each submap.
            print(f"Logging dense point clouds to {args.log_path.replace('.txt', '_logs')}")
            solver.map.save_framewise_pointclouds(solver.graph, args.log_path.replace(".txt", "_logs"))

    #input("Press Enter to continue...")

if __name__ == "__main__":
    main()
