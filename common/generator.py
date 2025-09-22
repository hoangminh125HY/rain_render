import os
import sys
import time

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageChops
from natsort import natsorted

from common import add_attenuation, my_utils
from common import solid_angle
from common.bad_weather import DBManager, DropType, RainRenderer, EnvironmentMapGenerator, FovComputation
from common.drop_depth_map import DropDepthMap

plt.ion()

FOG_ATT = 1
USE_DEPTH_WEIGHTING = 0  # TODO: not used for a while

class Generator:
    def __init__(self, args):
        # strategy
        self.conflict_strategy = args.conflict_strategy
        self.rendering_strategy = args.rendering_strategy

        # output paths
        if args.rendering_strategy is None:
            self.output_root = os.path.join(args.output, args.dataset)
        else:
            self.output_root = os.path.join(args.output, args.dataset + '_' + args.rendering_strategy)

        # dataset info
        self.dataset = args.dataset
        self.dataset_root = args.dataset_root
        self.images = args.images
        self.sequences = args.sequences
        self.depth = args.depth
        self.particles = args.particles
        self.weather = args.weather
        self.texture = args.texture
        self.norm_coeff = args.norm_coeff
        self.save_envmap = args.save_envmap
        self.settings = args.settings

        # dataset specific
        self.calib = args.calib

        # camera info
        self.exposure = args.settings["cam_exposure"]
        self.camera_gain = args.settings["cam_gain"]
        self.focal = args.settings["cam_focal"] / 1000.
        self.f_number = args.settings["cam_f_number"]
        self.focus_plane = args.settings["cam_focus_plane"]

        # aesthetic params
        self.noise_scale = args.noise_scale
        self.noise_std = args.noise_std
        self.opacity_attenuation = args.opacity_attenuation

        # generator run params
        self.frame_start = args.frame_start
        self.frame_end = args.frame_end
        self.frame_step = args.frame_step
        self.frames = args.frames
        self.verbose = args.verbose

        # options for environment map and irradiance types
        self.env_type = 'ours'  # 'pano' | 'ours'
        self.irrad_type = 'ambient'  # 'garg' | 'ambient'


        # initialize to None internal big frame by frame object
        self.db = None
        self.renderer = None
        self.fov_comp = None
        self.BGR_env_map = None
        self.env_map_xyY = None
        self.solid_angle_map = None

        # check if everything is fine
        self.check_folders()

    def check_folders(self):
        print('Output directory: {}'.format(self.output_root))

        # Verify existing folders
        existing_folders = []
        for sequence in self.sequences:
            for w in self.weather:
                # loading simulator file path
                out_dir = os.path.join(self.output_root, sequence, w["weather"], '{}mm'.format(w["fallrate"]))

                if os.path.exists(out_dir):
                    existing_folders.append(out_dir)

        if len(existing_folders) != 0 and self.conflict_strategy is None:
            print("\r\nFolders already exist: \n%s" % "\n".join([d for d in existing_folders]))
            while self.conflict_strategy not in ["overwrite", "skip", "rename_folder"]:
                self.conflict_strategy = input(
                    "\r\nWhat strategy to use (overwrite|skip|rename_folder):   ")

        assert(self.conflict_strategy in [None, "overwrite", "skip", "rename_folder"])

    @staticmethod
    def crop_drop(streak):
        streak = (streak * 255).astype(np.uint8)
        im = Image.fromarray(streak)
        background = Image.new(im.mode, im.size, im.getpixel((0, 0)))
        diff = ImageChops.difference(im, background)
        diff = ImageChops.add(diff, diff, 2.0, -100)
        bBox = diff.getbbox()

        im_cropped = im.crop(bBox)
        im_cropped = np.asarray(im_cropped) / 255
        return im_cropped.astype(np.float64)

    def compute_drop(self, bg, drop_dict, rainy_bg, rainy_mask, rainy_saturation_mask):
        # Drop taken from database
        streak_db_drop = self.db.take_drop_texture(drop_dict)

        image_height, image_width = bg.shape[:2]

        # Gaussian streaks do not need a perspective warping. If strak is not BIG -> Gaussian streak
        if drop_dict.drop_type == DropType.Big:
            pts1, pts2, maxC, minC = self.renderer.warping_points(drop_dict, streak_db_drop, image_width, image_height)
            shape = np.subtract(maxC, minC).astype(int)
            perspective_matrix = cv2.getPerspectiveTransform(pts1, pts2)
            drop = cv2.warpPerspective(streak_db_drop, perspective_matrix, (max(shape[0], 1), max(shape[1], 1)),
                                       flags=cv2.INTER_CUBIC)
            drop = np.clip(drop, 0, 1)
        else:
            # in case of drops from database
            # Gaussian noise to simulate soft wind (in degrees)
            noise = np.random.normal(0.0, self.noise_std) * self.noise_scale

            dir1 = drop_dict.image_position_start - drop_dict.image_position_end
            n1 = np.linalg.norm(dir1)
            dir1 = dir1 / n1
            dir2 = np.array([0, -1])

            # Drop angle in degrees; add small random gaussian noise to represent localized wind
            theta = np.rad2deg(np.arccos(np.dot(dir1, dir2)))

            # Note: The noise is added to the drop coordinates AFTER the drop angle is calculated so the rotate_bound
            # function, which uses interpolation (contrarily to the drop position which are in integers),
            # would be more accurate
            nx, ny = np.cos(np.deg2rad(noise)), np.sin(np.deg2rad(noise))
            mean_x = (drop_dict.image_position_end[0] + drop_dict.image_position_start[0]) / 2
            mean_y = (drop_dict.image_position_end[1] + drop_dict.image_position_start[1]) / 2
            drop_dict.image_position_start[:] = \
                (drop_dict.image_position_start[0] - mean_x) * nx - \
                (drop_dict.image_position_start[1] - mean_y) * ny + mean_x,\
                (drop_dict.image_position_start[0] - mean_x) * ny + \
                (drop_dict.image_position_start[1] - mean_y) * nx + mean_y
            drop_dict.image_position_end[:] = \
                (drop_dict.image_position_end[0] - mean_x) * nx - \
                (drop_dict.image_position_end[1] - mean_y) * ny + mean_x,\
                (drop_dict.image_position_end[0] - mean_x) * ny + \
                (drop_dict.image_position_end[1] - mean_y) * nx + mean_y

            drop = imutils.rotate_bound(streak_db_drop, theta + noise)

            drop = cv2.flip(drop, 0) if drop_dict.image_position_end[0] > rainy_bg.shape[1] // 2 else drop
            height = max(abs(drop_dict.image_position_end[1] - drop_dict.image_position_start[1]), 2)
            width = max(abs(
                drop_dict.image_position_end[0] - drop_dict.image_position_start[0]), drop_dict.max_width + 2)
            drop = cv2.resize(drop, (width, height), interpolation=cv2.INTER_AREA)
            drop = np.clip(drop, 0, 1)
            minC = drop_dict.image_position_start

        # Compute alpha channel from any other channel (since it was gray)
        drop = np.dstack([drop, drop[..., 0]])

        ###########################################   COLOUR DROP  #################################################
        drop_fov_pts, drop_fov_pts3d, drop_direction, drop_position = \
            self.fov_comp.compute_fov_plane_points(drop_dict, self.renderer.radius, self.renderer.fov,
                                                   20, self.BGR_env_map.shape)
        try:
            rainy_bg, rainy_mask, rainy_saturation_mask, drop, blended_drop, minC = \
                self.renderer.add_drop_to_image(self.dataset, self.env_map_xyY, self.solid_angle_map, drop_fov_pts,
                                                minC, bg, rainy_bg, rainy_mask, rainy_saturation_mask, drop, drop_dict,
                                                self.irrad_type, self.rendering_strategy, self.opacity_attenuation)
        except Exception as e:
            import traceback
            print('Erroneous drop (' + str(e) + ')')
            print(traceback.print_exc())
            blended_drop = None

        return rainy_bg, rainy_mask, rainy_saturation_mask, drop, blended_drop, minC

    def run(self):
        process_t0 = time.time()

        folders_num = len(self.images)

        # case for any number of sequences and supported rain intensities
        for folder_idx, sequence in enumerate(self.sequences):
            folder_t0 = time.time()
            print(f'\nSequence: {sequence}')
                    
            # Kiểm tra xem sequence có trong self.particles hay không
            if sequence not in self.particles:
                print(f"Error: No particles data found for sequence '{sequence}'")
                continue  # Bỏ qua sequence này nếu không có dữ liệu particles

            # Kiểm tra ảnh RGB và Depth có đúng không
            if sequence not in self.images:
                print(f"Error: No RGB images found for sequence '{sequence}'")
                continue

            if sequence not in self.depth:
                print(f"Error: No Depth images found for sequence '{sequence}'")
                continue

            # Lấy số lượng mô phỏng cho sequence này
            sim_num = len(self.particles[sequence])
            depth_folder = self.depth[sequence]

            for sim_idx, sim_weather in enumerate(self.weather):
                weather, fallrate = sim_weather["weather"], sim_weather["fallrate"]

                out_seq_dir = os.path.join(self.output_root, sequence)
                out_dir = os.path.join(out_seq_dir, weather, '{}mm'.format(fallrate))
                sim_file = self.particles[sequence][sim_idx]

                # Resolve output path
                path_exists = os.path.exists(out_dir)
                if path_exists:
                    if self.conflict_strategy == "skip":
                        pass
                    elif self.conflict_strategy == "overwrite":
                        pass
                    elif self.conflict_strategy == "rename_folder":
                        out_dir_, out_shift = out_dir, 0
                        while os.path.exists(out_dir_ + '_copy%05d' % out_shift):
                            out_shift += 1

                        out_dir = out_dir_ + '_copy%05d' % out_shift
                    else:
                        raise NotImplementedError

                # Create directory
                os.makedirs(out_dir, exist_ok=True)

                # Default fog-like rain parameters
                fog_params = {"rain_intensity": fallrate, "focal": self.focal, "f_number": self.f_number, "angle": 90,
                              "exposure": self.exposure, "camera_gain": self.camera_gain}

                files = natsorted(np.array([os.path.join(self.images[sequence], picture) for picture in my_utils.os_listdir(self.images[sequence])]))
                depth_files = natsorted(np.array([os.path.join(depth_folder, depth) for depth in my_utils.os_listdir(depth_folder)]))

                # Ensure only .jpg files are considered
                files = [f for f in files if f.endswith('.jpg')]
                depth_files = [f for f in depth_files if f.endswith('.jpg')]

                if len(files) != len(depth_files):
                    print(f"Warning: The number of RGB files and Depth files for sequence {sequence} does not match.")
                    continue

                for idx, (rgb_file, depth_file) in enumerate(zip(files, depth_files)):
                    rgb_path = os.path.join(self.images[sequence], rgb_file)
                    depth_path = os.path.join(self.depth[sequence], depth_file)

                    # Read images
                    rgb_image = cv2.imread(rgb_path)
                    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

                    if rgb_image is None or depth_image is None:
                        print(f"Error reading images {rgb_path} or {depth_path}. Skipping...")
                        continue

                    # Process the images as per your simulation code
                    # Add your processing code here...

