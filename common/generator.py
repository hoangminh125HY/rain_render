import os
import cv2
import glob
import random
import numpy as np

class Generator:
    def __init__(self, dataset_root, dataset, output_dir, rainstreakdb, rendering_strategy="naive_db"):
        self.dataset_root = dataset_root
        self.dataset = dataset
        self.output_dir = output_dir
        self.rendering_strategy = rendering_strategy
        self.rainstreakdb = rainstreakdb

        os.makedirs(self.output_dir, exist_ok=True)

        # load ảnh dataset
        self.rgb_images = sorted(glob.glob(os.path.join(self.dataset_root, self.dataset, "rgb", "*.png")))
        if not self.rgb_images:
            self.rgb_images = sorted(glob.glob(os.path.join(self.dataset_root, self.dataset, "rgb", "*.jpg")))

        # load rain streaks
        self.streaks = sorted(glob.glob(os.path.join(self.rainstreakdb, "*.png")))

        if not self.streaks:
            raise RuntimeError(f"No streaks found in {self.rainstreakdb}")

        print(f"Loaded {len(self.rgb_images)} input images, {len(self.streaks)} streak textures")

    def blend_streak(self, img, streak):
        """Overlay streak onto img with random alpha"""
        streak_resized = cv2.resize(streak, (img.shape[1], img.shape[0]))
        alpha = random.uniform(0.2, 0.6)
        beta = 1.0 - alpha
        blended = cv2.addWeighted(img, beta, streak_resized, alpha, 0)
        return blended

    def run(self):
        print("Running Rain Renderer (naive_db mode)...")
        out_dir = os.path.join(self.output_dir, "rgb")
        os.makedirs(out_dir, exist_ok=True)

        for i, img_path in enumerate(self.rgb_images):
            img = cv2.imread(img_path)
            streak_path = random.choice(self.streaks)
            streak = cv2.imread(streak_path)

            if img is None or streak is None:
                print(f"⚠️ Skip {img_path} (invalid image)")
                continue

            out_img = self.blend_streak(img, streak)
            out_path = os.path.join(out_dir, f"rain_{i:05d}.png")
            cv2.imwrite(out_path, out_img)

            if i % 10 == 0:
                print(f"Processed {i}/{len(self.rgb_images)} images...")

        print(f"✅ Done! Results saved in {out_dir}")
