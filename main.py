import argparse
import os
import sys
import warnings

import glob2
import numpy as np

from common import db, my_utils
from common.generator import Generator

np.random.seed(0)
warnings.filterwarnings("ignore")

def check_arg(args):
    parser = argparse.ArgumentParser(description='Rain renderer method')

    parser.add_argument('--dataset',
                        help='Enter dataset name. Dataset data must be located in: DATASET_ROOT/DATASET',
                        type=str, required=True)

    # Đường dẫn gốc cho dataset (ở đây là /kaggle/working/customdb)
    parser.add_argument('-k', '--dataset_root',
                        help='Path to database root',
                        default='/kaggle/working/customdb',  # Đường dẫn dataset chính
                        required=False)

    # Các tham số khác vẫn như cũ
    parser.add_argument('--particles',
                        help='Path to particles simulations',
                        default='/kaggle/working/particles',  # Đường dẫn tới particles
                        required=False)

    parser.add_argument('--streaks_db',
                        help='Path to rain streaks database (Garg and Nayar, 2006)',
                        default='/kaggle/input/database-rainrender',
                        required=False)

    parser.add_argument('--intensity',
                        help='Rain Intensities. List of fall rate comma-separated. E.g.: 1,15,25,50.',
                        type=str,
                        default='25',
                        required=False)

    parser.add_argument('--depth',
                        help='Path to depths',
                        default='/kaggle/working/customdb/depth',  # Đường dẫn tới thư mục depth
                        required=False)

    parser.add_argument('--output',
                        default='/kaggle/working/output',  # Đường dẫn lưu kết quả đầu ra
                        help='Where to save the output',
                        required=False)

    results = parser.parse_args(args)

    # Sử dụng các tham số đã cung cấp, tạo đối tượng results
    results.verbose = not results.noverbose

    # Cập nhật các đường dẫn liên quan đến ảnh và độ sâu
    results.images_root = os.path.join(results.dataset_root)  # Đường dẫn tới customdb
    results.images = {
        'all': os.path.join(results.dataset_root, 'rgb'),  # Đường dẫn tới thư mục rgb
    }

    results.depth = {
        'all': os.path.join(results.dataset_root, 'depth'),  # Đường dẫn tới thư mục depth
    }

    # Kiểm tra các thư mục 'rgb' và 'depth' có tồn tại không
    for key in results.images:
        assert os.path.exists(results.images[key]), f"RGB folder for {key} does not exist."
        assert os.path.exists(results.depth[key]), f"Depth folder for {key} does not exist."

    # Cập nhật các tham số khác
    results.texture = os.path.join(results.streaks_db, 'env_light_database', 'size32')
    results.norm_coeff = os.path.join(results.streaks_db, 'env_light_database', 'txt', 'normalized_env_max.txt')

    assert os.path.exists(results.streaks_db), ("rainstreakdb database is missing.", results.streaks_db)
    assert os.path.exists(results.texture), ("rainstreakdb database is not valid. Some files are missing.", results.texture)
    assert os.path.exists(results.norm_coeff), ("rainstreakdb database is not valid. Some files are missing.", results.norm_coeff)

    # Cập nhật các tham số về cường độ mưa
    results.intensity = [int(i) for i in results.intensity.split(",")]
    if results.frames:
        results.frames = [int(i) for i in results.frames.split(",")]

    dataset_name = results.dataset if "_gan" not in results.dataset else results.dataset[:-4]
    results.dataset_root = os.path.join(results.dataset_root, dataset_name)
    results.depth_root = os.path.join(results.depth, dataset_name)

    results.calib = None

    results.images_root = os.path.join(results.dataset_root)
    assert os.path.exists(results.images_root), ("Dataset folder does not exist.", results.images_root)

    sequences_filter = results.sequences.split(',')

    results = db.resolve_paths(results.dataset, results)
    results.settings = db.settings(results.dataset)

    # Filter sequences
    results.sequences = np.asarray([seq for seq in results.sequences if np.any([seq[:len(_s)] == _s for _s in sequences_filter])])

    # Build weathers to render
    results.weather = np.asarray([{"weather": "rain", "fallrate": i} for i in results.intensity])

    # Kiểm tra các sequence hợp lệ
    print("\nChecking sequences...")
    print(" {} sequences found: {}".format(len(results.sequences), [s for s in results.sequences]))
    for seq in results.sequences:
        valid = True
        if not os.path.exists(results.images[seq]):
            print(" Skip sequence '{}': images folder is missing {}".format(seq, results.images[seq]))
            valid = False
        if not os.path.exists(results.depth[seq]):
            print(" Skip sequence '{}': depth folder is missing {}".format(seq, results.depth[seq]))
            valid = False
        if results.calib[seq] is not None and not np.all([os.path.exists(f) for f in results.calib[seq]] if isinstance(results.calib[seq], list) else os.path.exists(results.calib[seq])):
            print(" Skip sequence '{}': calib data is missing {}".format(seq, results.calib[seq]))
            valid = False

        if not valid:
            results.sequences = results.sequences[results.sequences != seq]
            del results.images[seq]
            del results.depth[seq]
            del results.calib[seq]

    print("Found {} valid sequence(s): {}".format(len(results.sequences), [s for s in results.sequences]))

    # Resolving particle simulation files
    # Check simulation file exists, if not run particle simulation before hand
    print("\nResolving particles simulations...")
    particles_root = os.path.join(results.particles, results.dataset)

    sims_to_run = []
    results.particles = {}
    for seq in results.sequences:
        results.particles[seq] = db.sim(results.dataset, seq, particles_root)

        # Check if there is a need to run simulation
        weathers_to_run = [w for w in results.weather if len(glob2.glob(my_utils.particles_path(results.particles[seq]["path"], w))) == 0 or results.force_particles]
        if len(weathers_to_run) != 0:
            sims_to_run.append({"path": [results.particles[seq]["path"]], "options": [results.particles[seq]["options"]], "weather": weathers_to_run})

    if len(sims_to_run) == 0:
        print(" All particles simulations ready")
    else:
        print(" {} particles simulations to compute... Simulations will attempt to run automatically using weather particle simulator".format(len(sims_to_run)))
        for sim in sims_to_run:
            import tools.particles_simulation
            tools.particles_simulation.process(sim, force_recompute=True)  # Current script already decided it has to be re-run
        print(" All particles simulation completed")

    # Resolve particle simulation files path
    particles2 = {}
    for seq in results.sequences:
        try:
            particles2[seq] = [glob2.glob(my_utils.particles_path(results.particles[seq]["path"], w))[0] for w in results.weather]
        except Exception:
            print('Something went wrong, cannot locate particles simulation file for sequence {}'.format(seq))
            print("Might crash later on")

    results.particles = particles2

    return results


if __name__ == "__main__":
    print("\nBuilding internal parameters...")
    args = check_arg(sys.argv[1:])

    print("\nRunning renderers...")
    generator = Generator(args)
    generator.run()
