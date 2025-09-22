import os
import sys
import warnings
import numpy as np

from common import db, my_utils
from common.generator import Generator

np.random.seed(0)
warnings.filterwarnings("ignore")

def check_arg(args):
    # Thiết lập các tham số với giá trị mặc định hoặc cố định
    results = {}

    # Tách biệt tham số đường dẫn riêng biệt cho dataset
    results['dataset'] = '/kaggle/working/ppe'  # Đường dẫn đến thư mục chứa dataset 'ppe'
    results['train_root'] = os.path.join(results['dataset'], 'train')  # Đường dẫn đến thư mục train
    results['val_root'] = os.path.join(results['dataset'], 'val')  # Đường dẫn đến thư mục val
    results['test_root'] = os.path.join(results['dataset'], 'test')  # Đường dẫn đến thư mục test

    # Các thư mục chứa ảnh RGB và Depth cho từng phần của dataset
    results['train_rgb'] = os.path.join(results['train_root'], 'rgb')  # Đường dẫn đến thư mục RGB trong train
    results['train_depth'] = os.path.join(results['train_root'], 'depth')  # Đường dẫn đến thư mục Depth trong train

    results['val_rgb'] = os.path.join(results['val_root'], 'rgb')  # Đường dẫn đến thư mục RGB trong val
    results['val_depth'] = os.path.join(results['val_root'], 'depth')  # Đường dẫn đến thư mục Depth trong val

    results['test_rgb'] = os.path.join(results['test_root'], 'rgb')  # Đường dẫn đến thư mục RGB trong test
    results['test_depth'] = os.path.join(results['test_root'], 'depth')  # Đường dẫn đến thư mục Depth trong test

    # Kiểm tra sự tồn tại của các thư mục và tệp
    assert os.path.exists(results['train_rgb']), f"Train RGB folder is missing at {results['train_rgb']}"
    assert os.path.exists(results['train_depth']), f"Train Depth folder is missing at {results['train_depth']}"
    assert os.path.exists(results['val_rgb']), f"Val RGB folder is missing at {results['val_rgb']}"
    assert os.path.exists(results['val_depth']), f"Val Depth folder is missing at {results['val_depth']}"
    assert os.path.exists(results['test_rgb']), f"Test RGB folder is missing at {results['test_rgb']}"
    assert os.path.exists(results['test_depth']), f"Test Depth folder is missing at {results['test_depth']}"

    # Tách biệt các tham số đường dẫn khác như streaks_db, texture, norm_coeff
    results['streaks_db'] = '/kaggle/input/database-rainrender'  # Đường dẫn đến rain streaks database
    results['texture'] = '/kaggle/input/database-rainrender/env_light_database/size32'  # Đường dẫn đến texture database
    results['norm_coeff'] = '/kaggle/input/database-rainrender/env_light_database/txt/normalized_env_max.txt'  # Đường dẫn đến norm_coeff

    # Kiểm tra sự tồn tại của các thư mục và tệp liên quan đến streaks_db, texture và norm_coeff
    assert os.path.exists(results['streaks_db']), f"Streaks DB is missing at {results['streaks_db']}"
    assert os.path.exists(results['texture']), f"Texture database is missing at {results['texture']}"
    assert os.path.exists(results['norm_coeff']), f"Norm coefficient is missing at {results['norm_coeff']}"

    # Các tham số khác
    results['intensity'] = [25]  # Cường độ mưa mặc định
    results['frames'] = []

    # Cập nhật các giá trị đường dẫn liên quan đến dataset
    dataset_name = "/kaggle/input/customdb"  # Ví dụ dataset_name cố định, có thể thay đổi sau
    results['dataset_root'] = os.path.join(results['dataset'], dataset_name)


    # Kiểm tra sự tồn tại của thư mục chứa ảnh
    results['images_root'] = os.path.join(results['dataset_root'])
    assert os.path.exists(results['images_root']), f"Dataset folder does not exist at {results['images_root']}"

    # Các tham số cho sequence
    results['sequences'] = ['sequence1', 'sequence2']  # Ví dụ danh sách sequences
    results['weather'] = np.asarray([{"weather": "rain", "fallrate": i} for i in results['intensity']])

    # Kiểm tra các sequences hợp lệ
    print("\nChecking sequences...")
    print(f" {len(results['sequences'])} sequences found: {results['sequences']}")
    for seq in results['sequences']:
        valid = True
        if not os.path.exists(results['train_rgb']):
            print(f" Skip sequence '{seq}': train rgb folder is missing at {results['train_rgb']}")
            valid = False
        if not valid:
            results['sequences'] = [s for s in results['sequences'] if s != seq]

    print(f"Found {len(results['sequences'])} valid sequence(s): {results['sequences']}")

    # Resolving particle simulation files
    print("\nResolving particles simulations...")
    particles_root = os.path.join(results['particles'], results['dataset'])

    sims_to_run = []
    results['particles'] = {}
    for seq in results['sequences']:
        results['particles'][seq] = db.sim(results['dataset'], seq, particles_root)

        # Kiểm tra nếu cần chạy simulation
        weathers_to_run = [w for w in results['weather'] if len(glob2.glob(my_utils.particles_path(results['particles'][seq]["path"], w))) == 0 or results['force_particles']]
        if len(weathers_to_run) != 0:
            sims_to_run.append({"path": [results['particles'][seq]["path"]], "options": [results['particles'][seq]["options"]], "weather": weathers_to_run})

    if len(sims_to_run) == 0:
        print(" All particles simulations ready")
    else:
        print(f" {len(sims_to_run)} particles simulations to compute...")
        for sim in sims_to_run:
            import tools.particles_simulation
            tools.particles_simulation.process(sim, force_recompute=True)
        print(" All particles simulation completed")

    # Resolve particle simulation files path
    particles2 = {}
    for seq in results['sequences']:
        try:
            particles2[seq] = [glob2.glob(my_utils.particles_path(results['particles'][seq]["path"], w))[0] for w in results['weather']]
        except Exception:
            print(f"Something went wrong, cannot locate particles simulation file for sequence {seq}")
            print("Might crash later on")

    results['particles'] = particles2

    return results


if __name__ == "__main__":
    print("\nBuilding internal parameters...")
    args = check_arg(sys.argv[1:])

    print("\nRunning renderers...")
    generator = Generator(args)
    generator.run()
