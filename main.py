import os
import sys
import warnings
import glob2
import numpy as np

from common import db, my_utils
from common.generator import Generator

np.random.seed(0)
warnings.filterwarnings("ignore")

def check_arg():
    # Các đường dẫn trực tiếp thay vì tham số từ dòng lệnh
    results = {}

    # Đường dẫn dataset
    results['dataset'] = 'customdb'  # Tên dataset của bạn

    # Đường dẫn gốc cho dataset
    results['dataset_root'] = '/kaggle/working/customdb'

    # Đường dẫn particles
    results['particles'] = '/kaggle/working/particles'

    # Đường dẫn rain streaks database
    results['streaks_db'] = '/kaggle/input/database-rainrender'

    # Đường dẫn texture và norm_coeff
    results['texture'] = '/kaggle/input/database-rainrender/env_light_database/size32'
    results['norm_coeff'] = '/kaggle/input/database-rainrender/env_light_database/txt/normalized_env_max.txt'

    # Cường độ mưa
    results['intensity'] = [25]  # Bạn có thể thay đổi cường độ mưa tùy ý

    # Đường dẫn độ sâu
    results['depth'] = '/kaggle/working/customdb'  # Đảm bảo đường dẫn đúng

    # Kiểm tra các thư mục và tệp đã tồn tại hay chưa
    assert os.path.exists(results['streaks_db']), ("rainstreakdb database is missing.", results['streaks_db'])
    assert os.path.exists(results['texture']), ("Texture folder is missing.", results['texture'])
    assert os.path.exists(results['norm_coeff']), ("Normalized environment coefficients file is missing.", results['norm_coeff'])
    assert os.path.exists(results['dataset_root']), ("Dataset root folder does not exist.", results['dataset_root'])

    results['images_root'] = os.path.join(results['dataset_root'])

    # Kiểm tra xem thư mục images_root có tồn tại không
    assert os.path.exists(results['images_root']), ("Dataset images folder does not exist.", results['images_root'])

    # Thêm các phần còn lại của quá trình xử lý như trong mã gốc
    results['calib'] = None
    sequences_filter = ''
    results['sequences'] = ['train', 'val', 'test']  # Bạn có thể thay đổi danh sách sequences này theo yêu cầu của bạn

    # Lấy các đường dẫn đến các sequence (train, val, test)
    results = db.resolve_paths(results['dataset'], results)
    results['settings'] = db.settings(results['dataset'])

    # Lọc các sequence hợp lệ
    results['sequences'] = np.asarray([seq for seq in results['sequences'] if np.any([seq[:len(s)] == s for s in sequences_filter])])

    # Xây dựng các weather conditions
    results['weather'] = np.asarray([{"weather": "rain", "fallrate": i} for i in results['intensity']])

    # Kiểm tra các sequence hợp lệ
    print("\nChecking sequences...")
    print(f" {len(results['sequences'])} sequences found: {results['sequences']}")

    for seq in results['sequences']:
        valid = True
        if not os.path.exists(results['images'][seq]):
            print(f" Skip sequence '{seq}': images folder is missing {results['images'][seq]}")
            valid = False
        if not os.path.exists(results['depth'][seq]):
            print(f" Skip sequence '{seq}': depth folder is missing {results['depth'][seq]}")
            valid = False

        if not valid:
            results['sequences'] = results['sequences'][results['sequences'] != seq]
            del results['images'][seq]
            del results['depth'][seq]

    print(f"Found {len(results['sequences'])} valid sequence(s): {results['sequences']}")

    # Resolving particle simulation files
    print("\nResolving particles simulations...")
    particles_root = os.path.join(results['particles'], results['dataset'])

    sims_to_run = []
    results['particles'] = {}
    for seq in results['sequences']:
        results['particles'][seq] = db.sim(results['dataset'], seq, particles_root)

        # Check if there is a need to run simulation
        weathers_to_run = [w for w in results['weather'] if len(glob2.glob(my_utils.particles_path(results['particles'][seq]["path"], w))) == 0]
        if len(weathers_to_run) != 0:
            sims_to_run.append({"path": [results['particles'][seq]["path"]], "options": [results['particles'][seq]["options"]], "weather": weathers_to_run})

    if len(sims_to_run) == 0:
        print(" All particles simulations ready")
    else:
        print(f" {len(sims_to_run)} particles simulations to compute... Simulations will attempt to run automatically using weather particle simulator")
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
    args = check_arg()  # Không cần truyền đối số vào

    print("\nRunning renderers...")
    generator = Generator(args)
    generator.run()
