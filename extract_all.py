import sys
sys.path.append('src/landmark_extraction')
from extract_landmarks import LandmarkExtractor
import numpy as np
from pathlib import Path

src = Path('data/raw/include_raw/Indian Sign Language words with Landmarks-1/ProcessedData_vivit')
out = Path('data/raw/include_raw')

ext = LandmarkExtractor()
total, skipped = 0, 0

for class_dir in sorted(src.iterdir()):
    if not class_dir.is_dir():
        continue
    out_class = out / class_dir.name
    out_class.mkdir(exist_ok=True)
    for mov in sorted(class_dir.glob('*.MOV')):
        out_npy = out_class / (mov.stem + '.npy')
        if out_npy.exists():
            continue
        try:
            seq = ext.extract_from_video(str(mov))
            np.save(str(out_npy), seq)
            total += 1
            if total % 50 == 0:
                print(f'Done {total} videos...')
        except Exception as e:
            print(f'Skip {mov.name}: {e}')
            skipped += 1

ext.close()
print(f'Finished. Extracted={total} Skipped={skipped}')