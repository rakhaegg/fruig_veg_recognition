"""
Salin dataset 'train', 'validation', 'test' menjadi struktur standar:
dst/train, dst/val, dst/test.

Contoh:
    python scripts/split.py --src data/raw --dst data/dataset
"""

import argparse
import pathlib
import shutil
import sys

MAP_SPLIT = {          # src → dst
    "train": "train",
    "validation": "val",
    "test": "test",
}

def copy_split(src_root: pathlib.Path, dst_root: pathlib.Path) -> None:
    for src_name, dst_name in MAP_SPLIT.items():
        src_split = src_root / src_name
        dst_split = dst_root / dst_name

        if not src_split.exists():
            print(f"[WARNING] Folder {src_split} tidak ditemukan, lewati.", file=sys.stderr)
            continue

        for cls in src_split.iterdir():
            if not cls.is_dir():
                continue
            target_cls = dst_split / cls.name
            target_cls.mkdir(parents=True, exist_ok=True)
            for img in cls.iterdir():
                if img.is_file():
                    shutil.copy2(img, target_cls / img.name)

        print(f"Selesai menyalin {src_name} → {dst_name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Folder root dataset Kaggle (punya subdir train/ validation/ test/)")
    parser.add_argument("--dst", required=True, help="Folder tujuan")
    args = parser.parse_args()

    src_root = pathlib.Path(args.src)
    dst_root = pathlib.Path(args.dst)

    copy_split(src_root, dst_root)

if __name__ == "__main__":
    main()
