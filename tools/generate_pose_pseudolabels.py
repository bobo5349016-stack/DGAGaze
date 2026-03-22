import os
import csv
import cv2
import argparse
from tqdm import tqdm
from sixdrepnet import SixDRepNet



def is_image_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(image_root):
    image_list = []
    for root, _, files in os.walk(image_root):
        for fname in files:
            if is_image_file(fname):
                abs_path = os.path.join(root, fname)
                rel_path = os.path.relpath(abs_path, image_root).replace("\\", "/")
                image_list.append((abs_path, rel_path))
    image_list.sort(key=lambda x: x[1])
    return image_list


def main(args):
    image_root = os.path.abspath(args.image_root)
    output_csv = os.path.abspath(args.output_csv)

    if not os.path.isdir(image_root):
        raise FileNotFoundError(f"Image root does not exist: {image_root}")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    image_list = collect_images(image_root)
    if len(image_list) == 0:
        raise RuntimeError(f"No image files found under: {image_root}")

    model = SixDRepNet()

    num_written = 0
    num_failed = 0

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "yaw", "pitch", "roll"])
        for abs_path, rel_path in tqdm(image_list, desc="Generating pose pseudo-labels"):
            img = cv2.imread(abs_path)
            if img is None:
                print(f"[Warning] Failed to read image: {abs_path}")
                num_failed += 1
                continue

            try:
                
                pitch, yaw, roll = model.predict(img)

              
                writer.writerow([rel_path, float(yaw), float(pitch), float(roll)])
                num_written += 1

            except Exception as e:
                print(f"[Warning] Failed on {abs_path}: {e}")
                num_failed += 1

    print("\nFinished.")
    print(f"Image root   : {image_root}")
    print(f"Output csv   : {output_csv}")
    print(f"Written rows : {num_written}")
    print(f"Failed rows  : {num_failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate head pose pseudo-labels (yaw, pitch) using 6DRepNet."
    )
    parser.add_argument(
        "--image_root",
        type=str,
        required=True,
        help="Root directory of face images, /userdata/Diap.zip/EyeDiap/Image"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Output CSV path, /userdata/eyediap/pseudo_labels/train_pose_labels.csv"
    )

    args = parser.parse_args()
    main(args)