import os
import cv2
import csv
import json
import torch
import numpy as np
from easydict import EasyDict as edict
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def Decode_Diap(line):
    anno = edict()
    anno.face, anno.lefteye, anno.righteye = line[0], line[1], line[2]
    anno.name = line[3]
    anno.gaze3d, anno.head3d = line[4], line[5]
    anno.gaze2d, anno.head2d = line[6], line[7]
    return anno


def Decode_Gaze360(line):
    anno = edict()
    anno.face, anno.lefteye, anno.righteye = line[0], line[1], line[2]
    anno.name = line[3]
    anno.gaze3d = line[4]
    anno.gaze2d = line[5]
    return anno


def Decode_Dict():
    mapping = edict()
    mapping.eyediap = Decode_Diap
    mapping.gaze360 = Decode_Gaze360
    return mapping


def long_substr(str1, str2):
    substr = ''
    for i in range(len(str1)):
        for j in range(len(str1) - i + 1):
            if j > len(substr) and (str1[i:i + j] in str2):
                substr = str1[i:i + j]
    return len(substr)


def Get_Decode(name):
    mapping = Decode_Dict()
    keys = list(mapping.keys())
    name = name.lower()
    score = [long_substr(name, i) for i in keys]
    key = keys[score.index(max(score))]
    return mapping[key]


def norm_rel_path(p):
    return os.path.normpath(p).replace("\\", "/")


def load_pose_pseudo_labels(pose_label_path):
    """
    Supported formats:
    1) CSV:  path,yaw,pitch,roll
    2) JSON: {"relative/path.jpg": [yaw, pitch, roll], ...}
    3) TXT : relative/path.jpg,yaw,pitch,roll
             or relative/path.jpg yaw pitch roll
    """
    if pose_label_path is None or not os.path.exists(pose_label_path):
        raise ValueError(f"Pose pseudo-label file not found: {pose_label_path}")

    ext = os.path.splitext(pose_label_path)[1].lower()
    pose_dict = {}

    if ext == ".csv":
        with open(pose_label_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)

        start_idx = 0
        if len(rows) > 0 and len(rows[0]) >= 4:
            header = [x.strip().lower() for x in rows[0]]
            if "path" in header[0] and "yaw" in header[1] and "pitch" in header[2] and "roll" in header[3]:
                start_idx = 1

        for row in rows[start_idx:]:
            if len(row) < 4:
                continue
            rel_path = norm_rel_path(row[0].strip())
            yaw = float(row[1])
            pitch = float(row[2])
            roll = float(row[3])
            pose_dict[rel_path] = torch.tensor([yaw, pitch, roll], dtype=torch.float)

    elif ext == ".json":
        with open(pose_label_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        for k, v in raw.items():
            rel_path = norm_rel_path(k)
            if isinstance(v, dict):
                yaw = float(v["yaw"])
                pitch = float(v["pitch"])
                roll = float(v["roll"])
            else:
                yaw = float(v[0])
                pitch = float(v[1])
                roll = float(v[2])
            pose_dict[rel_path] = torch.tensor([yaw, pitch, roll], dtype=torch.float)

    else:
        with open(pose_label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if "," in line:
                parts = [x.strip() for x in line.split(",")]
            else:
                parts = line.split()

            if len(parts) < 4:
                continue

            rel_path = norm_rel_path(parts[0])
            yaw = float(parts[1])
            pitch = float(parts[2])
            roll = float(parts[3])
            pose_dict[rel_path] = torch.tensor([yaw, pitch, roll], dtype=torch.float)

    if len(pose_dict) == 0:
        raise ValueError(f"No valid pose pseudo-labels parsed from: {pose_label_path}")

    return pose_dict


class trainloader(Dataset):
    def __init__(self, dataset, sequence_length=2):
        self.data = edict()
        self.data.line = []
        self.data.root = dataset.image
        self.data.decode = Get_Decode(dataset.name)

        
        self.sequence_length = sequence_length

        if self.sequence_length != 2:
            print(f"[Warning] DGAGaze is designed for 2-frame input, but got sequence_length={self.sequence_length}")

        if isinstance(dataset.label, list):
            label_files = dataset.label
        elif os.path.isdir(dataset.label):
            label_files = [
                os.path.join(dataset.label, f)
                for f in os.listdir(dataset.label)
                if f.endswith(('.txt', '.csv', '.label'))
            ]
        else:
            label_files = [dataset.label]

        for label_file in label_files:
            with open(label_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            if dataset.header and len(lines) > 0:
                lines.pop(0)
            self.data.line.extend(lines)

        if len(self.data.line) == 0:
            raise ValueError("No label lines were read. Please check the file content and path.")

        # pose pseudo-labels generated by 6DRepNet
        self.pose_pseudo_path = getattr(dataset, "pose_label", None)
        self.pose_pseudo_dict = load_pose_pseudo_labels(self.pose_pseudo_path)

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.data.line)

    def _read_one(self, idx):
        line = self.data.line[idx].strip().split(" ")
        anno = self.data.decode(line)

        img_rel_path = norm_rel_path(anno.face)
        img_abs_path = os.path.normpath(os.path.join(self.data.root, anno.face))

        img = cv2.imread(img_abs_path)
        if img is None:
            raise ValueError(f"Image not found or cannot be read: {img_abs_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms:
            img = self.transforms(img)

        gaze_values = [float(v) for v in anno.gaze2d.split(",")]
        gaze_label = torch.tensor(gaze_values, dtype=torch.float)

        if img_rel_path not in self.pose_pseudo_dict:
            raise KeyError(
                f"Pose pseudo-label not found for image: {img_rel_path}. "
                f"Please ensure 6DRepNet pseudo-label file uses the same relative path as anno.face."
            )

        pose_label = self.pose_pseudo_dict[img_rel_path]  # [yaw, pitch, roll]

        return anno, img, gaze_label, pose_label, img_rel_path

    def __getitem__(self, idx):
        
        prev_idx = max(0, idx - 1)
        curr_idx = idx

        anno_prev, img_prev, _, pose_prev, _ = self._read_one(prev_idx)
        anno_curr, img_curr, gaze_curr, pose_curr, _ = self._read_one(curr_idx)

        img_sequence = torch.stack([img_prev, img_curr], dim=0)  

        data = edict()
        data.x_t = img_sequence
        data.name = anno_curr.name

        label = edict()
        label.gaze = gaze_curr          # supervise final gaze on current frame
        label.pose_prev = pose_prev     # pseudo-label for previous frame
        label.pose_curr = pose_curr     # pseudo-label for current frame

        return data, label


def loader(source, batch_size, shuffle=True, num_workers=8, sequence_length=2,
           device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    dataset = trainloader(source, sequence_length=sequence_length)
    print(f"-- [Read Data]: Source: {source.label}")
    print(f"-- [Read Data]: Pose pseudo-label source: {source.pose_label}")
    print(f"-- [Read Data]: Total num: {len(dataset)}")

    load = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return load