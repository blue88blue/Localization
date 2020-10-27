import torch
import os
from torch.utils.data import Dataset
import csv
from .transform import*
import numpy as np


class myDataset(Dataset):
    def __init__(self, data_root, target_root, crop_size, data_mode, k_fold=None, imagefile_csv=None, num_fold=None):
        super().__init__()
        self.crop_size = crop_size  # h, w
        self.data_root = data_root
        self.target_root = target_root
        self.data_mode = data_mode
        # 若不交叉验证，直接读取data_root下文件列表
        if k_fold == None:
            self.image_files = os.listdir(data_root)
            print(f"{data_mode} dataset: {len(self.image_files)}")
        # 交叉验证：传入包含所有数据集文件名的csv， 根据本次折数num_fold获取文件名列表
        else:
            with open(imagefile_csv, "r") as f:
                reader = csv.reader(f)
                image_files = list(reader)[0]
            fold_size = len(image_files) // k_fold  # 等分
            fold = num_fold - 1
            if data_mode == "train":
                self.image_files = image_files[0: fold*fold_size] + image_files[fold*fold_size+fold_size:]
            elif data_mode == "val" or data_mode == "test":
                self.image_files = image_files[fold*fold_size: fold*fold_size+fold_size]
            else:
                raise NotImplementedError
            print(f"{data_mode} dataset fold{num_fold}/{k_fold}: {len(self.image_files)} images")

        # 热力图
        self.heatmap = self.generate_heatmap(6)
        with open(target_root, "r") as f:  # 坐标金标准csv文件 : 文件名 + 坐标（x, y）
            reader = csv.reader(f)
            self.label_file = list(reader)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        file = self.image_files[idx]
        file_name, _ = os.path.splitext(file)

        image_path = os.path.join(self.data_root, file)
        image, _ = fetch(image_path)

        if self.data_mode == "train":  # 数据增强
            image, _ = random_transfrom(image)

        image, _ = convert_to_tensor(image)
        image_size = image.size()[-2:]
        image, _, ratio = scale_adaptive(self.crop_size, image)
        label, coordinate = self.build_label(file, image.size()[-2:], ratio)

        if self.data_mode == "train":  # 数据增强
            image, label = random_Top_Bottom_filp(image, label)
            image, label = random_Left_Right_filp(image, label)

        label = label.squeeze()
        image, label = pad(self.crop_size, image, label)
        label = label.unsqueeze(0)

        return {"image": image,
                "image_size": torch.tensor(image_size),
                "label": label,
                "file": file,
                "GTcoordinate": coordinate}

    def generate_heatmap(self, sigma):
        heatmap = torch.ones((201, 201))  # h, w
        for i in range(201):
            for j in range(201):
                x = i-101
                y = j-101
                dist = x**2+y**2
                heatmap[i, j] = np.exp(-0.5*dist/(sigma**2))
                if heatmap[i, j] < 0.01:
                    heatmap[i, j] = 0
        return heatmap

    def build_label(self, image_filename, size, ratio):
        image_h = size[0]
        image_w = size[1]
        x, y = 0, 0
        GTx, GTy =0, 0
        for row in self.label_file:
            if image_filename in row:
                GTx = float(row[3])
                GTy = float(row[4])
                x = int(GTx*ratio)
                y = int(GTy*ratio)
                break

        label = torch.zeros((image_h, image_w))

        if x != 0 or y != 0:
            right = min(x + 101, image_w)
            left = max(x - 100, 0)
            top = max(y - 100, 0)
            bottom = min(y + 101, image_h)
            label[top:bottom, left:right] = self.heatmap[0:bottom - top, 0:right - left]

        label.unsqueeze(0)
        return label, torch.tensor([GTx, GTy])







class PredictDataset(Dataset):
    def __init__(self, data_root, crop_size):
        super(PredictDataset, self).__init__()
        self.data_root = data_root
        self.crop_size = crop_size

        self.files = os.listdir(data_root)
        self.files = sorted(self.files)
        print(f"pred dataset: {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name, _ = os.path.splitext(self.files[idx])
        image_path = os.path.join(self.data_root, self.files[idx])

        image, _ = fetch(image_path)
        image_size = image.size  # w,h
        image, _ = convert_to_tensor(image)
        image, _, _ = scale_adaptive(self.crop_size, image)
        image, _ = pad(self.crop_size, image)

        return {"image": image,
                "file_name": file_name,
                "image_size": torch.tensor(image_size)}







