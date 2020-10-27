from dataset.dataset import PredictDataset
from dataset.transform import scale_adaptive
from torch.utils.data import DataLoader
import torch
import os
from PIL import Image
from torch.nn import functional as F
from settings import basic_setting
import numpy as np
import csv
from tqdm import tqdm
from torchvision.utils import save_image
import math
#models
from model.choose_model import seg_model


# #################################### predict settings 预测提交结果 ####################################
pred_data_root = '/media/sjh/disk1T/dataset/MICCAI2018/Refuge2_Validation'
pred_dir = "fovea_location_results.csv"
model_dir = "/home/sjh/Project/Localization/runs/REFUGE/2020-1027-0922_10_CENet__fold_3/checkpoints/fold_1/CP_epoch10.pth"
# #################################### predict settings 预测提交结果 ####################################


def pred(model, device, args):
    dataset_pred = PredictDataset(pred_data_root, args.crop_size)
    num_data = len(dataset_pred)
    dataloader_pred = DataLoader(dataset_pred, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers,pin_memory=True, drop_last=True)
    with open(pred_dir, "w") as f:
        w = csv.writer(f)
        w.writerow(['ImageName', 'Fovea_X', 'Fovea_Y'])
    model.eval()
    with tqdm(total=num_data, desc=f'predict', unit='img') as pbar:
        for batch in dataloader_pred:

            image = batch["image"]
            file_name = batch["file_name"]
            image_size = batch["image_size"]
            image = image.to(device, dtype=torch.float32)

            with torch.no_grad():
                outputs = model(image)
                pred = outputs["main_out"]
                pred = torch.sigmoid(pred).cpu()


            # 保存预测结果
            for i in range(image.shape[0]):
                save_image(image[i,:,:,:], args.plot_save_dir + "/"+file_name[i]+".jpg")
                recover_sigle_image_size_and_save(pred[i, :, :, :], args.crop_size, image_size[i], file_name[i])

            pbar.update(image.shape[0])


# 将预测tensor恢复到原图大小
def recover_sigle_image_size_and_save(image, crop_size, image_size, file_name):

    ratio_h = crop_size[0] / float(image_size[1])   # 高度比例
    ratio_w = crop_size[1] / float(image_size[0])   # 宽度比例
    ratio = min(ratio_h, ratio_w)
    h = int(image_size[1] *ratio)
    w = int(image_size[0] *ratio)

    # 裁剪去掉pad
    image = image[:, 0:h, 0:w].unsqueeze(0)
    image = F.interpolate(image, size=(image_size[1], image_size[0]), mode="bilinear", align_corners=True)
    image = image.squeeze()  # (h, w)

    p = torch.argmax(image.view(-1)).item()
    pred_h, pred_w = float(p // int(image_size[0])), float(p % int(image_size[0]))
    if math.fabs(pred_w-pred_h) > 2000 or min(pred_h, pred_w) < 200:
        pred_w = image_size[0].item()//2
        pred_h = image_size[1].item()//2

    # 预测结果写入CSV文件
    with open(pred_dir, "a") as f:
        w = csv.writer(f)
        w.writerow([file_name+".jpg", pred_w, pred_h])

    image = image.expand((3, -1, -1))
    image[2, int(pred_h), int(pred_w)] = 255
    save_image(image, args.plot_save_dir + "/"+file_name+".png")




if __name__ == "__main__":
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = basic_setting()
    pred_dir = os.path.join(args.dir, pred_dir)

    # 模型选择
    model = seg_model(args)
    model.to(device)
    model.load_state_dict(torch.load(model_dir, map_location=device))
    print("model loaded!")

    pred(model, device, args)