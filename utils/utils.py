
import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import csv
import os
import random
import h5py
import math

# def fast_hist(label_true, label_pred, n_class):
#     '''
#     :param label_true: 0 ~ n_class (batch, h, w)
#     :param label_pred: 0 ~ n_class (batch, h, w)
#     :param n_class: 类别数
#     :return: 对角线上是每一类分类正确的个数，其他都是分错的个数
#     '''
#
#     assert n_class > 1
#
#     mask = (label_true >= 0) & (label_true < n_class)
#     hist = torch.bincount(
#         n_class * label_true[mask].int() + label_pred[mask].int(),
#         minlength=n_class ** 2,
#     ).reshape(n_class, n_class)
#
#     return hist
#
# # 计算指标
# def cal_scores(hist, smooth=1):
#     TP = np.diag(hist)
#     FP = hist.sum(axis=0) - TP
#     FN = hist.sum(axis=1) - TP
#     TN = hist.sum() - TP - FP - FN
#     union = TP + FP + FN
#
#     dice = (2*TP+smooth) / (union+TP+smooth)
#
#     iou = (TP+smooth) / (union+smooth)
#
#     Precision = np.diag(hist).sum() / hist.sum()   # 分类正确的准确率  acc
#
#     Sensitivity = (TP+smooth) / (TP+FN+smooth)  # recall
#
#     Specificity = (TN+smooth) / (FP+TN+smooth)
#
#     return dice[1:]*100, iou[1:]*100, Precision*100, Sensitivity[1:]*100, Specificity[1:]*100
#


# 保存打印指标
def save_print_score(ed, file):
    ed = np.array(ed)

    test_mean = ["mean"]+[ed.mean()]
    test_std = ["std"]+[ed.std()]
    title = [' ', 'Euclidean distance']
    with open(file, "a") as f:
        w = csv.writer(f)
        w.writerow(["Test Mean Result"])
        w.writerow(title)
        w.writerow(test_mean)
        w.writerow(test_std)

    print("\n##############Test Result##############")
    print(f'Euclidean Distance: {ed.mean()}')



def heatmap2coordinate(image, crop_size, image_size):
    ratio_h = crop_size[0] / float(image_size[0])   # 高度比例
    ratio_w = crop_size[1] / float(image_size[1])   # 宽度比例
    ratio = min(ratio_h, ratio_w)
    h = int(image_size[1] *ratio)
    w = int(image_size[0] *ratio)

    image = image[:, 0:h, 0:w].unsqueeze(0)  # 裁剪掉padding
    image = F.interpolate(image, size=(image_size[1], image_size[0]), mode="bilinear", align_corners=True)
    image = image.squeeze()  # (h, w)

    p = torch.argmax(image.view(-1)).item()
    pred_h, pred_w = float(p // int(image_size[0])), float(p % int(image_size[0]))

    return torch.tensor([pred_w, pred_h])



# 从验证指标中选择最优的epoch
def best_model_in_fold(val_result, num_fold):
    best_epoch = 0
    best_score = 1e6
    for row in val_result:
        if str(num_fold) in row:
            if best_score > float(row[2]):
                best_score = float(row[2])
                best_epoch = int(row[1])
    return best_epoch


# 读取数据集目录内文件名，保存至csv文件
def get_dataset_filelist(data_root, save_file):
    file_list = os.listdir(data_root)
    random.shuffle(file_list)
    with open(save_file, 'w') as f:
        w = csv.writer(f)
        w.writerow(file_list)


def poly_learning_rate(args, optimizer, epoch):
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs(step = 30)
    """
    lr = args.lr * (1 - epoch / args.num_epochs) ** 0.9
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr




# one hot转成0,1,2,..这样的标签
def make_class_label(mask):
    b = mask.size()[0]
    mask = mask.view(b, -1)
    class_label = torch.max(mask, dim=-1)[0]
    return class_label


# 把0,1,2...这样的类别标签转化为one_hot
def make_one_hot(targets, num_classes):
    targets = targets.unsqueeze(1)
    label = []
    for i in range(num_classes):
        label.append((targets == i).float())
    label = torch.cat(label, dim=1)
    return label



# 保存训练过程中最大的checkpoint
class save_checkpoint_manager:
    def __init__(self, max_save=5):
        self.checkpoints = {}
        self.max_save = max_save

    def save(self, model, path, score):
        if len(self.checkpoints) < self.max_save:
            self.checkpoints[path] = score
            torch.save(model.state_dict(), path)
        else:
            if score > min(self.checkpoints.values()):
                os.remove(min(self.checkpoints))
                self.checkpoints.pop(min(self.checkpoints))
                self.checkpoints[path] = score
                torch.save(model.state_dict(), path)



def save_h5(train_data, train_label, val_data, filename):
    file = h5py.File(filename, 'w')
    # 写入
    file.create_dataset('train_data', data=train_data)
    file.create_dataset('train_label', data=train_label)
    file.create_dataset('val_data', data=val_data)
    file.close()


def load_h5(path):
    file = h5py.File(path, 'r')
    train_data = torch.tensor(np.array(file['train_data'][:]))
    train_label = torch.tensor(np.array(file['train_label'][:]))
    val_data = torch.tensor(np.array(file['val_data'][:]))
    file.close()
    return train_data, train_label, val_data



