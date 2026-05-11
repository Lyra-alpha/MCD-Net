import os
import random

import numpy as np
from PIL import Image
from tqdm import tqdm

#-------------------------------------------------------#
#   直接设置训练集、验证集、测试集的比例
#-------------------------------------------------------#
train_ratio = 0.8
val_ratio   = 0.1
test_ratio  = 0.1   # 注意三者之和应为 1.0
#-------------------------------------------------------#
#   指向VOC数据集所在的文件夹
#-------------------------------------------------------#
VOCdevkit_path = 'dataset'

if __name__ == "__main__":
    random.seed(0)
    print("Generate txt in ImageSets.")
    segfilepath  = os.path.join(VOCdevkit_path, 'Moraine_dataset/SegmentationClass')
    saveBasePath = os.path.join(VOCdevkit_path, 'Moraine_dataset/ImageSets/Segmentation')

    # 获取所有 .png 标签文件
    temp_seg = [f for f in os.listdir(segfilepath) if f.endswith(".png")]
    total_seg = temp_seg
    num = len(total_seg)

    # 按比例计算各集合数量
    num_train = int(num * train_ratio)
    num_val   = int(num * val_ratio)
    num_test  = num - num_train - num_val   # 剩余的给测试集

    print(f"Total images: {num}")
    print(f"Train size : {num_train}")
    print(f"Val size   : {num_val}")
    print(f"Test size  : {num_test}")

    # 随机打乱索引
    indices = list(range(num))
    random.shuffle(indices)

    train_indices = indices[:num_train]
    val_indices   = indices[num_train:num_train + num_val]
    test_indices  = indices[num_train + num_val:]

    # 打开文件
    ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
    fval   = open(os.path.join(saveBasePath, 'val.txt'), 'w')
    ftest  = open(os.path.join(saveBasePath, 'test.txt'), 'w')
    fall   = open(os.path.join(saveBasePath, 'all.txt'), 'w')

    # 写入各文件
    for i, name in enumerate(total_seg):
        name_no_ext = name[:-4] + '\n'
        fall.write(name_no_ext)                     # 所有图片都写入 all.txt
        if i in train_indices:
            ftrain.write(name_no_ext)
        elif i in val_indices:
            fval.write(name_no_ext)
        else:
            ftest.write(name_no_ext)

    ftrain.close()
    fval.close()
    ftest.close()
    fall.close()

    print("Generate txt in ImageSets done.")

    # ------------------- 数据集格式检查 -------------------
    print("Checking dataset format, this may take a while.")
    classes_nums = np.zeros([256], int)
    for i in tqdm(range(num)):
        name = total_seg[i]
        png_file_name = os.path.join(segfilepath, name)
        if not os.path.exists(png_file_name):
            raise ValueError(f"Label image {png_file_name} not found. Please check the path and file extension (should be .png).")

        png = np.array(Image.open(png_file_name), np.uint8)
        if len(np.shape(png)) > 2:
            print(f"Label image {name} has shape {str(np.shape(png))}, which is not a grayscale or 8-bit color image. Please check the dataset format.")
            print("Label images must be grayscale or 8-bit color images, where each pixel value represents the class index.")

        classes_nums += np.bincount(np.reshape(png, [-1]), minlength=256)

    print("Pixel value and its count:")
    print('-' * 37)
    print("| %15s | %15s |" % ("Key", "Value"))
    print('-' * 37)
    for i in range(256):
        if classes_nums[i] > 0:
            print("| %15s | %15s |" % (str(i), str(classes_nums[i])))
            print('-' * 37)

    if classes_nums[255] > 0 and classes_nums[0] > 0 and np.sum(classes_nums[1:255]) == 0:
        print("Detected only pixel values 0 and 255 in the label images. Format error.")
        print("For binary classification, background should be 0 and target should be 1.")
    elif classes_nums[0] > 0 and np.sum(classes_nums[1:]) == 0:
        print("Detected only background pixels in the label images. Format error. Please check the dataset.")

    print("Images in JPEGImages should be .jpg files, and images in SegmentationClass should be .png files.")
