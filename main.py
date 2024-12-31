import os
import random
import shutil
def rename():
    data_dir="./data"
    for data_name in os.listdir(data_dir):
        print(data_name)
        data_path = os.path.join(data_dir, data_name)
        for idx,image_name in enumerate(os.listdir(data_path)):
                #建新的文件名，这里可以根据自己的需求来定义新的命名规则
                new_image_name = data_name +str(idx)+".jpg"
                # 重命名文件
                os.rename(os.path.join(data_path, image_name), os.path.join(data_path, new_image_name))

def split():
    data_dir="./data"
    train_dir="./data/train"
    test_dir="./data/valid"
    for subdir in os.listdir(data_dir):
        if subdir == "train" or subdir == "valid":
            continue
        subdir_path = os.path.join(data_dir, subdir)
        if os.path.isdir(subdir_path):
            # 获取子目录下所有的图片文件
            images = [f for f in os.listdir(subdir_path) if f.endswith('.jpg')]
            # 打乱顺序
            random.shuffle(images)
            # 计算划分的索引
            split_index = int(0.7 * len(images))
            # 划分训练集和测试集
            train_images = images[:split_index]
            test_images = images[split_index:]
            # 创建训练集和测试集的目录
            os.makedirs(os.path.join(train_dir, subdir), exist_ok=True)
            os.makedirs(os.path.join(test_dir, subdir), exist_ok=True)
            # 将图片复制到训练集和测试集目录
            for img in train_images:
                shutil.copy(os.path.join(subdir_path, img), os.path.join(train_dir, subdir, img))
            for img in test_images:
                shutil.copy(os.path.join(subdir_path, img), os.path.join(test_dir, subdir, img))
import torch
import torchvision.models as models

# 创建一个预训练的ResNet模型
model= models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

# 将模型保存到文件
torch.save(model, './dataresnet_model.pth')
