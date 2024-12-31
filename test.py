import torch
from torchvision import models, transforms
import torch.nn as nn
import cv2

classes = ["小叶和小林", "小林","其他人","小叶"]

transf = transforms.ToTensor()
device = torch.device('cuda:0')
num_classes = 4
model_path = "./1/_model_91.pt"
image_input = cv2.imread("./data/lin/lin55.jpg")
image_input = transf(image_input)
image_input = torch.unsqueeze(image_input, dim=0).cuda()
# 搭建模型
resnet50 = models.resnet50()
for param in resnet50.parameters():
    param.requires_grad = False

fc_inputs = resnet50.fc.in_features
resnet50.fc = nn.Sequential(
    nn.Linear(fc_inputs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, num_classes),
    nn.LogSoftmax(dim=1)
)
resnet50 = torch.load(model_path)

outputs = resnet50(image_input)
value, id = torch.max(outputs, 1)
print(outputs, "\n", "result is：", classes[id])