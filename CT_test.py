import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from PIL import Image
from torchvision import transforms, models
import time
import os

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor()
])
right = []
device = torch.device("cuda")
model = models.shufflenet_v2_x1_0()
# model.classifier = torch.nn.Sequential(
#     torch.nn.Linear(1000, 512),
#     torch.nn.ReLU(),
#     torch.nn.Dropout(p=0.5),
#     torch.nn.Linear(512, 3)
# )
model.load_state_dict(torch.load('ConvNext.pth'))
print(model)
model.eval().cuda()
label = np.array(['A', 'B', 'C'])


def predict(image_path, name):
    # 打开图片
    img = Image.open(image_path)
    # 数据处理，增加一个维度
    img = transform(img).unsqueeze(0)
    # 预测得到结果
    inputs = Variable(img.cuda())
    outputs = model(inputs)
    # 获得最大值所在位置
    _, predicted = torch.max(outputs, 1)
    if predicted.item() > len(label):
        return 0
        # 转化为类别名称
    else:
        print("pred:", label[predicted.item()], end=",")
    if (label[predicted.item()]) == name:
        right.append(1)


classes = 'C'
img_path = 'X-G/'
files = os.listdir(img_path + classes + '/')
# print(files)
time1 = time.time()
for file_name in files:
    predict(img_path + classes + '/' + file_name, classes)
    print("real:", file_name)
print("\n测试数：%d\n正确率:%f\n正确数：%d\n" % (len(files), len(right) / len(files), len(right)))
print("耗时：%.2fs" % (time.time() - time1))
