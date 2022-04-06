import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from scipy import interp
import os
import torch
from ConvNext import convnext_tiny
import torch.nn as nn
from torch.autograd import Variable
from PIL import Image
from torchvision import transforms, models
import time
from sklearn.metrics import confusion_matrix
import shufflenet

label = np.array(['A', 'B', 'C'])

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    # transforms.Grayscale(),  # 灰度图
    transforms.ToTensor()
])
device = torch.device("cuda")
model = shufflenet.ShuffleNetv2_Att()
# model = models.shufflenet_v2_x0_5()
# model = models.googlenet(progress=True)
# model.fc = torch.nn.Sequential(
#     torch.nn.Linear(1024, 512),
#     # torch.nn.ReLU(),
#     torch.nn.Dropout(p=0.5),
#     torch.nn.Linear(512, 3)
# )
model.load_state_dict(torch.load('shufflenetv2_att.pth'))
print(model)
model.eval().cuda()
predictions_labels = []
# y_label = np.array([
#     [1, 0, 0], [1, 0, 0], [1, 0, 0],
#     [0, 1, 0], [0, 1, 0], [0, 1, 0],
#     [0, 0, 1], [0, 0, 1], [0, 0, 1]
# ])
# print(y_label)
# print(np.append(y_label, [[2, 0, 0]], axis=0))
# y_score = np.array([
#     [0.8, 0.1, 0.1], [0.2, 0.32, 0.48], [0.6, 0.1, 0.3],
#     [0.2, 0.5, 0.3], [0.1, 0.6, 0.3], [0.2, 0.75, 0.05],
#     [0.05, 0.05, 0.9], [0.1, 0.3, 0.6], [0.12, 0.8, 0.08],
# ])
y_score = np.array([
    [0.8, 0.1, 0.1]
])
y_label = np.array([
    [1, 0, 0]
])


def predict(image_path, char):
    global y_label
    global y_score
    img = Image.open(image_path)
    img = transform(img).unsqueeze(0)
    inputs = Variable(img.cuda())
    outputs = model(inputs)
    _, predicted = torch.max(outputs, 1)
    # print(predicted.item())
    # print(outputs[0].data.cpu().numpy().shape)
    y_score = np.append(y_score, [outputs[0].data.cpu().numpy()], axis=0)
    if char == 'A':
        y_label = np.append(y_label, [[1, 0, 0]], axis=0)
    if char == 'B':
        y_label = np.append(y_label, [[0, 1, 0]], axis=0)
    if char == 'C':
        y_label = np.append(y_label, [[0, 0, 1]], axis=0)
    # if predicted.item() == 0:
    #     y_label = np.append(y_label, [[1, 0, 0]], axis=0)
    # elif predicted.item() == 1:
    #     y_label = np.append(y_label, [[0, 1, 0]], axis=0)
    # elif predicted.item() == 2:
    #     y_label = np.append(y_label, [[0, 0, 1]], axis=0)

    # print(outputs[0][predicted.item()])
    # if predicted.item() > len(label):
    #     return 0
    #     # 转化为类别名称
    # else:
    #     predictions_labels.append(label[predicted.item()])
    # print("pred:", label[predicted.item()], end=",")
    # if (label[predicted.item()]) == name:
    #     right.append(1)


time1 = time.time()
X = []  # 定义图像名称
Y = []  # 定义图像分类类标
for i in range(65, 68):
    # 遍历文件夹，读取图片
    for f in os.listdir("X-G/" + chr(i)):
        # 获取图像名称
        X.append("X-G/" + chr(i) + "/" + str(f))
        predict("X-G/" + chr(i) + "/" + str(f), chr(i))
        # 获取图像类标即为文件夹名称
        Y.append(chr(i))
print(y_label.shape)
print(y_score.shape)
# print(y_label.shape)
print("耗时：", time.time() - time1)
# print(X)
# print(Y)
# print(y_score)
n_classes = 3
label_class = ["A", "B", "C"]
# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    print(y_label[:, i], y_score[:, i])
    fpr[i], tpr[i], _ = roc_curve(y_label[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# micro（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_label.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# macro（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw = 2
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
                   ''.format(label_class[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('multi-calss ROC')
plt.legend(loc="lower right")
plt.savefig('ROC.pdf', format='pdf')
plt.show()
