import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
from PIL import Image
import pandas as pd

densnet = pd.read_csv('csv/densnet.csv', index_col=None)
googlenet = pd.read_csv('csv/googlenet.csv', index_col=None)
resnet = pd.read_csv('csv/resnet.csv', index_col=None)
shufflenet_att = pd.read_csv('csv/shufflenetv2_att.csv', index_col=None)
shufflenet = pd.read_csv('csv/shufflenetv2.csv', index_col=None)
mobilenet = pd.read_csv('csv/mobilenet.csv', index_col=None)
DensNet101 = densnet.values[:, 1:3]
GoogleNet = googlenet.values[:, 1:3]
ResNet50 = resnet.values[:, 1:3]
ShuffleNetv2_att = shufflenet_att.values[:, 1:3]
ShuffleNetv2 = shufflenet.values[:, 1:3]
MobileNet = mobilenet.values[:, 1:3]
dens = []
res = []
google = []
shuff_att = []
shuff = []
mobi = []
for i in range(30):
    if i % 4 == 0:
        dens.append(DensNet101[i, :])
        res.append(ResNet50[i, :])
        google.append(GoogleNet[i, :])
        shuff.append(ShuffleNetv2[i, :])
        shuff_att.append(ShuffleNetv2_att[i, :])
        mobi.append(MobileNet[i, :])
dens = np.array(dens)
res = np.array(res)
google = np.array(google)
shuff = np.array(shuff)
shuff_att = np.array(shuff_att)
mobi = np.array(mobi)
myparams = {

    'axes.labelsize': '10',

    'xtick.labelsize': '10',

    'ytick.labelsize': '10',

    'lines.linewidth': 1,

    'legend.fontsize': '10',

    'font.family': 'Times New Roman',

    'figure.figsize': '7, 5'  # 图片尺寸

}

pylab.rcParams.update(myparams)  # 更新自己的设置
x = np.arange(1, 31, 4)
print(x)
fig1 = plt.figure(1)
plt.grid()
plt.plot(x, dens[:, 0], linewidth=2, label='DensNet101', marker='o',
         markersize=6)
plt.plot(x, res[:, 0], linewidth=2, label='ResNet50', color='b', marker='x',
         markersize=6)
plt.plot(x, mobi[:, 0], linewidth=2, label='MobileNet', marker='s',
         markersize=6)
plt.plot(x, shuff[:, 0], linewidth=2, label='ShuffleNet', marker='v',
         markersize=6)
plt.plot(x, shuff_att[:, 0], linewidth=2, label='ShuffleNet-Att', color='r', marker='*',
         markersize=6)
axes1 = plt.gca()
axes1.grid(True)  # add grid

plt.legend(loc="lower right")  # 图例位置 右下角
plt.ylabel('Acc')
plt.xlabel('Epoch')
# plt.savefig("Acc.pdf", dpi=72, format='pdf')
plt.show()
