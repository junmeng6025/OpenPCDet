import torch
from torch import Tensor
import matplotlib.pyplot as plt


def vis_hist(ten: Tensor):
    numpy_array = ten.numpy()

    # 使用Matplotlib绘制直方图
    plt.hist(numpy_array, bins=50, alpha=0.7, color='blue', edgecolor='black')

    # 添加标题和标签
    plt.title('Histogram of PyTorch Tensor')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # 显示图形
    plt.show()

