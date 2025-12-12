import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

data = loadmat("faces_data.mat")    # data 是一个字典
faces = data["faces"]   # 拿出faces键, 里面是numpy数组表示的图像, 默认uint8
faces = torch.tensor(faces, dtype=torch.float32).T    # 为进行张量计算, 要转成float, 且注意转置, 让每一行是一个样本
img_h, img_w = 60, 43

mean = faces.mean(dim=0, keepdim=True)
faces_centered = faces - mean
faces_non_centered = faces

def PCA_process(data):
    n = data.shape[0]
    cov_matrix = (data.T @ data) / (n-1)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
    sort_indices = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[sort_indices]
    eigenvectors = eigenvectors[:, sort_indices]    # 特征向量是按列的, 所以排序也要按列排
    return eigenvalues, eigenvectors

eigenvalues1, eigenvectors1 = PCA_process(faces_centered)
eigenvalues2, eigenvectors2 = PCA_process(faces_non_centered)

def plot_eigenfaces(eigenvectors, title_prefix):
    plt.figure(figsize=(10, 10))
    for i in range(10):
        eigenface = eigenvectors[:, i].reshape(img_h, img_w)
        plt.subplot(2, 5, i+1)
        plt.imshow(eigenface, cmap='gray')
        plt.title(f'{title_prefix}\npic{i+1}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

plot_eigenfaces(eigenvectors1, 'Centered')
plot_eigenfaces(eigenvectors2, 'Non-centered')
