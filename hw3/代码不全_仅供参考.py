import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


#class Encoder(nn.Module):

#class Decoder(nn.Module):

#class VAE(nn.Module):


def main():
    # 超参数
    original_dim = 28 * 28        # MNIST 28x28
    intermediate_dim = 256
    latent_dim = 2                # 2 维便于可视化
    batch_size = 1000             # MNIST 60000 可以被 1000 整除
    num_epochs = 50
    learning_rate = 0.01
    weight_decay = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据集：MNIST
    transform = transforms.ToTensor()  # 输出 [0,1]
    train_dataset = datasets.MNIST(root="./data",
                                   train=True,
                                   transform=transform,
                                   download=True)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)

    # 初始化模型与优化器
    model = VAE(original_dim, intermediate_dim, latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=learning_rate,
                                  weight_decay=weight_decay)

    # 训练
    model.train()
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        num_samples = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)

            optimizer.zero_grad()
            loss = model(imgs)
            loss.backward()
            optimizer.step()

            bs = imgs.size(0)
            epoch_loss += loss.item() * bs
            num_samples += bs

        avg_loss = epoch_loss / num_samples
        print(f"Epoch [{epoch:02d}/{num_epochs:02d}]  Loss: {avg_loss:.4f}")

    # 训练结束，做隐空间可视化
    model.eval()
    with torch.no_grad():
        # 在 (0,0) 附近取一个 11x11 的网格点，这里取 [-3,3] 区间
        n = 11
        grid_x = torch.linspace(-3, 3, n)
        grid_y = torch.linspace(-3, 3, n)

        # 大画布：把 11x11 张 28x28 的图粘在一起
        figure = np.zeros((28 * n, 28 * n), dtype=np.float32)

        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                # 构造当前的 2D 隐向量 (xi, yi)
                z = torch.tensor([[xi.item(), yi.item()]], dtype=torch.float32, device=device)
                x_decoded = model.generate(z)
                digit = x_decoded[0].cpu().numpy().reshape(28, 28)
                # 简单裁剪到 [0,1]，避免数值溢出影响展示
                digit = np.clip(digit, 0.0, 1.0)

                row_start = i * 28
                row_end = row_start + 28
                col_start = j * 28
                col_end = col_start + 28
                figure[row_start:row_end, col_start:col_end] = digit

        plt.figure(figsize=(8, 8))
        plt.imshow(figure, cmap="gray")
        plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
