import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


class Encoder(nn.Module):
    def __init__(self, original_dim, intermediate_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(original_dim, intermediate_dim)
        self.fc_mu = nn.Linear(intermediate_dim, latent_dim)
        self.fc_logvar = nn.Linear(intermediate_dim, latent_dim)    # log(\delta^2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, original_dim, intermediate_dim, latent_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, original_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        z = self.relu(self.fc1(z))
        x_recon = self.sigmoid(self.fc2(z))
        return x_recon

class VAE(nn.Module):
    def __init__(self, original_dim, intermediate_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(original_dim, intermediate_dim, latent_dim)
        self.decoder = Decoder(original_dim, intermediate_dim, latent_dim)

    def reparameterize(self, mu, logvar):
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)
    
    def forward(self, imgs):
        imgs_flat = imgs.view(-1, 28*28)
        mu, logvar = self.encoder(imgs_flat)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)

        recon_loss = nn.functional.binary_cross_entropy(x_recon, imgs_flat, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = (recon_loss + kl_loss) / imgs.size(0)
        return loss

    def generate(self, z):
        return self.decoder(z)


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
            loss = model(imgs)      # 这里是自动触发 forward 了
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
