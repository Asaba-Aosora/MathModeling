import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os

def load_img(img_path, target_size):
    img = Image.open(img_path).convert('RGB')
    origin_size = img.size
    trans = T.Compose([
        T.Resize(target_size),
        T.ToTensor(),
        T.Normalize(0.5, 0.5)   # [0, 1] -> [-1, 1]
    ])
    tensor = trans(img).unsqueeze(0)
    return tensor, origin_size


def save_img(tensor, img_path, origin_size=None):
    tensor = (tensor + 1) / 2  # [-1, 1] -> [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    img = T.ToPILImage()(tensor.squeeze(0))

    if origin_size is not None:
        img = img.resize(origin_size, Image.Resampling.LANCZOS)

    img.save(img_path)

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU()
        )
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, out_ch),
            nn.ReLU()
        )
        if in_ch < out_ch:
            self.down = nn.Conv2d(out_ch, out_ch, 2, stride=2)
        else:
            self.down = nn.Identity()

    def forward(self, x, t_emb):
        conv_out = self.conv(x) + self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        return conv_out, self.down(conv_out)


class FlowUNet(nn.Module):
    def __init__(self, in_ch=3, time_dim=32):
        super().__init__()
        self.time_emb = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim)
        )
        self.down1 = ConvBlock(in_ch, 64, time_dim)
        self.down2 = ConvBlock(64, 128, time_dim)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU()
        )
        self.up2 = ConvBlock(256 + 128, 128, time_dim)
        self.up1 = ConvBlock(128 + 64, 64, time_dim)
        self.out = nn.Conv2d(64, in_ch, 3, 1, padding=1)

    
    def forward(self, x, t):
        t_emb = self.time_emb(t)
        h1, x1 = self.down1(x, t_emb)
        h2, x2 = self.down2(x1, t_emb)
        bn = self.bottleneck(x2)
        bn_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(bn)
        up2, _ = self.up2(torch.cat([bn_up, h2], dim=1), t_emb)
        up2_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(up2)
        up1, _ = self.up1(torch.cat([up2_up, h1], dim=1), t_emb)
        out = self.out(up1)
        return out

def infer(model, x_init, t_start, t_end, device, num_steps=100):
    model.eval()
    x = x_init.to(device)
    dt = (t_end - t_start) / num_steps
    with torch.no_grad():
        for i in range(num_steps):
            t_curr = torch.tensor([[t_start + i * dt]], device=device)
            v_pred = model(x, t_curr)
            x = x + v_pred * dt
    return x

def main():
    # 设置参数
    input_img_path = 'input.png'
    output_img_path = 'output.png'
    model_save_path = 'model.pth'
    test_input_img_path = 'test_input.png'
    test_output_img_path = 'test_output.jpg'
    test_input_result_img_path = 'test_input_result.png'
    test_output_result_img_path = 'test_output_result.png'
    target_size = (256, 256)
    epochs = 1000
    lr = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载图像
    x0, _ = load_img(input_img_path, target_size)
    x1, _ = load_img(output_img_path, target_size)
    x0 = x0.to(device)
    x1 = x1.to(device)
    test_x0, test_x0_origin_size = load_img(test_input_img_path, target_size)
    test_x1, test_x1_origin_size = load_img(test_output_img_path, target_size)


    # 初始化模型
    model = FlowUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()  # 均方误差损失

    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        print("成功加载本地模型")
    else:
        print("未找到本地模型，开始训练")
        for epoch in range(epochs):
            model.train()
            t = torch.rand(1, 1, device=device)
            x_t = (1-t) * x0 + t * x1
            v_pred = model(x_t, t)
            loss = loss_fn(v_pred, x1-x0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch+1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

        torch.save(model.state_dict(), model_save_path)

    # 生成结果
    input_result = infer(model, test_x0, t_start=0.0, t_end=1.0, device=device)
    output_result = infer(model, test_x1, t_start=1.0, t_end=0.0, device=device)

    # 保存结果
    save_img(input_result, test_input_result_img_path, test_x0_origin_size)
    save_img(output_result, test_output_result_img_path, test_x1_origin_size)


if __name__ == '__main__':
    main()