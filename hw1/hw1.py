import torch
from PIL import Image

def load_image(path):
    img = Image.open(path).convert("RGB")
    img_tensor = torch.tensor(list(img.getdata()), dtype=torch.float32) / 255.0
    return img_tensor, img.size     # 这里是 宽*高

def build_poly_features(x):
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    features = torch.cat([
        torch.ones_like(r),
        r, g, b,
        r**2, g**2, b**2,
        r*g, r*b, g*b,
        r*g*b
    ], dim=1)
    return features

if __name__ == "__main__":
    input_path = "./input.png"
    output_path = "./output.png"
    test_path = "./test.png"

    X_rgb, input_size = load_image(input_path)
    Y_rgb, output_size = load_image(output_path)
    assert input_size == output_size, "Input and output images must have the same size"

    X = build_poly_features(X_rgb)
    Y = Y_rgb

    # W = (X^T X)^(-1) X^T Y
    X_T = X.t()
    X_T_X = X_T @ X
    X_T_Y = X_T @ Y
    W = torch.linalg.pinv(X_T_X) @ X_T_Y

    test_rgb, test_size = load_image(test_path)
    X_test = build_poly_features(test_rgb)
    Y_pred = X_test @ W
    Y_pred = Y_pred.clamp(0, 1)

    pred_img = Y_pred.reshape(test_size[1], test_size[0], 3)    # 这里要 高*宽*通道数, 所以顺序是反的
    pred_img = (pred_img * 255).to(torch.uint8).numpy()
    pred_pil = Image.fromarray(pred_img)
    pred_pil.save("./pred.png")

