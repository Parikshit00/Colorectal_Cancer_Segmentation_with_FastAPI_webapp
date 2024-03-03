import torch
import torchvision.transforms as transforms
from model import custom_model
from PIL import Image
import numpy as np
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

def load_model(model_path):
    model = custom_model(256,1).to(device)
    print("Model allocated on:", next(model.parameters()).device)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    state_dict = {k.replace("model.", ""): v.to(device) for k, v in checkpoint['state_dict'].items()} 
    model.load_state_dict(state_dict)
    model.eval()
    return model

def predict_single(model, img_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    img = Image.open(img_path).convert("RGB")
    img = transform(img).to(device)
    print("Image allocated on:", img.device)  # Print allocation device
    img = torch.unsqueeze(img, dim=0)

    result = model(img)
    return result.detach().numpy()

def mask_parse(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return (mask * 255).astype(np.uint8)

if __name__ == "__main__":
    save_path = "checkpoints/best_model.ckpt"
    img_in = "test.jpg"
    img_out = "out1.jpg"
    model = load_model(save_path)
    masks = predict_single(model, img_in)
    out = mask_parse(masks)
    Image.fromarray(out).save(img_out)
