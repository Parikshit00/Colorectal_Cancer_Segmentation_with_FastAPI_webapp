import io
import torch
import base64
import datetime
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import json
from train_metapolyp.model import custom_model
import numpy as np
import sys
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = torch.device("cpu")
def load_model(model_path):
    model = custom_model(256,1).to(device)
    print("Model allocated on:", next(model.parameters()).device)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    state_dict = {k.replace("model.", ""): v.to(device) for k, v in checkpoint['state_dict'].items()} 
    model.load_state_dict(state_dict)
    return model

# model 
model = load_model("train_metapolyp/checkpoints/best_model.ckpt")
model.eval()

def mask_parse(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return (mask * 255).astype(np.uint8)

def transform_image(image_bytes):
    my_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).to(device)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    tensor = torch.unsqueeze(tensor, dim=0)
    output = model(tensor)
    return output.detach().numpy()

def get_result(image_file,is_api = False):
    start_time = datetime.datetime.now()
    image_bytes = image_file.file.read()
    #mask = get_prediction(image_bytes)
    #out = mask_parse(mask)
    out = "hi"
    out = out.tobytes()
    print(type(image_bytes))
    print(sys.getsizeof(image_bytes))
    print(type(out))
    print(sys.getsizeof(out))
    end_time = datetime.datetime.now()
    time_diff = (end_time - start_time)
    execution_time = f'{round(time_diff.total_seconds() * 1000)} ms'
    input_encoded_string = base64.b64encode(image_bytes)
    ibs64 = input_encoded_string.decode('utf-8')
    mask_encoded_string = base64.b64encode(out)
    mbs64 = mask_encoded_string.decode('utf-8')
    input_image_data = f'data:image/jpeg;base64,{ibs64}'
    mask_image_data = f'data:image/jpeg;base64,{mbs64}'
    result = {
        "inference_time":execution_time,
    }
    if not is_api: 
        result["image_data"]= input_image_data
        result["mask_data"]= mask_image_data
    return result