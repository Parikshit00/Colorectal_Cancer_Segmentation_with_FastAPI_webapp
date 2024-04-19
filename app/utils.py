
import io
import torch
import base64
import datetime
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image, ImageDraw
import json
from model import custom_model
import numpy as np
import sys
import cv2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#device = torch.device("cpu")
def load_model(model_path):
    model = custom_model(256,1).to(device)
    print("Model allocated on:", next(model.parameters()).device)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = {k.replace("model.", ""): v.to(device) for k, v in checkpoint['state_dict'].items()} 
    model.load_state_dict(state_dict)
    return model

def cv2_to_pil_box(cv2_box):
    x, y, w, h = cv2_box
    return x, y, x + w, y + h

def pil_to_bytes(image_pil):
    # Create a BytesIO object to hold the encoded image
    with io.BytesIO() as output_bytes:
        # Save the image to the BytesIO object
        image_pil.save(output_bytes, format='JPEG')
        # Get the value of the BytesIO object (encoded image bytes)
        bytes_data = output_bytes.getvalue()
    return bytes_data

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


def get_prediction(model, image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    tensor = torch.unsqueeze(tensor, dim=0)
    output = model(tensor)
    return output.detach().numpy()

def get_result(image_file,is_api = False):
    # model
    model = load_model("weights/best_model.ckpt")
    model.eval()
    start_time = datetime.datetime.now()
    image_bytes = image_file.file.read()
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize((256,256))
    mask = get_prediction(model, image_bytes)
    out = mask_parse(mask)
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    #make mask smooth
    gray_mask = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    smoothed_mask = cv2.GaussianBlur(gray_mask, (5, 5), 0)
    kernel = np.ones((5, 5), np.uint8)
    for i in range(2):
        smoothed_mask = cv2.morphologyEx(smoothed_mask, cv2.MORPH_CLOSE, kernel)
    _, smoothed_mask = cv2.threshold(smoothed_mask, 127, 255, cv2.THRESH_BINARY)

    smoothed_mask_bbox = smoothed_mask.astype(np.uint8)
    contours, _ = cv2.findContours(smoothed_mask_bbox, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]
    pil_bounding_boxes = [cv2_to_pil_box(box) for box in bounding_boxes]

    #draw bbox in input
    draw = ImageDraw.Draw(image)
    for box in pil_bounding_boxes:
        draw.rectangle(box, outline="green", width=2)
        text = "TUMOR"
        text_position = (box[0], box[1] - 12)
        draw.text(text_position, text, fill="green")
    # prepare PIL input and CV2 mask to bytes for UI
    image_bytes = pil_to_bytes(image)
    _, out = cv2.imencode(".jpg", out)
    out = out.tobytes()
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