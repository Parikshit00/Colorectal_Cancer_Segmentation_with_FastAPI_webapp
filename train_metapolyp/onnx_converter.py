import torch
import os
os.environ['KECAM_BACKEND'] = 'torch'
from keras_cv_attention_models import caformer
from keras_cv_attention_models.backend import models
import onnx
import onnxruntime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone = caformer.CAFormerS18(input_shape=(3, 256, 256), pretrained="imagenet", num_classes=0).to(device)

# Define layer names to extract outputs from
layer_names = ['stack4_block3_mlp_Dense_1', 'stack3_block9_mlp_Dense_1', 'stack2_block3_mlp_Dense_1', 'stack1_block3_mlp_Dense_1']

# Prepare a dummy input tensor
dummy_input = torch.randn(1, 3, 256, 256).to(device)

backbone = models.Model(backbone.inputs, [backbone.get_layer(ii).output for ii in layer_names])
# Save the model to ONNX format
onnx_path = "caformer_s18.onnx"
torch.onnx.export(backbone, dummy_input, onnx_path, input_names=['input'], output_names=['output'])
