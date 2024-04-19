import torch
import torch.nn as nn
import os
os.environ['KECAM_BACKEND'] = 'torch'
from keras_cv_attention_models import caformer
from keras_cv_attention_models.backend import models
from layers.upsampling import decode
from layers.convformer import convformer 
from layers.util_layers import merge, conv_bn_act
import onnx
import onnxruntime
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
'''
backbone = caformer.CAFormerS18(input_shape=(3, 256, 256), pretrained="imagenet", num_classes=0).to(device)
layer_names = ['stack4_block3_mlp_Dense_1', 'stack3_block9_mlp_Dense_1', 'stack2_block3_mlp_Dense_1', 'stack1_block3_mlp_Dense_1']
class custom_model(nn.Module):
    def __init__(self, img_size = 256,num_classes=1):
        super(custom_model, self).__init__()
        self.backbone = models.Model(backbone.inputs, [backbone.get_layer(ii).output for ii in layer_names])
        self.num_classes = num_classes
        self.decode = decode(in_channel=512,filters =512,scale = 4)
        self.decode0 = decode(in_channel=512,filters = 320,scale = 2)
        self.convformer0 = convformer(input_shape=torch.Size([320, 16, 16]),filters=320, padding='same')
        self.decode1 = decode(in_channel=320,filters = 128,scale = 2)
        self.convformer1 = convformer(input_shape=torch.Size([128, 32, 32]),filters=128, padding='same')
        self.decode2 = decode(in_channel=128,filters = 64,scale = 2)
        self.convformer2 = convformer(input_shape=torch.Size([64, 64, 64]),filters=64, padding='same')
        self.conv1 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, padding="same")
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding="same")
        self.act2 = nn.ReLU()
        self.conv_bn_act0 = conv_bn_act(in_channel =320 ,filters=320, kernel_size=(1, 1))
        self.conv_bn_act1 = conv_bn_act(in_channel =128 ,filters=128, kernel_size=(1, 1))
        self.conv_bn_act2 = conv_bn_act(in_channel =64 ,filters=64, kernel_size=(1, 1))
        self.decode11 = decode(in_channel=128,filters = 128,scale = 8)
        self.upscale_final =conv_bn_act(in_channel =128 ,filters=32, kernel_size=(1, 1))
        self.decode_final = decode(in_channel=64,filters = 32,scale = 4)
        self.conv_bn_act_final =conv_bn_act(in_channel =32 ,filters=32, kernel_size=(1, 1))
        self.conv_final = nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.act_final = nn.Sigmoid()

    def forward(self, x):
        layers = [tensor for tensor in self.backbone(x)]
        layers = [l.permute(0,3,1,2) for l in layers]
        x = layers[0]
        upscale_feature = self.decode(x)
        for i, layer in enumerate(layers[1:]):
            if i == 0:
                x = self.decode0(x)
                layer_fusion = self.convformer0(layer)
            if i == 1:
                x = self.decode1(x)
                layer_fusion = self.convformer1(layer)
            if i == 2:
                x = self.decode2(x)
                layer_fusion = self.convformer2(layer)
            if (i%2 == 1):
                upscale_feature = self.act1(self.conv1(upscale_feature))
                x = torch.add(x, upscale_feature)
                x= self.act2(self.conv2(x))
            x = merge([x, layer_fusion], layer.shape[1])
            if i == 0:
                x = self.conv_bn_act0(x)
            if i == 1:
                x = self.conv_bn_act1(x)
            if i == 2:
                x = self.conv_bn_act2(x)
            if (i%2 == 1):
                upscale_feature = self.decode11(x)
        upscale_feature = self.upscale_final(upscale_feature)
        x = self.decode_final(x)
        x = torch.add(x, upscale_feature)
        x = self.conv_bn_act_final(x)
        x = self.act_final(self.conv_final(x))
        return x
'''
onnx_path = "caformer_s18.onnx"
class custom_model(nn.Module):
    def __init__(self, img_size = 256,num_classes=1):
        super(custom_model, self).__init__()
        self.backbone = torch.onnx.load(onnx_path)
        self.num_classes = num_classes
        self.decode = decode(in_channel=512,filters =512,scale = 4)
        self.decode0 = decode(in_channel=512,filters = 320,scale = 2)
        self.convformer0 = convformer(input_shape=torch.Size([320, 16, 16]),filters=320, padding='same')
        self.decode1 = decode(in_channel=320,filters = 128,scale = 2)
        self.convformer1 = convformer(input_shape=torch.Size([128, 32, 32]),filters=128, padding='same')
        self.decode2 = decode(in_channel=128,filters = 64,scale = 2)
        self.convformer2 = convformer(input_shape=torch.Size([64, 64, 64]),filters=64, padding='same')
        self.conv1 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, padding="same")
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding="same")
        self.act2 = nn.ReLU()
        self.conv_bn_act0 = conv_bn_act(in_channel =320 ,filters=320, kernel_size=(1, 1))
        self.conv_bn_act1 = conv_bn_act(in_channel =128 ,filters=128, kernel_size=(1, 1))
        self.conv_bn_act2 = conv_bn_act(in_channel =64 ,filters=64, kernel_size=(1, 1))
        self.decode11 = decode(in_channel=128,filters = 128,scale = 8)
        self.upscale_final =conv_bn_act(in_channel =128 ,filters=32, kernel_size=(1, 1))
        self.decode_final = decode(in_channel=64,filters = 32,scale = 4)
        self.conv_bn_act_final =conv_bn_act(in_channel =32 ,filters=32, kernel_size=(1, 1))
        self.conv_final = nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.act_final = nn.Sigmoid()

    def forward(self, x):
        #layers = [tensor for tensor in self.backbone(x)]
        layers = [l.permute(0,3,1,2) for l in layers]
        x = layers[0]
        upscale_feature = self.decode(x)
        for i, layer in enumerate(layers[1:]):
            if i == 0:
                x = self.decode0(x)
                layer_fusion = self.convformer0(layer)
            if i == 1:
                x = self.decode1(x)
                layer_fusion = self.convformer1(layer)
            if i == 2:
                x = self.decode2(x)
                layer_fusion = self.convformer2(layer)
            if (i%2 == 1):
                upscale_feature = self.act1(self.conv1(upscale_feature))
                x = torch.add(x, upscale_feature)
                x= self.act2(self.conv2(x))
            x = merge([x, layer_fusion], layer.shape[1])
            if i == 0:
                x = self.conv_bn_act0(x)
            if i == 1:
                x = self.conv_bn_act1(x)
            if i == 2:
                x = self.conv_bn_act2(x)
            if (i%2 == 1):
                upscale_feature = self.decode11(x)
        upscale_feature = self.upscale_final(upscale_feature)
        x = self.decode_final(x)
        x = torch.add(x, upscale_feature)
        x = self.conv_bn_act_final(x)
        x = self.act_final(self.conv_final(x))
        return x
