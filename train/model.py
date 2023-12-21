import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Recurrent_block(nn.Module):
    def __init__(self, cout, t):
        super().__init__()
        self.t = t
        self.cout = cout
        self.conv = nn.Sequential(
            nn.Conv2d(cout,cout,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)
            
            x1 = self.conv(x+x1)
        return x1




class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.x_ = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.gate(g)
        x1 = self.x_(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi

class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class AttentionR2Unet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder path
        #Recurrent_CNN + Relu
        self.recurrent_1 = RRCNN_block(ch_in=3,ch_out=64,t=2)
        
        #Recurrent_CNN + Relu + Maxpool
        self.maxpool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.recurrent_2 = RRCNN_block(ch_in=64,ch_out=128,t=2)
        
        #Recurrent_CNN + Relu + Maxpool
        self.maxpool3 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.recurrent_3 = RRCNN_block(ch_in=128,ch_out= 256,t=2)
        
        #Recurrent_CNN + Relu + Maxpool
        self.maxpool4 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.recurrent_4 = RRCNN_block(ch_in=256,ch_out=512,t=2)
        
        #Recurrent_CNN + Relu + Maxpool
        self.maxpool5 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.recurrent_5 = RRCNN_block(ch_in=512,ch_out=1024,t=2 )

        #Decoder Path
        #Upconvolution + Attention + UpRecurrent
        self.up_conv5 = up_conv(ch_in=1024,ch_out=512)
        self.attention5 = Attention_block(F_g=512,F_l=512,F_int= 256)
        self.up_recurrent_5 = RRCNN_block(ch_in=1024,ch_out=512,t=2)
        
        #Upconvolution + Attention + UpRecurrent
        self.up_conv4 = up_conv(ch_in=512,ch_out=256)
        self.attention4 = Attention_block(F_g=256,F_l=256,F_int= 128)
        self.up_recurrent_4 = RRCNN_block(ch_in=512,ch_out=256,t=2)

        
        #Upconvolution + Attention + UpRecurrent
        self.up_conv3 = up_conv(ch_in=256,ch_out=128)
        self.attention3 = Attention_block(F_g=128,F_l=128,F_int= 64)
        self.up_recurrent_3 = RRCNN_block(ch_in=256,ch_out=128,t=2)

        #Upconvolution + Attention + UpRecurrent
        self.up_conv2 = up_conv(ch_in=128,ch_out=64)
        self.attention2 = Attention_block(F_g=64,F_l=64,F_int= 32)
        self.up_recurrent_2 = RRCNN_block(ch_in=128,ch_out= 64,t=2)
        
        #Final conv1x1
        self.conv1x1_out = nn.Conv2d(64,1,kernel_size=1,stride=1,padding=0)
    
    def forward(self, x):
        #encoder
        x1 = self.recurrent_1(x)
        
        x2 = self.maxpool2(x1)
        x2 = self.recurrent_2(x2)

        x3 = self.maxpool3(x2)
        x3 = self.recurrent_3(x3)

        x4 = self.maxpool4(x3)
        x4 = self.recurrent_4(x4)

        x5 = self.maxpool5(x4)
        x5 = self.recurrent_5(x5)
        
        #decoder
        d5 = self.up_conv5(x5)
        x4 = self.attention5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.up_recurrent_5(d5)
        
        d4 = self.up_conv4(x4)
        x3 = self.attention4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)        
        d4= self.up_recurrent_4(d4)

        d3 = self.up_conv3(d4)
        x2 = self.attention3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.up_recurrent_3(d3)

        d2 = self.up_conv2(d3)
        x1 = self.attention2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.up_recurrent_2(d2)



        d1 = self.conv1x1_out(d2)

        return d1

    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def save(self, path):
        print('Saving model... %s' % path)
        torch.save(self, path)