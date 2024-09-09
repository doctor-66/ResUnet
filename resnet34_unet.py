# Implement your ResNet34_UNet model here

# assert False, "Not implemented yet!"
import torch
import torch.nn as nn
from torchsummary import summary

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        


    def _make_layer(self, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes !=  planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes,  planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(ResidualBlock(self.in_planes, planes, stride, downsample=downsample))
        self.in_planes = planes
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)

        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        return x1,x2,x3,x4,x5
    
class double_conv(nn.Module):
    def __init__(self,input,output):
        super().__init__()
        self.db_conv=nn.Sequential(
                nn.Conv2d(input,output,kernel_size=3,padding=1),
                nn.BatchNorm2d(output),
                nn.ReLU(inplace=True),
                nn.Conv2d(output,output,kernel_size=3,padding=1),
                nn.BatchNorm2d(output),
                nn.ReLU(inplace=True)
                )
    def forward(self,x):
        return self.db_conv(x)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.upsample1 = nn.ConvTranspose2d(512,256,kernel_size=2,stride=2)
        self.up1 = double_conv(512,256)

        self.upsample2 = nn.ConvTranspose2d(256,128,kernel_size=2,stride=2)
        self.up2 = double_conv(256,128)

        self.upsample3 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.up3 = double_conv(128,64)

        self.upsample4 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)

        self.dropout = nn.Dropout(0.5)

        self.out = nn.Conv2d(64,1,kernel_size=1)

    def forward(self,x1,x2,x3,x4,x5):
        
        x= self.upsample1(x5)
        x= torch.cat([x,x4],dim=1)
        x= self.up1(x)
        # x = self.dropout(x)


        x= self.upsample2(x)
        x= torch.cat([x,x3],dim=1)
        x= self.up2(x)
        # x = self.dropout(x)

        x= self.upsample3(x)
        x= torch.cat([x,x2],dim=1)
        x= self.up3(x)
        # x = self.dropout(x)

        x= self.upsample4(x)
        out= self.out(x)

        return out
    
class ResNet34_Unet(nn.Module):
    def __init__(self):
        super(ResNet34_Unet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, x):
        e1 ,e2 , e3, e4, e5 = self.encoder(x)
        x = self.decoder(e1 ,e2 , e3, e4, e5 )
        return x 

# print(ResNet34_Unet().eval())
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = ResNet34_Unet().to(device)
# summary(model, input_size=(3, 224, 224))