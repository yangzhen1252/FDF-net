import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class UP(nn.Module):
    def __init__(self, high_in_plane, low_in_plane, out_plane):
        super(UP, self).__init__()
        self.conv3x3 = nn.Conv2d(high_in_plane, out_plane, 3, 1, 1)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        #self.upsample1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv3x31 = nn.Conv2d(low_in_plane, out_plane, 3, 1, 1)
        self.conv1x1 = nn.Conv2d(low_in_plane, out_plane, 1)
        self.conv1x11 = nn.Conv2d(low_in_plane, out_plane, 1)
        self.ASPPConv=ASPPConv(low_in_plane, out_plane, 1)

    def forward(self, high_x, low_x):
        low_x1=low_x
        high_x = self.upsample(self.conv3x3(high_x))
        high_x= self.conv1x11(high_x)
        low_x=self.conv3x31(low_x)
        low_x1=self.ASPPConv(low_x1)
        low_x = self.conv1x1(low_x+low_x1)

        return high_x +low_x

class UP1(nn.Module):
    def __init__(self, high_in_plane, low_in_plane, out_plane):
        super(UP1, self).__init__()
        self.conv3x3 = nn.Conv2d(high_in_plane, out_plane, 3, 1, 1)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        #self.upsample1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv3x31 = nn.Conv2d(low_in_plane, out_plane, 3, 1, 1)
        self.conv1x1 = nn.Conv2d(low_in_plane, out_plane, 1)
        self.conv1x11 = nn.Conv2d(low_in_plane, out_plane, 1)


    def forward(self, high_x, low_x):

        high_x = self.upsample(self.conv3x3(high_x))
        high_x= self.conv1x11(high_x)
        low_x=self.conv3x31(low_x)

        low_x = self.conv1x1(low_x)

        return high_x +low_x


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)
class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)
class ASPP(nn.Module):
    def __init__(self, in_channels,out_channels1, atrous_rates):
        super(ASPP, self).__init__()
        out_channels =out_channels1
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels1, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))
        rate1, rate2 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        # modules.append(ASPPConv(in_channels, out_channels, rate3))
        # modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(4*out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))
    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

class Newblock(nn.Module):
    def __init__(self, in_channels,out_channels1):
        super(Newblock, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels, out_channels1, 3, 1, 1)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels1, 1)
        self.aspp=ASPP(in_channels,out_channels1,[1,2,3])

    def forward(self, l1):
        x1 = self.conv3x3(l1)
        x2 = self.conv1x1(l1)
        x3 = self.aspp(l1)

        return x2+x3+x1

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Residual, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 　ｘ卷积后shape发生改变,比如:x:[1,64,56,56] --> [1,128,28,28],则需要1x1卷积改变x
        if in_channels != out_channels:
            self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv1x1 = None

    def forward(self, x):
        # print(x.shape)
        o1 = self.relu(self.bn1(self.conv1(x)))
        # print(o1.shape)
        o2 = self.bn2(self.conv2(o1))
        # print(o2.shape)

        if self.conv1x1:
            x = self.conv1x1(x)

        out = self.relu(o2 + x)
        return out


class ResNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ResNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            Residual(64, 64),
            Residual(64, 64),
            Residual(64, 64),
        )

        self.conv3 = nn.Sequential(
            Residual(64, 128, stride=2),
            Residual(128, 128),
            Residual(128, 128),
            Residual(128, 128),
            Residual(128, 128),
        )

        self.conv4 = nn.Sequential(
            Residual(128, 256, stride=2),
            Residual(256, 256),
            Residual(256, 256),
            Residual(256, 256),
            Residual(256, 256),
            Residual(256, 256),
        )

        self.conv5 = nn.Sequential(
            Residual(256, 512, stride=2),
            Residual(512, 512),
            Residual(512, 512),
        )

        # self.avg_pool = nn.AvgPool2d(kernel_size=7)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 代替AvgPool2d以适应不同size的输入
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        out = self.avg_pool(out)
        out = out.view((x.shape[0], -1))

        out = self.fc(out)

        return out
class ChannelAttention(nn.Module):           # Channel Attention Module
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 3, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 3, in_planes, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)
        avg_out = self.fc1(avg_out)
        avg_out = self.relu(avg_out)
        avg_out = self.fc2(avg_out)

        max_out = self.max_pool(x)
        max_out = self.fc1(max_out)
        max_out = self.relu(max_out)
        max_out = self.fc2(max_out)

        out = avg_out + max_out
        out = self.sigmoid(out)
        return out


class SpatialAttention(nn.Module):           # Spatial Attention Module
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        out = self.sigmoid(out)
        return out


class BasicBlock(nn.Module):      # 左侧的 residual block 结构（18-layer、34-layer）
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):      # 两层卷积 Conv2d + Shutcuts
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.channel = ChannelAttention(self.expansion*planes)     # Channel Attention Module
        self.spatial = SpatialAttention()                          # Spatial Attention Module
        self.ASPPConv1=ASPPConv(in_planes, self.expansion*planes,2)
        self.ASPPConv2 = ASPPConv(in_planes, self.expansion * planes, 1)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:      # Shutcuts用于构建 Conv Block 和 Identity Block
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        CBAM_Cout = self.channel(out)
        out = out * CBAM_Cout
        CBAM_Sout = self.spatial(out)
        out = out * CBAM_Sout
        #out1=self.ASPPConv1(out)
        #out2 = self.ASPPConv2(out)
        out += self.shortcut(x)
       # out=out+out1+out2
        out = F.relu(out)
        return out
class IRModel(nn.Module):
    """
    downsample ratio=2
    """

    def __init__(self):
        super(IRModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(6)
        self.relu1 = nn.ReLU(True)
        self.maxpool1 = nn.MaxPool2d((2, 2))

        self.conva = nn.Conv2d(1, 6, 3, 1, 1)
        self.bna = nn.BatchNorm2d(6)
        self.relua = nn.ReLU(True)
        self.maxpoola = nn.MaxPool2d((2, 2))

        self.conv11 = nn.Conv2d(3, 6, 3, 1, 1)
        self.bn11 = nn.BatchNorm2d(6)
        self.relu11 = nn.ReLU(True)
        self.maxpool11 = nn.MaxPool2d((2, 2))

        self.conv111 = nn.Conv2d(3, 6, 3, 1, 1)
        self.bn111 = nn.BatchNorm2d(6)
        self.relu111 = nn.ReLU(True)
        self.maxpool111 = nn.MaxPool2d((2, 2))


        self.conv2 = nn.Conv2d(6, 12, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(12)
        self.relu2 = nn.ReLU(True)
        self.maxpool2 = nn.MaxPool2d((2, 2))

        self.conv2a = nn.Conv2d(6, 12, 3, 1, 1)
        self.bn2a = nn.BatchNorm2d(12)
        self.relu2a = nn.ReLU(True)
        self.maxpool2a = nn.MaxPool2d((2, 2))

        self.conv211 = nn.Conv2d(6, 12, 3, 1, 1)
        self.bn211 = nn.BatchNorm2d(12)
        self.relu211 = nn.ReLU(True)

        self.conv21 = nn.Conv2d(6, 12, 3, 1, 1)
        self.bn21 = nn.BatchNorm2d(12)
        self.relu21 = nn.ReLU(True)
        self.maxpool21 = nn.MaxPool2d((2, 2))



        self.conv3 = nn.Conv2d(12, 24, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(24)
        self.relu3 = nn.ReLU(True)
        self.maxpool3 = nn.MaxPool2d((2, 2))

        self.conv3a = nn.Conv2d(12, 24, 3, 1, 1)
        self.bn3a = nn.BatchNorm2d(24)
        self.relu3a = nn.ReLU(True)
        self.maxpool3a = nn.MaxPool2d((2, 2))

        self.conv31 = nn.Conv2d(12, 24, 3, 1, 1)
        self.bn31 = nn.BatchNorm2d(24)
        self.relu31 = nn.ReLU(True)


        self.conv4 = nn.Conv2d(24, 48, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(48)
        self.relu4 = nn.ReLU(True)

        self.conv4a = nn.Conv2d(24, 48, 3, 1, 1)
        self.bn4a = nn.BatchNorm2d(48)
        self.relu4a = nn.ReLU(True)




        self.aspp1 = ASPP(6,6, [1, 2])
        self.aspp2 = ASPP(12, 12, [1, 2])
        self.aspp3 = ASPP(24, 24, [1, 2])
        self.aspp11 = ASPP(6, 6, [1, 2])
        self.aspp21 = ASPP(12, 12, [1, 2])
        self.aspp111 = ASPP(6, 6, [1, 2])
        self.aspp1a = ASPP(6, 6, [1, 2])
        self.aspp2a = ASPP(12, 12, [1, 2])
        self.aspp3a = ASPP(24, 24, [1, 2])


        self.resnet = nn.Sequential(
            Residual(6, 6),
            Residual(6, 6),
            Residual(6, 6),

        )
        self.resneta = nn.Sequential(
            Residual(6, 6),
            Residual(6, 6),
            Residual(6, 6),

        )
        self.resnet1 = nn.Sequential(
            Residual(6, 6),
            Residual(6, 6),
            Residual(6, 6),

        )
        self.resnet11 = nn.Sequential(
            Residual(6, 6),
            Residual(6, 6),
            Residual(6, 6),

        )

        self.BB = BasicBlock(6, 6)
        self.BB1 = BasicBlock(12, 12)





        self.seb1 = UP(48, 24, 24)
        self.seb2 = UP(24, 12, 12)
        self.seb3 = UP(12, 6, 6)
        self.seb1a = UP(48, 24, 24)
        self.seb2a = UP(24, 12, 12)
        self.seb3a = UP(12, 6, 6)
        self.seb11 = UP(24, 12, 12)
        self.seb21 = UP(12, 6, 6)
        self.seb111 = UP(12, 6, 6)

        self.map = nn.Conv2d(6, 1, 1)
        self.map1 = nn.Conv2d(6, 1, 1)
        self.map11 = nn.Conv2d(6, 1, 1)
        self.mapa = nn.Conv2d(6, 1, 1)


    def forward(self, x):
        x1 = self.conv1(x)
        x1=self.resnet(x1)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        m1 = self.maxpool1(x1)
        y1=self.aspp1(m1)

        x11 = self.conv11(x)
        x11 = self.resnet1(x11)
        x11 = self.bn11(x11)
        x11 = self.relu11(x11)
        m11 = self.maxpool11(x11)
        y11 = self.aspp11(m11)

        x111 = self.conv111(x)
        x111 = self.resnet11(x111)
        x111 = self.bn111(x111)
        x111 = self.relu111(x111)
        m111 = self.maxpool111(x111)
        y111 = self.aspp111(m111)

        x2 = self.conv2(y1+y11+y111)
        x2 = self.bn2(x2)
        x2 = self.relu2(x2)
        m2 = self.maxpool2(x2)
        m3=self.aspp2(m2)



        x21 = self.conv21(y11)
        x21 = self.bn21(x21)
        x21 = self.relu21(x21)
        m21 = self.maxpool21(x21)
        m31 = self.aspp21(m21)

        x211 = self.conv211(y111)
        x211 = self.bn211(x211)
        x211 = self.relu211(x211)

        x3= self.conv3(m3+m31)
        x3 = self.bn3(x3)
        x3 = self.relu3(x3)
        m4 = self.maxpool3(x3)
        m5 = self.aspp3(m4)



        x31 = self.conv31(m31)
        x31 = self.bn31(x31)
        x31= self.relu31(x31)


        x4 = self.conv4(m5)
        x4 = self.bn4(x4)
        x4 = self.relu4(x4)


        up1 = self.seb1(x4, x3)
        up11 = self.seb11(x31, x21)
        up111 = self.seb111(x211, x111)
        up2 = self.seb2(up1, x2)
        up21 = self.seb21(up11, x11)

        up3 = self.seb3(up2, x1)


        out = self.map(up3)
        out1 = self.map1(up21)
        out11 = self.map11(up111)

        x1a = self.conva(out+out1+out11)
        x1a = self.resneta(x1a)
        x1a = self.bna(x1a)
        x1a = self.relua(x1a)
        m1a = self.maxpoola(x1a)
        y1a = self.aspp1a(m1a)
        cc=self.BB(y111+y11+y1)
        x2a = self.conv2a(y1a+cc+y111+y11+y1)
        x2a = self.bn2a(x2a)
        x2a = self.relu2a(x2a)
        m2a = self.maxpool2a(x2a)
        m3a = self.aspp2a(m2a)
        cc1 = self.BB1(m21 + m2)
        x3a = self.conv3a(m3a+cc1+m21 + m2)
        x3a = self.bn3a(x3a)
        x3a = self.relu3a(x3a)
        m4a = self.maxpool3a(x3a)
        m5a = self.aspp3a(m4a)

        x4a = self.conv4a(m5a+m4)
        x4a = self.bn4a(x4a)
        x4a = self.relu4a(x4a)

        up1a = self.seb1a(x4a, x3a)
        up2a = self.seb2a(up1a, x2a)
        up3a = self.seb3a(up2a, x1a)
        outa = self.mapa(up3a)
        return out,out1,out11,outa



class FU(nn.Module):
    def __init__(self, high_in_plane1, high_in_plane2,out_plane1,out_plane2):
        super(FU, self).__init__()
        self.conv1 = nn.Conv2d(high_in_plane1, out_plane1,  1)

        self.upsample1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv2 = nn.Conv2d(high_in_plane2, high_in_plane1, 1, 1)

        self.conv22 = nn.Conv2d(high_in_plane1, out_plane2, 3,1,1)

        self.maxpool1 = nn.MaxPool2d((2, 2))
        self.maxpool2 = nn.MaxPool2d((2, 2))




    def forward(self, x1,x2):
        x1=self.conv1(x1)
        x2 = self.conv2( self.upsample1(x2) )

        x4=x1+x2
        out1=x4
        out2=self.conv22(x4)
        out2 = self.maxpool1(out2)

        return out1,out2
