import math
from torch import nn as nn

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def deconv4x4(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=stride, padding=1, bias=False)


class InvBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, layer_normalization='batch'):
        super(InvBasicBlock, self).__init__()
        self.layer_normalization = layer_normalization
        if upsample is not None:
            self.conv1 = deconv4x4(inplanes, planes, stride)
        else:
            self.conv1 = conv3x3(inplanes, planes, stride)
        if layer_normalization == 'batch':
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)
        elif layer_normalization == 'instance':
            self.in1 = nn.InstanceNorm2d(planes)
            self.in2 = nn.InstanceNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.layer_normalization == 'batch':
            out = self.bn1(out)
        elif self.layer_normalization == 'instance':
            out = self.in1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.layer_normalization == 'batch':
            out = self.bn2(out)
        elif self.layer_normalization == 'instance':
            out = self.in2(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        return self.relu(out)


class InvResNet(nn.Module):

    def __init__(self, block, layers, output_size=64, output_channels=1, input_dims=99, layer_normalization='none'):
        super(InvResNet, self).__init__()
        self.layer_normalization = layer_normalization
        self.lin_landmarks = None
        self.inplanes = 512
        self.output_size = output_size
        self.output_channels = output_channels
        self.fc = nn.Linear(input_dims, 512)
        self.conv1 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=1, padding=0, bias=False)
        self.add_in_tensor = None

        if layer_normalization == 'batch':
            self.bn1 = nn.BatchNorm2d(512)
        elif layer_normalization == 'instance':
            self.in1 = nn.InstanceNorm2d(64)

        if layer_normalization == 'batch':
            self.norm = nn.BatchNorm2d
        elif layer_normalization == 'instance':
            self.norm = nn.InstanceNorm2d
        else:
            raise ValueError(layer_normalization)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 256, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 64, layers[3], stride=2)
        self.tanh = nn.Tanh()
        self.sigmoid=nn.Sigmoid()
        self.x2 = None
        if self.output_size == 256:
            self.layer5 = self._make_layer(block,  64, layers[3], stride=2)
        elif self.output_size == 512:
            self.layer5 = self._make_layer(block,  64, layers[3], stride=2)
            self.layer6 = self._make_layer(block,  64, layers[3], stride=2)
            
        

        self.lin = nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer_down(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self.norm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes * block.expansion,
                          kernel_size=4, stride=stride, padding=1, bias=False),
                self.norm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, upsample, layer_normalization=self.layer_normalization))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, with_finetuning=False):

        x = self.fc(x)
        x = x.view(x.size(0), -1, 1,1)

        x = self.conv1(x)
        if self.layer_normalization == 'batch':
            x = self.bn1(x)
        elif self.layer_normalization == 'instance':
            x = self.in1(x)

        x = self.relu(x)
        self.x1 = self.layer1(x)
        self.x2 = self.layer2(self.x1)

            
        if self.output_size == 64:
            self.x3 = self.layer3(self.x2)
            self.x4 = self.layer4(self.x3)
            x=self.x3
        
        elif self.output_size == 128:
            self.x3 = self.layer3(self.x2)
            self.x4 = self.layer4(self.x3)
            x = self.x4
        elif self.output_size == 256:
            self.x3 = self.layer3(self.x2)
            self.x4 = self.layer4(self.x3)
            x = self.layer5(self.x4)
        elif self.output_size == 512:
            self.x3 = self.layer3(self.x2)
            self.x4 = self.layer4(self.x3)
            x = self.layer5(self.x4)
            x = self.layer6(x)

        x = self.lin(x)
        x = self.sigmoid(x)
        return x

def invresnet18(dim=99,output_size=64,grayscale=True):
    if grayscale==True:
        channel=1
    else:
        channel=3
        
    return InvResNet(InvBasicBlock, [2, 2, 2, 2],layer_normalization='batch',output_channels=channel,input_dims=dim,output_size=output_size,grayscale=grayscale)    