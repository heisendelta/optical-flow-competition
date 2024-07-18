import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, Any
import time

import imageio
imageio.plugins.freeimage.download()
import torch.nn.functional as nn_F

from torch import nn
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import argparse

# From https://github.com/Kwanss/PCLNet
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        endpoint = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        endpoint.append(x) # output here

        x = self.maxpool(x)
        x = self.layer1(x)
        endpoint.append(x)
        x = self.layer2(x)
        endpoint.append(x)
        x = self.layer3(x)
        endpoint.append(x)
        x = self.layer4(x)
        endpoint.append(x)

        return endpoint


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        state = model.state_dict()
        state_ckp = model_zoo.load_url(model_urls['resnet18'])
        cnt = 0
        for k, val in state_ckp.items():
            if k in state.keys():
                state[k] = val
                cnt += 1
        model.load_state_dict(state)
        print ("RestNet checkpoint loaded: %d" % cnt)
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        state = model.state_dict()
        state_ckp = model_zoo.load_url(model_urls['resnet34'])
        cnt = 0
        for k, val in state_ckp.items():
            if k in state.keys():
                state[k] = val
                cnt += 1
        model.load_state_dict(state)
        print ("RestNet checkpoint loaded: %d" % cnt)
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model



class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.bias = bias
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)
        self.Gates = nn.Conv2d(self.input_channels + self.hidden_channels , 4*self.hidden_channels,
                self.kernel_size, 1, self.padding, bias=True)

    def forward(self, x, h, c):

        stacked_inputs = torch.cat((x, h), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across the channel dimension
        xi, xf, xo, xg = gates.chunk(4, 1)

        # apply sigmoid non linearity
        xi = torch.sigmoid(xi)
        xf = torch.sigmoid(xf)
        xo = torch.sigmoid(xo)
        xg = torch.tanh(xg)

        # compute current cell and hidden state
        c = (xf * c) + (xi * xg)
        h = xo * torch.tanh(c)

        return h, c

    def init_hidden(self, batch_size, hidden, shape):
        return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])),
                Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])))


class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, step=1, effective_step=[1], bias=True):
        super(ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.bias = bias
        self.effective_step = effective_step
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size, self.bias)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input):
        #input : (num, seq_len, channel, H,W)
        internal_state = []
        outputs = []
        for step in range(self.step):
            x = input[:, step, :,:,:]
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)
                if step == 0:
                    bsize, _, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                            shape=(height, width))
                    internal_state.append((h, c))

                # do forward
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)
            # only record effective steps
            if step in self.effective_step:
                outputs.append(x)
        return outputs, (x, new_c)
    
def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    if type(in_planes) == np.int64:
        in_planes = np.asscalar(in_planes)
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.LeakyReLU(0.1))


def predict_flow(in_planes):
    if type(in_planes) == np.int64:
        in_planes = np.asscalar(in_planes)
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=True)


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    if type(in_planes) == np.int64:
        in_planes = np.asscalar(in_planes)
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)

def in_f(flow):
    return nn_F.interpolate(flow, size=(480, 640), mode='bilinear', align_corners=False)


class PCLNet(nn.Module):
    """
    PCLNet: Unsupervised Learning for Optical Flow Estimation Using Pyramid Convolution LSTM
    Author: Shuosen Guan
    """

    def __init__(self, args):

        super(PCLNet, self).__init__()
        self.args = args

        snippet_len = args.snippet_len
        self.feature_net = eval(args.backbone)(pretrained=True, num_classes=args.class_num)

        if args.freeze_vgg:
            for p in self.feature_net.parameters():
                p.required_grad = False
            print("[>>>> Feature head frozen.<<<<]")

        # Motion Encoding
        # in_size: 1/2
        self.clstm_encoder_1 = ConvLSTM(input_channels=64, hidden_channels=[64],
                                        kernel_size=3, step=snippet_len, effective_step=list(range(snippet_len)))
        # in_size: 1/4
        self.clstm_encoder_2 = ConvLSTM(input_channels=64, hidden_channels=[64],
                                        kernel_size=3, step=snippet_len, effective_step=list(range(snippet_len)))
        # in_size: 1/8
        self.clstm_encoder_3 = ConvLSTM(input_channels=128, hidden_channels=[128],
                                        kernel_size=3, step=snippet_len, effective_step=list(range(snippet_len)))
        # in_size: 1/16
        self.clstm_encoder_4 = ConvLSTM(input_channels=256, hidden_channels=[256],
                                        kernel_size=3, step=snippet_len, effective_step=list(range(snippet_len)))

        self.conv_B1    = conv(64, 64, stride=1, kernel_size=3, padding=1)
        self.conv_S1_1  = conv(64, 64, stride=1, kernel_size=3, padding=1)
        self.conv_S1_2  = conv(64, 64, stride=1, kernel_size=3, padding=1)
        self.conv_D1    = conv(64, 64, stride=2)
        self.Pool1      = nn.MaxPool2d(8, 8)

        self.conv_B2    = conv(64, 64, stride=1, kernel_size=3, padding=1)
        self.conv_S2_1  = conv(64 + 64, 128, stride=1, kernel_size=3, padding=1)
        self.conv_S2_2  = conv(128, 128, stride=1, kernel_size=3, padding=1)
        self.conv_D2    = conv(128, 64, stride=2)
        self.Pool2      = nn.MaxPool2d(4, 4)

        self.conv_B3    = conv(128, 128, stride=1, kernel_size=3, padding=1)
        self.conv_S3_1  = conv(128 + 64, 128, stride=1, kernel_size=3, padding=1)
        self.conv_S3_2  = conv(128, 128, stride=1, kernel_size=3, padding=1)
        self.conv_D3    = conv(128, 64, stride=2)
        self.Pool3      = nn.MaxPool2d(2, 2)

        self.conv_B4    = conv(256, 128, stride=1, kernel_size=3, padding=1)
        self.conv_S4_1  = conv(128 + 64, 128, stride=1, kernel_size=3, padding=1)
        self.conv_S4_2  = conv(128, 128, stride=1, kernel_size=3, padding=1)

        # Motion feature
        self.conv_M = conv((64 + 128 + 128 + 128), 256, stride=1, kernel_size=3, padding=1)

        # Motion reconstruction
        if self.args.couple:
            rec_in_size = [0, 64 + 64 + 2, 128 + 128 + 2, 128 + 196 + 2, 128 + 256]
        else:
            rec_in_size = [0, 64 + 2, 128 + 2, 196 + 2, 256]

        self.conv_4     = conv(rec_in_size[4], 256)
        self.pred_flow4 = predict_flow(256)
        self.up_flow4   = deconv(2, 2)
        self.up_feat4   = deconv(256, 196)

        self.conv_3     = conv(rec_in_size[3], 196)
        self.pred_flow3 = predict_flow(196)
        self.up_flow3   = deconv(2, 2)
        self.up_feat3   = deconv(196, 128)

        self.conv_2     = conv(rec_in_size[2], 96)
        self.pred_flow2 = predict_flow(96)
        self.up_flow2   = conv(2, 2)
        self.up_feat2   = conv(96, 64)

        self.conv_1     = conv(rec_in_size[1], 64)
        self.pred_flow1 = predict_flow(64)

        self.dc_conv1 = conv(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.dc_conv2 = conv(64, 64, kernel_size=3, stride=1, padding=2, dilation=2)
        self.dc_conv3 = conv(64, 64, kernel_size=3, stride=1, padding=4, dilation=4)
        self.dc_conv4 = conv(64, 64, kernel_size=3, stride=1, padding=8, dilation=8)
        self.dc_conv5 = conv(64, 64, kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc_conv6 = conv(64, 32, kernel_size=3, stride=1, padding=1, dilation=1)
        self.dc_conv7 = predict_flow(32)

        # for some reason, conv returns shape torch.Size([B*T, 3, 482, 642])
        self.reduce_channels = nn.Conv2d(4, 3, kernel_size=1)


    def forward(self, x):

        if x.dim() == 6:    # (batch_size, K, snippet_len, channel, H, W)
            batch_size, K, snippet_len, channel, H, W = x.size()
        elif x.dim() == 5:  # (batch_size, snippet_len, channel, H, W)
            batch_size, snippet_len, channel, H, W = x.size()
            K = 1
        elif x.dim() == 4:  # (batch_size, channel * snippet_len, H, W)
            batch_size, _channels, H, W = x.size()
            K, channel = 1, 3
            snippet_len = _channels // channel
        else:
            raise RuntimeError('Input format not suppored!')

        x = x.contiguous().view(-1, channel, H, W)
        if channel > 3:
            x = self.reduce_channels(x)

        la1, la2, la3, la4, _ = self.feature_net(x)

        la1 = la1.view((-1, snippet_len) + la1.size()[1:])
        la2 = la2.view((-1, snippet_len) + la2.size()[1:])
        la3 = la3.view((-1, snippet_len) + la3.size()[1:])
        la4 = la4.view((-1, snippet_len) + la4.size()[1:])
        # la5 = la5.view((-1, snippet_len) + la5.size()[1:])

        h1, _ = self.clstm_encoder_1(la1)
        h2, _ = self.clstm_encoder_2(la2)
        h3, _ = self.clstm_encoder_3(la3)
        h4, _ = self.clstm_encoder_4(la4)
        # list for each step (batch_size * K, channel, H, W)

        # (batch_size * K*(snippet_len -1), channel, H, W)
        h1 = torch.stack(h1[1:], 1).view((-1,) + h1[0].size()[-3:])
        h2 = torch.stack(h2[1:], 1).view((-1,) + h2[0].size()[-3:])
        h3 = torch.stack(h3[1:], 1).view((-1,) + h3[0].size()[-3:])
        h4 = torch.stack(h4[1:], 1).view((-1,) + h4[0].size()[-3:])

        x1 = self.conv_B1(h1)
        x1 = self.conv_S1_2(self.conv_S1_1(x1))

        x2 = torch.cat((self.conv_B2(h2), self.conv_D1(x1)), 1)
        x2 = self.conv_S2_2(self.conv_S2_1(x2))

        x3 = torch.cat((self.conv_B3(h3), self.conv_D2(x2)), 1)
        x3 = self.conv_S3_2(self.conv_S3_1(x3))

        x4 = torch.cat((self.conv_B4(h4), self.conv_D3(x3)), 1)
        x4 = self.conv_S4_2(self.conv_S4_1(x4))

        xm = self.conv_M(torch.cat((self.Pool1(x1), self.Pool2(x2), self.Pool3(x3), x4), 1))

        rec_x4 = torch.cat((x4, xm), 1) if self.args.couple else xm
        x = self.conv_4(rec_x4)
        flow4 = self.pred_flow4(x)
        up_flow4 = self.up_flow4(flow4)
        up_feat4 = self.up_feat4(x)

        rec_x3 = torch.cat((x3, up_feat4, up_flow4), 1) if self.args.couple else torch.cat((up_feat4, up_flow4), 1)
        x = self.conv_3(rec_x3)
        flow3 = self.pred_flow3(x)
        up_flow3 = self.up_flow3(flow3)
        up_feat3 = self.up_feat3(x)

        rec_x2 = torch.cat((x2, up_feat3, up_flow3), 1) if self.args.couple else torch.cat((up_feat3, up_flow3), 1)
        x = self.conv_2(rec_x2)
        flow2 = self.pred_flow2(x)
        up_flow2 = self.up_flow2(flow2)
        up_feat2 = self.up_feat2(x)

        rec_x1 = torch.cat((x1, up_feat2, up_flow2), 1) if self.args.couple else torch.cat((up_feat2, up_flow2), 1)
        x = self.conv_1(rec_x1)
        flow1 = self.pred_flow1(x)

        x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
        flow1 += self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))

        re_dict = {
            'flow0': flow4,
            'flow1': flow3,
            'flow2': flow2,
            'flow3': flow1
        }

        # output size: (batch_size, K, snippet_len -1 , C,H,W)

#         flow_pyramid = [flo.view((batch_size, K, snippet_len - 1,) + flo.size()[-3:])
#                         for flo in [flow1, flow2, flow3, flow4]]
#         re_dict = {}
#         re_dict['flow_pyramid'] = flow_pyramid

#         return re_dict

        flow1 = flow1.view((batch_size, K, snippet_len - 1,) + flow1.size()[-3:])
        flow2 = flow2.view((batch_size, K, snippet_len - 1,) + flow2.size()[-3:])
        flow3 = flow3.view((batch_size, K, snippet_len - 1,) + flow3.size()[-3:])
        flow4 = flow4.view((batch_size, K, snippet_len - 1,) + flow4.size()[-3:])

        flow1_arr = [in_f(flow1[:, :, i, :, :, :].squeeze(1).squeeze(1)) for i in range(flow1.size(2))]
        flow2_arr = [in_f(flow2[:, :, i, :, :, :].squeeze(1).squeeze(1)) for i in range(flow2.size(2))]
        flow3_arr = [in_f(flow3[:, :, i, :, :, :].squeeze(1).squeeze(1)) for i in range(flow3.size(2))]
        flow4_arr = [in_f(flow4[:, :, i, :, :, :].squeeze(1).squeeze(1)) for i in range(flow4.size(2))]

        combined_flow = [torch.mean(torch.stack([f1, f2, f3, f4], dim=0), dim=0)
                             for f1, f2, f3, f4 in zip(flow1_arr, flow2_arr, flow3_arr, flow4_arr)]
        return re_dict, combined_flow
    

def train(train_data):
    # ------------------
    #   optimizer
    # ------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00002, weight_decay=args.wdecay, eps=args.epsilon)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 1e-5, args.num_steps + 100,
                                                    pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
    # loss_fn = TotalLoss(smoothness_weight=0.5)

    # num_epochs = args.train.epochs
    num_epochs = 1

    epe_losses = [[] for _ in range(num_epochs)]
    overall_losses = [[] for _ in range(num_epochs)]

    WINDOW_FACTOR = 2
    WINDOW = args.snippet_len * WINDOW_FACTOR

    def save_model(model, additional_string: str = None):
        current_time = time.strftime("%Y%m%d-%H%M%S")
        model_path = f"models/model_{current_time}"
        if additional_string:
            model_path += '_' + additional_string
        model_path += '.pth'
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    # ------------------
    #   Start training
    # ------------------
    model.train()

    for epoch in range(num_epochs):

        total_loss = 0
        prev_event_volumes = [torch.zeros([args.batch_size, 4, 480, 640])] * WINDOW # Acts as a queue

        print("on epoch: {}".format(epoch + 1))
        for i, batch in enumerate(tqdm(train_data)):

            try:
                batch: Dict[str, Any]

                event_image = batch["event_volume"].to(device) # [B, 3, 480, 640]
                ground_truth_flow = batch["flow_gt"].to(device) # [B, 2, 480, 640]

                prev_event_volumes.append(event_image)

                # input_tensor1 = [v.to(device) for v in prev_event_volumes[-WINDOW:]]
                input_tensor1 = prev_event_volumes[-WINDOW:]
                input_tensor = torch.stack([
                    torch.mean(torch.stack(input_tensor1[b: b + WINDOW_FACTOR], dim=0), dim=0)
                    for b in range(0, WINDOW, WINDOW_FACTOR) ], dim=1)
                _, flows = model(input_tensor) # [B, 3, 480, 640]

                # Overall loss requires flow0, ..., flow3 so we don't implement it here
                # What if you created flow_dict from flow0, ..., flow11 (n=12) to and then use overall loss?

                for j, flow in enumerate(flows):
                    print(f'batch {i} | flow #{j + 1} | EPE LOSS:', compute_epe_error(flow, ground_truth_flow).item())

                avg_flow = torch.mean(torch.stack(flows, dim=0), dim=0)
                epe_loss: torch.Tensor = compute_epe_error(avg_flow, ground_truth_flow)

                print(f"batch {i} average EPE LOSS: {epe_loss.item()}")
                epe_losses[epoch].append(epe_loss.item())

    #             print(f'batch {i} OVERALL LOSS: {loss_fn()}')

                optimizer.zero_grad()

                epe_loss.backward() # Change this to which loss function is to be updated
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                # xm.optimizer_step(optimizer)
                # xm.mark_step()

                total_loss += epe_loss.item() # This too

                if len(prev_event_volumes) >= WINDOW:
                    prev_event_volumes.pop(0) # Remove first element

                if (i + 1) % 10 == 0:
                    save_model(model, f'batch{i + 1}')

            except KeyboardInterrupt:
                save_model(model)
                continue
                # raise SystemExit("KeyboardInterrupt")

        scheduler.step()

        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_data)}')

def test(test_data):
    WINDOW_FACTOR = 2
    WINDOW = args.snippet_len * WINDOW_FACTOR

    model.eval()
    flow: torch.Tensor = torch.tensor([]).to(device)

    prev_event_volumes = [torch.zeros([1, 4, 480, 640])] * WINDOW # Acts as a queue

    with torch.no_grad():
        print("start test")
        for batch in tqdm(test_data):
            batch: Dict[str, Any]

            event_image = batch["event_volume"].to(device)
            prev_event_volumes.append(event_image)

            input_tensor1 = prev_event_volumes[-WINDOW:]
            input_tensor = torch.stack([
                torch.mean(torch.stack(input_tensor1[b: b + WINDOW_FACTOR], dim=0), dim=0)
                for b in range(0, WINDOW, WINDOW_FACTOR) ], dim=1)
            _, flows = model(input_tensor)

            batch_flow = torch.mean(torch.stack(flows, dim=0), dim=0)
            flow = torch.cat((flow, batch_flow), dim=0)  # [N, 2, 480, 640]

            if len(prev_event_volumes) >= WINDOW:
                prev_event_volumes.pop(0)

        print("test done")

if __name__ == '__main__':
    device = 'cpu'

    args = argparse.Namespace(
        name='pclnet',
        snippet_len=5,
        backbone='resnet34',
        class_num=101,
        freeze_vgg=True,
        couple=False,

        lr=0.01, # 2e-5
        num_steps=100000,
        batch_size=32, # default: 16
        image_size=[480, 640],
        mixed_precision=False,
        iters=12,
        wdecay=0.00005,
        epsilon=1e-8,
        clip=1.0,
        dropout=0.0,
        gamma=0.8,
        add_noise=False,
        seed=27,
        dataset_path='data/',
    )
    model = torch.nn.DataParallel(PCLNet(args).to(device))
    model.load_state_dict(torch.load('models/pclnet/resnet34/epoch3_batch40.pth'))
