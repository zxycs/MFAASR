#加入aspp和emau，并且最后恢复层的的通道数也设置为32

import torch
import torch.nn as nn
from .blocks import ConvBlock, DeconvBlock, MeanShift
import torch.nn.functional as F
from einops import rearrange
import numbers
from networks.deeplabv3_aspp import ASPP
from networks.EMANET_EMAU import EMAU
# from networks.my_nonlocal import MYNONLOCAL
from networks.nonlocal_arch import MYNONLOCAL

class FeedbackBlock(nn.Module):
    def __init__(self, num_features, num_groups, upscale_factor, act_type, norm_type):
        super(FeedbackBlock, self).__init__()
        if upscale_factor == 2:
            stride = 2
            padding = 2
            kernel_size = 6
        elif upscale_factor == 3:
            stride = 3
            padding = 2
            kernel_size = 7
        elif upscale_factor == 4:
            stride = 4
            padding = 2
            kernel_size = 8
        elif upscale_factor == 8:
            stride = 8
            padding = 2
            kernel_size = 12

        self.num_groups = num_groups

        self.compress_in = ConvBlock(2*num_features, num_features,
                                     kernel_size=1,
                                     act_type=act_type, norm_type=norm_type)

        self.upBlocks = nn.ModuleList()
        self.downBlocks = nn.ModuleList()
        self.uptranBlocks = nn.ModuleList()
        self.downtranBlocks = nn.ModuleList()

        for idx in range(self.num_groups):
            self.upBlocks.append(DeconvBlock(num_features, num_features,
                                             kernel_size=kernel_size, stride=stride, padding=padding,
                                             act_type=act_type, norm_type=norm_type))
            self.downBlocks.append(ConvBlock(num_features, num_features,
                                             kernel_size=kernel_size, stride=stride, padding=padding,
                                             act_type=act_type, norm_type=norm_type, valid_padding=False))
            if idx > 0:
                self.uptranBlocks.append(ConvBlock(num_features*(idx+1), num_features,
                                                   kernel_size=1, stride=1,
                                                   act_type=act_type, norm_type=norm_type))
                self.downtranBlocks.append(ConvBlock(num_features*(idx+1), num_features,
                                                     kernel_size=1, stride=1,
                                                     act_type=act_type, norm_type=norm_type))

        self.compress_out = ConvBlock((num_groups)*num_features, num_features,
                                      kernel_size=1,
                                      act_type=act_type, norm_type=norm_type)

        self.should_reset = True
        self.last_hidden = None

        # self.sml = SML(num_features, 128, 0.999)
        self.zxyfdam=FDAM(n_feats=num_features)

        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=num_features, num_heads=8, ffn_expansion_factor= 2.66,
                             bias=False, LayerNorm_type='WithBias') for i in range(1)])

        # self.feedaspp1=ASPP()
        # self.feed_EMAU = EMAU(32, 64, upscale_factor)  # 注意，第二个参数不是通道数
        # self.my_local=MYNONLOCAL()

    def forward(self, x):
        if self.should_reset:#第一次的时候设置为true
            self.last_hidden = torch.zeros(x.size()).cuda()#在计算multi-adds的时候应该去掉.cuda()
            #self.last_hidden = torch.zeros(x.size())
            self.last_hidden.copy_(x)
            self.should_reset = False

        x = torch.cat((x, self.last_hidden), dim=1)#图3中1*1卷积的前面的输入部分
        x = self.compress_in(x)#就是经过一个1*1的卷积

        # y=self.feedaspp1(x)

        zxy2 = self.zxyfdam(x)

        lr_features = []
        hr_features = []
        lr_features.append(x)

        for idx in range(self.num_groups):
            LD_L = torch.cat(tuple(lr_features), 1)    # when idx == 0, lr_features == [x]
            if idx > 0:
                LD_L = self.uptranBlocks[idx-1](LD_L)
            LD_H = self.upBlocks[idx](LD_L)

            hr_features.append(LD_H)

            LD_H = torch.cat(tuple(hr_features), 1)
            if idx > 0:
                LD_H = self.downtranBlocks[idx-1](LD_H)
            LD_L = self.downBlocks[idx](LD_H)

            lr_features.append(LD_L)

        del hr_features
        output = torch.cat(tuple(lr_features[1:]), 1)   # leave out input x, i.e. lr_features[0]
        # output = torch.cat([output,y], 1)
        output = self.compress_out(output)


        zxy1=self.encoder_level3(output)


        output=zxy1+zxy2

        self.last_hidden = output

        return output

    def reset_state(self):
        self.should_reset = True

class CSUNET(nn.Module):
    def __init__(self, in_channels, out_channels, num_features, num_steps, num_groups, upscale_factor, act_type = 'prelu', norm_type = None):
        super(CSUNET, self).__init__()

        if upscale_factor == 2:
            stride = 2
            padding = 2
            kernel_size = 6
        elif upscale_factor == 3:
            stride = 3
            padding = 2
            kernel_size = 7
        elif upscale_factor == 4:
            stride = 4
            padding = 2
            kernel_size = 8
        elif upscale_factor == 8:
            stride = 8
            padding = 2
            kernel_size = 12

        self.num_steps = num_steps
        self.num_features = num_features
        self.upscale_factor = upscale_factor

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(rgb_mean, rgb_std)

        # LR feature extraction block
        self.conv_in = ConvBlock(in_channels, 4*num_features,
                                 kernel_size=3,
                                 act_type=act_type, norm_type=norm_type)

        self.feat_in = ConvBlock(4*num_features, num_features,
                                 kernel_size=1,
                                 act_type=act_type, norm_type=norm_type)

        # basic block
        self.block = FeedbackBlock(num_features, num_groups, upscale_factor, act_type, norm_type)

        # reconstruction block
      # uncomment for pytorch 0.4.0
        # self.upsample = nn.Upsample(scale_factor=upscale_factor, mode='bilinear')

        self.out = DeconvBlock(num_features, num_features,
                               kernel_size=kernel_size, stride=stride, padding=padding,
                               act_type='prelu', norm_type=norm_type)
        self.conv_out = ConvBlock(num_features, out_channels,
                                  kernel_size=3,
                                  act_type=None, norm_type=norm_type)

        self.add_mean = MeanShift(rgb_mean, rgb_std, 1)


    def forward(self, x):
        self._reset_state()#设置self.should_reset = True

        x = self.sub_mean(x)#print(x.shape)  torch.Size([16, 3, 40, 40])
      # uncomment for pytorch 0.4.0
        # inter_res = self.upsample(x)

      # comment for pytorch 0.4.0
        inter_res = nn.functional.interpolate(x, scale_factor=self.upscale_factor, mode='bilinear', align_corners=False)
        #这里的就是upasmple模块，还是没有采用pixshuffle模块

        # LR feature extraction block，就是两个卷积，做初始特征提取部分
        x = self.conv_in(x)#这里是先激活，然后经过一个卷积
        x = self.feat_in(x)


        outs = []
        for _ in range(self.num_steps):
            h = self.block(x)#FeedbackBlock部分

            h = torch.add(inter_res, self.conv_out(self.out(h)))
            h = self.add_mean(h)
            outs.append(h)
        #print('输出的长度',len(outs))
        return outs# return output of every timesteps


    def _reset_state(self):
        self.block.reset_state()

class SML(nn.Module):
    def __init__(self, hdim, kdim, moving_average_rate=0.999):
        super().__init__()
        self.c = hdim
        self.k = kdim
        self.moving_average_rate = moving_average_rate
        self.units = nn.Embedding(kdim, hdim)

    def update(self, x, score, m=None):
        if m is None:
            m = self.units.weight.data
        x = x.detach()
        embed_ind = torch.max(score, dim=1)[1]
        embed_onehot = F.one_hot(embed_ind, self.k).type(x.dtype)
        embed_onehot_sum = embed_onehot.sum(0)
        embed_sum = x.transpose(0, 1) @ embed_onehot  # (c, k)
        embed_mean = embed_sum / (embed_onehot_sum + 1e-6)
        new_data = m * self.moving_average_rate + embed_mean.t() * (1 - self.moving_average_rate)
        if self.training:
            self.units.weight.data = new_data
        return new_data

    def forward(self, x, update_flag=True):
        b, c, h, w = x.size()
        assert c == self.c
        k, c = self.k, self.c
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(-1, c)  # (n, c)
        m = self.units.weight.data  # (k, c)
        xn = F.normalize(x, dim=1)  # (n, c)
        mn = F.normalize(m, dim=1)  # (k, c)
        score = torch.matmul(xn, mn.t())  # (n, k)
        # for i in range(3):
        m = self.update(x, score, m)
            # mn = F.normalize(m, dim=1)  # (k, c)
            # score = torch.matmul(xn, mn.t())  # (n, k)
        soft_label = F.softmax(score, dim=1)
        out = torch.matmul(soft_label, m)  # (n, c)
        out = out.view(b, h, w, c).permute(0, 3, 1, 2)
        return out

class FDAM(nn.Module):
    def __init__(self, n_feats=32):
        super(FDAM, self).__init__()

        self.c1 = default_conv(n_feats, n_feats, 1)
        self.c2 = default_conv(n_feats, n_feats // 2, 3)
        self.c3 = default_conv(n_feats, n_feats // 2, 3)
        self.c4 = default_conv(n_feats*2, n_feats, 3)
        self.c5 = default_conv(n_feats // 2, n_feats // 2, 3)
        self.c6 = default_conv(n_feats*2, n_feats, 1)

        self.se = CALayer(channel=2*n_feats, reduction=16)
        self.sa = SpatialAttention()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x

        y1 = self.act(self.c1(x))
        y2 = self.act(self.c2(y1))
        y3 = self.act(self.c3(y1))
        cat1 = torch.cat([y1, y2, y3], 1)
        y4 = self.act(self.c4(cat1))
        y5 = self.c5(y3)  # 16
        cat2 = torch.cat([y2, y5, y4], 1)
        ca_out = self.se(cat2)
        sa_out = self.sa(cat2)
        y6 = ca_out + sa_out
        y7 = self.c6(y6)
        output = res + y7

        return output

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        res = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        y = self.sigmoid(x)
        return y * res


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
