import torch
import torch.nn as nn
import torch.nn.functional as F
from ops.dcn.deform_conv import ModulatedDeformConv
from model.attentionlayer import DSTA
from einops import rearrange
import numbers
import numpy as np


# ==========
# Neighborhood alignment (NA) block
# ==========

class STDF(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, base_ks=3, deform_ks=3):
        """
        Args:
            in_nc: num of input channels.
            out_nc: num of output channels.
            nf: num of channels (filters) of each conv layer.
            nb: num of conv layers.
            deform_ks: size of the deformable kernel.
        """
        super(STDF, self).__init__()

        self.nb = nb
        self.in_nc = in_nc
        self.deform_ks = deform_ks
        self.size_dk = deform_ks ** 2

        # u-shape backbone
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_nc, nf, base_ks, padding=base_ks // 2),
            nn.ReLU(inplace=True)
        )
        for i in range(1, nb):
            setattr(
                self, 'dn_conv{}'.format(i), nn.Sequential(
                    nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks // 2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(nf, nf, base_ks, padding=base_ks // 2),
                    nn.ReLU(inplace=True)
                )
            )
            setattr(
                self, 'up_conv{}'.format(i), nn.Sequential(
                    nn.Conv2d(2 * nf, nf, base_ks, padding=base_ks // 2),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True)
                )
            )
        self.tr_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, base_ks, padding=base_ks // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, padding=base_ks // 2),
            nn.ReLU(inplace=True)
        )

        self.offset_mask_5 = nn.Conv2d(
            nf, in_nc * 3 * self.size_dk, base_ks, padding=base_ks // 2
        )
        self.deform_conv_5 = ModulatedDeformConv(
            in_nc, out_nc, deform_ks, padding=deform_ks // 2, deformable_groups=in_nc
        )

    def forward(self, inputs):
        nb = self.nb
        in_nc = self.in_nc
        n_off_msk = self.deform_ks * self.deform_ks
        # feature extraction (with downsampling)
        out_lst = [self.in_conv(inputs)]  # record feature maps for skip connections
        for i in range(1, nb):
            dn_conv = getattr(self, 'dn_conv{}'.format(i))
            out_lst.append(dn_conv(out_lst[i - 1]))
        # trivial conv
        out = self.tr_conv(out_lst[-1])
        # feature reconstruction (with upsampling)
        for i in range(nb - 1, 0, -1):
            up_conv = getattr(self, 'up_conv{}'.format(i))
            out = up_conv(
                torch.cat([out, out_lst[i]], 1)
            )
        # compute offset and mask
        # offset: conv offset
        # mask: confidence
        out = self.out_conv(out)
        off_msk_5 = self.offset_mask_5(out)
        off_5 = off_msk_5[:, :in_nc * 2 * n_off_msk, ...]
        msk_5 = torch.sigmoid(
            off_msk_5[:, in_nc * 2 * n_off_msk:, ...]
        )
        # perform deformable convolutional fusion
        fused_feat_5 = F.relu(
            self.deform_conv_5(inputs, off_5, msk_5),
            inplace=True)

        return fused_feat_5


# ==========
# Reference alignment (RA) block
# ==========


class AligNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, base_ks=3, deform_ks=5):

        super(AligNet, self).__init__()
        self.in_nc = in_nc
        self.deform_ks = deform_ks
        self.size_dk = deform_ks ** 2
        # u-shape backbone

        self.offset_mask_5 = nn.Conv2d(
            nf, in_nc * 3 * self.size_dk, base_ks, padding=base_ks // 2
        )
        self.deform_conv_5 = ModulatedDeformConv(
            in_nc, out_nc, deform_ks, padding=deform_ks // 2, deformable_groups=in_nc
        )
        self.offset_mask_3 = nn.Conv2d(
            nf, in_nc * 3 * (3*3), base_ks, padding=base_ks // 2
        )
        self.deform_conv_3 = ModulatedDeformConv(
            in_nc, out_nc, 3, padding=3 // 2, deformable_groups=in_nc
        )
        self.offset_mask_1 = nn.Conv2d(
            nf, in_nc * 3 * (1*1), base_ks, padding=base_ks // 2
        )
        self.deform_conv_1 = ModulatedDeformConv(
            in_nc, out_nc,1, padding=1 // 2, deformable_groups=in_nc
        )

        self.select = SF(in_channels=nf, height=3, reduction=8, bias=False, n_feats=nf)
    def forward(self,out,inputs):
        in_nc = self.in_nc
        n_off_msk = self.deform_ks * self.deform_ks
        # feature extraction (with downsampling)

        off_msk_5 = self.offset_mask_5(out)
        off_5 = off_msk_5[:, :in_nc * 2 * n_off_msk, ...]
        msk_5 = torch.sigmoid(
            off_msk_5[:, in_nc * 2 * n_off_msk:, ...]
        )
        # perform deformable convolutional fusion
        fused_feat_5 = F.relu(
            self.deform_conv_5(inputs, off_5, msk_5),
            inplace=True)

        off_msk_3 = self.offset_mask_3(out)
        off_3 = off_msk_3[:, :in_nc * 2 * (3*3), ...]
        msk_3 = torch.sigmoid(
            off_msk_3[:, in_nc * 2 * (3*3):, ...]
        )
        # perform deformable convolutional fusion
        fused_feat_3 = F.relu(
            self.deform_conv_3(inputs, off_3, msk_3),
            inplace=True)

        off_msk_1 = self.offset_mask_1(out)
        off_1 = off_msk_1[:, :in_nc * 2 * (1 * 1), ...]
        msk_1 = torch.sigmoid(
            off_msk_1[:, in_nc * 2 * (1 * 1):, ...]
        )
        # perform deformable convolutional fusion
        fused_feat_1 = F.relu(
            self.deform_conv_1(inputs, off_1, msk_1),
            inplace=True)


        fused_feat = self.select(torch.cat([fused_feat_1,fused_feat_3,fused_feat_5],dim=1))

        return fused_feat


class FEnet(nn.Module):
    def __init__(self, in_nc, nf, base_ks=3):
        super(FEnet, self).__init__()

        self.in_nc = in_nc

        # u-shape backbone
        self.in_conv = nn.Sequential(
            nn.Conv2d(2*in_nc, nf, base_ks, padding=base_ks // 2),
            nn.ReLU(inplace=True)
        )
        for i in range(1, 3):
            setattr(
                self, 'dn_conv{}'.format(i), nn.Sequential(
                    nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks // 2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(nf, nf, base_ks, padding=base_ks // 2),
                    nn.ReLU(inplace=True)
                )
            )
            setattr(
                self, 'up_conv{}'.format(i), nn.Sequential(
                    nn.Conv2d(2 * nf, nf, base_ks, padding=base_ks // 2),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True)
                )
            )
        self.tr_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, base_ks, padding=base_ks // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, padding=base_ks // 2),
            nn.ReLU(inplace=True))

    def forward(self, x):
        nb =3
        out_lst_x = [self.in_conv(x)]  # record feature maps for skip connections
        for i in range(1, nb):
            dn_conv = getattr(self, 'dn_conv{}'.format(i))
            out_lst_x.append(dn_conv(out_lst_x[i - 1]))
        # trivial conv
        out = self.tr_conv(out_lst_x[-1])
        # feature reconstruction (with upsampling)
        for i in range(nb - 1, 0, -1):
            up_conv = getattr(self, 'up_conv{}'.format(i))
            out = up_conv(
                torch.cat([out, out_lst_x[i]], 1)
            )
        out_x = self.out_conv(out)

        return out_x

      
# ==========
# Ref-based Refinement Module (RRM)
# ==========

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
        x = F.elu(x1) * x2
        x = self.project_out(x)
        return x

class MDTA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim , kernel_size=1, bias=bias)
        self.qkv_dwconv_q = nn.Conv2d(dim , dim , kernel_size=3, stride=1, padding=1, groups=dim , bias=bias)
        self.qkv_dwconv_kv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x ,y = x.chunk(2,dim=1)

        b, c, h, w = x.shape

        q = self.qkv_dwconv_q(self.qkv(x))
        k = self.qkv_dwconv_kv(self.qkv(y))
        v = self.qkv_dwconv_kv(self.qkv(y))
        # q, k, v = qkv.chunk(3, dim=1)

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
     

class RRM(nn.Module):
    def __init__(self, in_nc, nf, out_nc, base_ks):
        super(RRM, self).__init__()

        self.fuse = SF(in_channels=nf, height=4, reduction=8, bias=False, n_feats=nf)

        self.conv1_a = nn.Sequential(
            LayerNorm(2*nf, 'BiasFree'),
            MDTA(dim=nf, num_heads=nf, bias=False))
        self.conv1_m = nn.Sequential(
            LayerNorm(nf, 'BiasFree'),
            FeedForward(dim=nf, ffn_expansion_factor=2, bias=False))

        self.down_1 = nn.Sequential(
            nn.Conv2d(nf, 2*nf, base_ks, stride=2, padding=base_ks // 2),
            nn.PReLU())
        self.down_1_a = nn.Sequential(
            LayerNorm(4*nf, 'BiasFree'),
            MDTA(dim=2*nf, num_heads=2*nf, bias=False))
        self.down_1_m = nn.Sequential(
            LayerNorm(2*nf, 'BiasFree'),
            FeedForward(dim=2*nf, ffn_expansion_factor=2, bias=False))

        self.down_2 = nn.Sequential(
            nn.Conv2d(2*nf, 4*nf, base_ks, stride=2, padding=base_ks // 2),
            nn.PReLU())
        self.down_2_a = nn.Sequential(
            LayerNorm(8*nf, 'BiasFree'),
            MDTA(dim=4*nf, num_heads=4*nf, bias=False))
        self.down_2_m = nn.Sequential(
            LayerNorm(4*nf, 'BiasFree'),
            FeedForward(dim=4*nf, ffn_expansion_factor=2, bias=False))

        self.conv2_1 = nn.Sequential(
            nn.Conv2d(4*nf, 4*nf, base_ks, stride=1, padding=base_ks // 2),
            nn.PReLU())
        self.conv2_a = nn.Sequential(
            LayerNorm(8*nf, 'BiasFree'),
            MDTA(dim=4*nf, num_heads=2*nf, bias=False))
        self.conv2_m = nn.Sequential(
            LayerNorm(4*nf, 'BiasFree'),
            FeedForward(dim=4*nf, ffn_expansion_factor=2, bias=False))

        self.up_1 = nn.Sequential(
            nn.ConvTranspose2d(4*nf, 2*nf, 4, stride=2, padding=1),
            nn.ReLU(inplace=True))
        self.up_1_a = nn.Sequential(
            LayerNorm(4*nf, 'BiasFree'),
            MDTA(dim=2*nf, num_heads=nf, bias=False))
        self.up_1_m = nn.Sequential(
            LayerNorm(2*nf, 'BiasFree'),
            FeedForward(dim=2*nf, ffn_expansion_factor=2, bias=False))

        self.up_2 = nn.Sequential(
            nn.ConvTranspose2d(2*nf, nf, 4, stride=2, padding=1),
            nn.ReLU(inplace=True))
        self.up_2_a = nn.Sequential(
            LayerNorm(2*nf, 'BiasFree'),
            MDTA(dim=nf, num_heads=nf, bias=False))
        self.up_2_m = nn.Sequential(
            LayerNorm(nf, 'BiasFree'),
            FeedForward(dim=nf, ffn_expansion_factor=2, bias=False))

    def forward(self, x, y):
        y= self.fuse(y)
        #extract fea
        out1_a = self.conv1_a(torch.cat([x,y],dim=1)) + x
        out1_m = self.conv1_m(out1_a) + out1_a

        down1_y = self.down_1(y)
        down1 = self.down_1(out1_m)
        down1_a = self.down_1_a(torch.cat([down1,down1_y],dim=1)) + down1
        down1_m = self.down_1_m(down1_a) + down1_a
        #
        down2_y = self.down_2(down1_y)
        down2 = self.down_2(down1_m)
        down2_a = self.down_2_a(torch.cat([down2,down2_y],dim=1)) + down2
        down2_m = self.down_2_m(down2_a) + down2_a

        out2_1 = self.conv2_1(down2_m) + down2_m
        out_y = self.conv2_1(down2_y)+down2_y
        out2_a = self.conv2_a(torch.cat([out2_1,out_y],dim=1)) + out2_1
        out2_m = self.conv2_m(out2_a) + out2_a

        up1 = self.up_1(out2_m) + down1_m
        up1_y=self.up_1(out_y)+down1_y
        up1_a = self.up_1_a(torch.cat([up1,up1_y],dim=1)) + up1
        up1_m = self.up_1_m(up1_a) + up1_a

        up2 = self.up_2(up1_m) + out1_m
        up2_y=self.up_2(up1_y)+y
        up2_a = self.up_2_a(torch.cat([up2,up2_y],dim=1)) + up2
        up2_m = self.up_2_m(up2_a) + up2_a

        return up2_m
         
# ==========
# up-sample Net (Upnet)
# ==========

class UP(nn.Module):
    def __init__(self, in_nc, base_ks, scale_factor,nf):
        super(UP, self).__init__()
        self.shuffle = nn.Sequential(
                                nn.Conv2d(nf, scale_factor*scale_factor*nf, kernel_size=1, stride=1, padding=1//2),
                                nn.PReLU(),
                                nn.PixelShuffle(scale_factor))
        self.res_up = nn.Sequential(
                                nn.Conv2d(nf, scale_factor*nf,  base_ks, stride=1, padding=base_ks//2),
                                nn.PReLU(),
                                ResidualUpSample(nf*scale_factor,scale_factor))
        self.conv = SF(in_channels=nf, height=2, reduction=8,bias=False, n_feats=nf)
        for i in range(0, 2):
            setattr(
                self, 'br_conv{}'.format(i), BResnet(in_channels=nf, kernel_size=base_ks)
            )

    def forward(self, y):
        y_shuffle = self.shuffle(y)
        y_up = self.res_up(y)
        for i in range(0, 2):
            br = getattr(self, 'br_conv{}'.format(i))
            y_shuffle, y_up = br(y_shuffle,y_up)
        out = self.conv(torch.cat([y_shuffle, y_up], dim=1))
        return out

class UpSample(nn.Module):
    def __init__(self, in_channels, scale_factor, stride=2):
        super(UpSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(ResidualUpSample(in_channels))
            in_channels = int(in_channels // stride)

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x

class ResidualUpSample(nn.Module):
    def __init__(self, in_channels, scale_factor,bias=False):
        super(ResidualUpSample, self).__init__()

        self.top = nn.Sequential(nn.Conv2d(in_channels, in_channels,   1, stride=1, padding=0, bias=bias),
                                nn.PReLU(),
                                nn.ConvTranspose2d(in_channels, in_channels, 3, stride=2, padding=1, output_padding=1,bias=bias),
                                nn.PReLU(),
                                nn.Conv2d(in_channels, in_channels//2, 1, stride=1, padding=0, bias=bias))
        self.bot = nn.Sequential(nn.Upsample(scale_factor=scale_factor, mode='bicubic', align_corners=True),
                                nn.Conv2d(in_channels, in_channels//2, 1, stride=1, padding=0, bias=bias))

    def forward(self, x):
        top = self.top(x)
        bot = self.bot(x)
        out = top+bot
        return out

# ==========
# Attention Guided Enhancement (AGE) module
# ==========

class AGE(nn.Module):
    def __init__(self,  nf , base_ks=3,nb=3):
        super(AGE, self).__init__()

        for i in range(0, nb):
            setattr(
                self, 'br_conv1{}'.format(i), BResnet(in_channels=nf,kernel_size=base_ks)
            )
            setattr(
                self, 'br_conv2{}'.format(i), BResnet(in_channels=nf, kernel_size=base_ks)
            )

        self.alig1 = SF(in_channels=nf, height=2, reduction=8, bias=False, n_feats=nf)
        self.alig2 = SF(in_channels=nf, height=2, reduction=8, bias=False, n_feats=nf)
        self.br = BResnet(in_channels=2*nf, kernel_size=base_ks)
        self.fuse =nn.Sequential(
            SF(in_channels=nf, height=4, reduction=8, bias=False, n_feats=nf),
            nn.Conv2d(nf, 1, 1, padding=1 // 2)
        )
        self.nb = nb
    def forward(self, x,y,z):
        y = self.alig1(y)
        z = self.alig2(z)
        x1=x
        x2=x
        for i in range(0, self.nb):
            br1 = getattr(self, 'br_conv1{}'.format(i))
            x1, y = br1(x1, y)
            br2 = getattr(self, 'br_conv2{}'.format(i))
            x2, z = br2(x2, z)

        out1,out2 =self.br(torch.cat([x1,x2], dim=1),torch.cat([y,z], dim=1))
        out = self.fuse(torch.cat([out1,out2],dim=1))

        return out
        
# ==========
# Dual-Branch Residual Block (DBRB)
# ==========  

class BResnet(nn.Module):
    def __init__(self, in_channels,kernel_size):
        super(BResnet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=1, padding=kernel_size // 2),
            nn.PReLU(),
            DSTA(in_channels),
            nn.Conv2d(in_channels,in_channels,kernel_size,1,1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=1, padding=kernel_size // 2),
            nn.PReLU(),
            DSTA(in_channels),
            nn.Conv2d(in_channels,in_channels,kernel_size,1,1)
        )

        self.relu = nn.PReLU()

    def forward(self, x, y):
        x_out=self.conv1(x)
        y_out=self.conv2(y)
        out1 =self.relu(x_out+x+y_out)
        out2=self.relu(x_out+y+y_out)
        return out1,out2

class SF(nn.Module):
    def __init__(self, in_channels, height, reduction=8, bias=False, n_feats=48):
        super(SF, self).__init__()
        self.height = height
        d = max(int(in_channels / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, d, 1, padding=0, bias=bias),
            nn.PReLU())
        self.n_feats = n_feats
        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        inp_feats = inp_feats.view(inp_feats.shape[0], self.height, self.n_feats, inp_feats.shape[2], inp_feats.shape[3])
        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(inp_feats.shape[0], self.height, self.n_feats, 1, 1)
        # stx()
        attention_vectors = self.softmax(attention_vectors)
        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)
        return feats_V
  
# ==========
# RCVSR network
# ==========
class SR(nn.Module):

    def __init__(self, opts_dict):
        """
        Arg:
            opts_dict: network parameters defined in YAML.
        """
        super(SR, self).__init__()
        self.scale_factor = 2
        self.mode = 'bicubic'
        self.radius = opts_dict['radius']
        self.input_len = 2 * self.radius + 1
        self.in_nc = opts_dict['stdf']['in_nc']
        self.nf = opts_dict['stdf']['nf']
        
        self.ffnet = STDF(
            in_nc=self.in_nc * self.input_len,
            out_nc=opts_dict['stdf']['out_nc'],
            nf=opts_dict['stdf']['nf'],
            nb=opts_dict['stdf']['nb'],
            deform_ks=opts_dict['stdf']['deform_ks']
        )
        
        for i in range(2):
            setattr(
                self, 'self_alig{}'.format(i), AligNet(
            in_nc=self.in_nc,
            out_nc=opts_dict['self_alig']['out_nc'],
            nf=opts_dict['self_alig']['nf'],
            deform_ks=opts_dict['self_alig']['deform_ks'])
            )
            setattr(
                self, 'extractor{}'.format(i),FEnet(
            base_ks=opts_dict['extractor']['base_ks'],
            in_nc=opts_dict['extractor']['in_nc'],
            nf=opts_dict['extractor']['nf'])
            )

        self.fusion = RRM(
            in_nc=opts_dict['fusion']['in_nc'],
            base_ks=opts_dict['fusion']['base_ks'],
            out_nc=opts_dict['fusion']['out_nc'],
            nf=opts_dict['fusion']['nf'],
            )

        self.upnet = UP(
            base_ks=opts_dict['upnet']['base_ks'],
            scale_factor=opts_dict['upnet']['scale_factor'],
            in_nc=opts_dict['upnet']['in_nc'],
            nf=opts_dict['upnet']['nf'],
        )

        self.attention = AGE(
            base_ks=opts_dict['attention']['base_ks'],
            nf=opts_dict['attention']['nf'],
            nb=opts_dict['attention']['nb'],
        )
        
    def forward(self, x,ref):
        frm_lst = [self.radius + idx_c * self.input_len for idx_c in range(self.in_nc)]
        frame = x[:, frm_lst, ...]
        
        # Neighborhood alignment (NA) block
        neibor_alig = self.ffnet(x)
        
        # Reference alignment (RA) block.
        up_frame = F.interpolate(frame, scale_factor=self.scale_factor, mode=self.mode, align_corners=True)
        up_fea =[]
        for i in range(2):
            extractor = getattr(self, 'extractor{}'.format(i))
            if len(ref[:, i, ...].shape)==3:
                fea = extractor(torch.cat([ref[:, i, ...].unsqueeze(dim=0), up_frame],dim=1))
            else:
                fea = extractor(torch.cat([ref[:, i, ...], up_frame],dim=1))
            up_fea.append(fea)

        up_fea = torch.cat(up_fea, dim=1)
        
        self_alig=[]
        for i in range(2):
            alignet =getattr(self, 'self_alig{}'.format(i))
            if len(ref[:, i, ...].shape) == 3:
                alig= alignet(up_fea[:,(i)*self.nf:(i+1)*self.nf,...],up_frame)
            else:
                alig = alignet(up_fea[:,(i)*self.nf:(i+1)*self.nf,...],up_frame)
            self_alig.append(alig)

        self_alig = torch.cat(self_alig,dim=1)
        
        # upsample
        up_out = self.upnet(neibor_alig)
        
        #Ref-based Refinement Module (RRM)
        out = self.fusion(up_out,torch.cat([up_fea,self_alig],dim=1))
        
        # Attention Guided Enhancement (AGE) module
        out = self.attention(out,self_alig,up_fea)
        out = up_frame+out

        return out