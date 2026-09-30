import torch
import torch.nn as nn
import torch.nn.functional as functional
from torchvision.ops import deform_conv2d
import numpy as np
from einops import rearrange


##########################################  拉普拉斯金字塔分解与重建  ##########################################
# 拉普拉斯金字塔分解
def lap_pyr_decom(in_x, max_level=4):
    # 从最高分辨率开始，构建高斯金字塔
    gaussian_list = [in_x]
    for _ in range(max_level - 1):
        in_x = functional.interpolate(in_x, scale_factor=0.5, mode='bicubic', align_corners=False, antialias=True)  # 下采样
        gaussian_list.append(in_x)

    # 从最高分辨率开始，构建拉普拉斯金字塔
    laplacian_list = []
    for k in range(max_level - 1):
        gauss_current = gaussian_list[k]
        gauss_up = functional.interpolate(gaussian_list[k + 1], scale_factor=2, mode='bicubic', align_corners=False, antialias=True)
        laplacian = gauss_current - gauss_up
        laplacian_list.append(laplacian)

    laplacian_list.append(gaussian_list[-1])    # 从高频到低频排列
    return laplacian_list


# 拉普拉斯金字塔重建
def lap_pyr_recon(laplacian_list):
    # 从最底层低频开始，恢复图像
    current = laplacian_list[-1]
    for k in range(len(laplacian_list) - 2, -1, -1):
        current = functional.interpolate(current, scale_factor=2, mode='bicubic', align_corners=False, antialias=True)
        current = current + laplacian_list[k]
    return current


###############################################  核心模块  ###############################################
class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.res_conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.res_conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, in_x):
        # [(b,ah,aw),c,h,w]
        res_x = self.act(self.res_conv1(in_x))
        res_x = self.res_conv2(res_x)
        out_x = res_x + in_x
        return out_x


###########################  浅层特征提取模块 (Shallow Feature Extraction)  ###########################
class ShallowFE(nn.Module):
    def __init__(self, channels, res_num=1):
        super(ShallowFE, self).__init__()
        self.in_conv = nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.res_block = nn.ModuleList([ResBlock(channels) for _ in range(res_num)])
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, in_x):
        # [(b,ah,aw),c,h,w]
        out_x = self.act(self.in_conv(in_x))

        for res_ind in self.res_block:
            out_x = res_ind(out_x)

        return out_x


#######################  可变形对齐 (Deformable Alignment)  ###############################
class ResASPP(nn.Module):
    def __init__(self, channels):
        super(ResASPP, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.conv_2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.conv_3 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=4, dilation=4, bias=False)
        self.conv_r = nn.Conv2d(in_channels=channels*3, out_channels=channels, kernel_size=1, stride=1, padding=0)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, in_x):
        # [b,c,h,w]
        res_x1 = self.act(self.conv_1(in_x))
        res_x2 = self.act(self.conv_2(in_x))
        res_x3 = self.act(self.conv_3(in_x))

        res_x = self.conv_r(torch.cat((res_x1, res_x2, res_x3), dim=1))
        out_x = res_x + in_x
        return out_x


# 偏移量估计，用于可变形卷积
class OffsetEstimation(nn.Module):
    def __init__(self, channels):
        super(OffsetEstimation, self).__init__()
        self.conv_cat = nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.ASPP = ResASPP(channels)
        self.conv_off = nn.Conv2d(in_channels=channels, out_channels=2 * 9, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_off.lr_mult = 0.1
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def init_offset(self):
        self.conv_off.weight.data.zero_()

    def forward(self, in_aux, in_tar):
        # [b,c,h,w]
        cat_x = torch.cat((in_aux, in_tar), dim=1)
        buffer = self.act(self.conv_cat(cat_x))
        buffer = self.ASPP(buffer)
        offset = self.conv_off(buffer)      # [b,18,h,w]
        return offset


# 2D参考特征与光场中心视图特征的偏移量估计
class RefToCenter(nn.Module):
    def __init__(self, channels):
        super(RefToCenter, self).__init__()
        self.OE = OffsetEstimation(channels)

    def forward(self, in_ref, in_center):
        # [b,c,h,w]
        off_x = self.OE(in_ref, in_center)
        return off_x


# 光场中心视图特征与周边视图特征的偏移量估计
class CenterToBoundary(nn.Module):
    def __init__(self, channels, ang_res):
        super(CenterToBoundary, self).__init__()
        self.ang_res = ang_res
        self.OE = OffsetEstimation(channels)

    def forward(self, in_center, in_lf):
        # [b,c,h,w] && [b*ah*aw,c,h,w]
        in_center = torch.repeat_interleave(in_center, self.ang_res ** 2, dim=0)     # 将中心视图复制U×V倍
        off_x = self.OE(in_center, in_lf)
        return off_x


# 两个偏移量估计完相加，得到最终偏移量
class LFOffsetEsti(nn.Module):
    def __init__(self, channels, ang_res):
        super(LFOffsetEsti, self).__init__()
        self.ang_res = ang_res
        self.R2C = RefToCenter(channels)
        self.C2B = CenterToBoundary(channels, ang_res)

    def forward(self, in_ref, in_center, in_lf):
        # [b,c,h,w] && [b,c,h,w] && [buv,c,h,w]
        off_x1 = self.R2C(in_ref, in_center)
        off_x2 = self.C2B(in_center, in_lf)
        off_x = torch.repeat_interleave(off_x1, self.ang_res ** 2, dim=0) + off_x2
        return off_x


# 利用偏移量将参考高频特征与光场特征进行对齐
class RefHighAlignLF(nn.Module):
    def __init__(self, channels, ang_res):
        super(RefHighAlignLF, self).__init__()
        self.ang_res = ang_res
        self.weight = nn.Parameter(torch.Tensor(channels, channels, 3, 3))
        nn.init.kaiming_uniform_(self.weight, 0.1)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, in_ref, in_offset, scale):
        # [b,c,h,w]
        in_ref = torch.repeat_interleave(in_ref, self.ang_res ** 2, dim=0)
        off_ref = functional.interpolate(in_offset, scale_factor=scale, mode='bilinear', align_corners=False, antialias=True)  # 上采样
        off_ref = off_ref * scale        # 乘上上采样因子进行重尺度化
        align_ref = self.act(deform_conv2d(in_ref, off_ref, self.weight, padding=(1, 1)))
        return align_ref


# 利用偏移量将参考特征与光场特征进行对齐
class RefLowAlignLF(nn.Module):
    def __init__(self, channels, ang_res):
        super(RefLowAlignLF, self).__init__()
        self.ang_res = ang_res
        self.weight = nn.Parameter(torch.Tensor(channels, channels, 3, 3))
        nn.init.kaiming_uniform_(self.weight, 0.1)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, in_ref, in_lf, in_offset):
        # [b,c,h,w]
        in_ref = torch.repeat_interleave(in_ref, self.ang_res ** 2, dim=0)
        align_ref = self.act(deform_conv2d(in_ref, in_offset, self.weight, padding=(1, 1)))

        zero_offset = torch.zeros_like(in_offset)
        conv_lf = self.act(deform_conv2d(in_lf, zero_offset, self.weight, padding=(1, 1)))
        return align_ref, conv_lf


#######################  交叉注意力精细对齐 (Cross-Attention Fine Alignment)  ###############################
# 估计交叉注意力矩阵
class CrossAttnEsti(nn.Module):
    def __init__(self, channels, top_k=4):
        super(CrossAttnEsti, self).__init__()
        self.top_k = top_k
        self.temperature = nn.Parameter(torch.ones(1))
        self.q_conv = nn.Conv2d(in_channels=channels, out_channels=channels//4, kernel_size=1, stride=1, padding=0, bias=False)
        self.k_conv = nn.Conv2d(in_channels=channels, out_channels=channels//4, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, in_tar, in_coarse):
        # [b*ah*aw,c,h,w]
        b, c, h, w = in_tar.shape
        n = h * w
        query = self.q_conv(in_tar)
        key = self.k_conv(in_coarse)

        # reshape to 2D and normalization
        query = query.view(b, -1, n).permute(0, 2, 1)   # [b,hw,c]
        key = key.view(b, -1, n).permute(0, 2, 1)       # [b,hw,c]

        query = functional.normalize(query, dim=-1)
        key = functional.normalize(key, dim=-1)

        # Cross-attention
        raw_attn = torch.bmm(query, key.transpose(-2, -1))  # [b,hw,hw]
        raw_attn = raw_attn / (self.temperature + 1e-6)

        # Top-K sparsification
        top_k_values, top_k_indices = torch.topk(raw_attn, k=self.top_k, dim=-1)
        sparse_mask = torch.full_like(raw_attn, float('-inf'))
        sparse_attn = sparse_mask.scatter(-1, top_k_indices, top_k_values)

        att_map = functional.softmax(sparse_attn, dim=-1)   # [b,hw,hw]
        return att_map


# 基于交叉注意力矩阵的精细对齐
class AttnFineAlign(nn.Module):
    def __init__(self, channels):
        super(AttnFineAlign, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=1, stride=1, padding=0, bias=False)

    def __call__(self, in_x, in_attn, scale):
        # [b*ah*aw,c,h,w]
        # HR into patches, and reshape [b*ah*aw,lr_h,lr_w,c,s,s]
        # [b*ah*aw,c*s*s,lr_h*lr_w]
        value_patch = functional.unfold(in_x, kernel_size=scale, stride=scale)

        # Attention align
        align_out = torch.bmm(value_patch, in_attn.permute(0, 2, 1))
        align_out = functional.fold(align_out, output_size=(in_x.shape[2], in_x.shape[3]), kernel_size=scale, stride=scale)

        out = self.conv1x1(torch.cat((in_x, align_out), dim=1))
        return out


#######################  配准高分辨特征嵌入光场特征中  ###############################
class SFTLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.SFT_conv_a1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.SFT_conv_a2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.SFT_conv_b1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.SFT_conv_b2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, in_lf, in_ref):
        feat_a = self.act(self.SFT_conv_a1(in_lf))
        feat_a = self.SFT_conv_a2(feat_a)

        feat_b = self.act(self.SFT_conv_b1(in_lf))
        feat_b = self.SFT_conv_b2(feat_b)
        return feat_a * in_ref + feat_b


#######################  光场特征提取模块  ###############################
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels//reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=channels//reduction, out_channels=channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid())

    def forward(self, in_x):
        avg_x = (self.avg_pool(in_x))
        ca_x = self.conv_du(avg_x)
        out_x = ca_x * in_x
        return out_x


class CSBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CSBlock, self).__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=channels, out_channels=channels//reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.sa = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, bias=False)
        self.fuse_conv = nn.Conv2d(in_channels=channels//reduction+1, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, in_x):
        _, c, h, w = in_x.shape
        ca_weight = self.ca(in_x).expand(-1, -1, h, w)

        avg_out = torch.mean(in_x, dim=1, keepdim=True)
        max_out, _ = torch.max(in_x, dim=1, keepdim=True)
        sa_weight = self.sa(torch.cat((avg_out, max_out), dim=1))

        mixed_weight = torch.cat((ca_weight, sa_weight), dim=1)
        mixed_attn = self.sigmoid(self.fuse_conv(mixed_weight))
        out_x = mixed_attn * in_x
        return out_x


class LFFEBlock(nn.Module):
    def __init__(self, channels, ang_res, mode):
        super(LFFEBlock, self).__init__()
        self.ang_res = ang_res
        self.mode = mode
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, in_x):
        # [b*ah*aw, c, h, w]
        _, _, h, w = in_x.shape

        if self.mode == 'spatial':
            res_x = self.act(self.conv1(in_x))
            res_x = self.conv2(res_x)
            out_x = res_x + in_x

        elif self.mode == 'hepi':
            in_x = rearrange(in_x, '(b ah aw) c h w -> (b ah h) c aw w', ah=self.ang_res, aw=self.ang_res)
            res_x = self.act(self.conv1(in_x))
            res_x = self.conv2(res_x)
            out_x = res_x + in_x
            out_x = rearrange(out_x, '(b ah h) c aw w -> (b ah aw) c h w', ah=self.ang_res, h=h)

        elif self.mode == 'vepi':
            in_x = rearrange(in_x, '(b ah aw) c h w -> (b aw w) c ah h', ah=self.ang_res, aw=self.ang_res)
            res_x = self.act(self.conv1(in_x))
            res_x = self.conv2(res_x)
            out_x = res_x + in_x
            out_x = rearrange(out_x, '(b aw w) c ah h -> (b ah aw) c h w', aw=self.ang_res, w=w)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        return out_x


class DistilledResLFBlock(nn.Module):
    def __init__(self, channels, ang_res, distillation_rate=0.5):
        super(DistilledResLFBlock, self).__init__()
        self.distilled_channels = int(channels * distillation_rate)
        self.extract1 = LFFEBlock(channels, ang_res, 'spatial')
        self.distill1 = nn.Conv2d(in_channels=channels, out_channels=self.distilled_channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.extract2 = LFFEBlock(channels, ang_res, 'hepi')
        self.distill2 = nn.Conv2d(in_channels=channels, out_channels=self.distilled_channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.extract3 = LFFEBlock(channels, ang_res, 'vepi')
        self.distill3 = nn.Conv2d(in_channels=channels, out_channels=self.distilled_channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.fuse = nn.Conv2d(in_channels=self.distilled_channels * 3, out_channels=channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.csb = CSBlock(channels)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, in_x):
        # [b*ah*aw, c, h, w]
        retained_1 = self.extract1(in_x)
        distilled_1 = self.act(self.distill1(retained_1))

        retained_2 = self.extract2(retained_1)
        distilled_2 = self.act(self.distill2(retained_2))

        retained_3 = self.extract3(retained_2)
        distilled_3 = self.act(self.distill3(retained_3))

        out_x = torch.cat((distilled_1, distilled_2, distilled_3), dim=1)
        out_x = self.fuse(out_x)
        out_x = self.csb(out_x)
        return out_x + in_x


class DRLFGroup(nn.Module):
    def __init__(self, channels, ang_res):
        super(DRLFGroup, self).__init__()
        self.lf_block = nn.ModuleList([DistilledResLFBlock(channels, ang_res) for _ in range(4)])
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, in_x):
        # [b*ah*aw,c,h,w]
        out_x = in_x
        for lfb_ind in self.lf_block:
            out_x = lfb_ind(out_x)

        out_x = self.conv(self.act(out_x))
        return out_x + in_x


class DRLFModule(nn.Module):
    def __init__(self, channels, ang_res, lf_num):
        super(DRLFModule, self).__init__()
        self.lf_module = nn.ModuleList([DRLFGroup(channels, ang_res) for _ in range(lf_num)])

    def forward(self, in_x):
        # [(b,ah,aw),c,h,w]
        out_x = in_x
        for lfm_ind in self.lf_module:
            out_x = lfm_ind(out_x)
        return out_x


class ResSASBlock(nn.Module):
    def __init__(self, channels, ang_res):
        super(ResSASBlock, self).__init__()
        self.ang_res = ang_res
        self.spa_conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.ang_conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.spa_conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, in_x):
        # [(b,ah,aw),c,h,w]
        spa_x = self.act(self.spa_conv1(in_x))

        ang_x = rearrange(spa_x, '(b ah aw) c h w -> (b h w) c ah aw', ah=self.ang_res, aw=self.ang_res)
        ang_x = self.act(self.ang_conv(ang_x))
        res_x = rearrange(ang_x, '(b h w) c ah aw -> (b ah aw) c h w', h=in_x.shape[2], w=in_x.shape[3])

        res_x = self.spa_conv2(res_x)
        out_x = res_x + in_x
        return out_x


class ResSASModule(nn.Module):
    def __init__(self, channels, ang_res, sas_num):
        super(ResSASModule, self).__init__()
        self.sas_module = nn.ModuleList([ResSASBlock(channels, ang_res) for _ in range(sas_num)])

    def forward(self, in_x):
        # [(b,ah,aw),c,h,w]
        out_x = in_x
        for sas_ind in self.sas_module:
            out_x = sas_ind(out_x)
        return out_x


###########################################  特征上采样模块  ###########################################
class UpSample(nn.Module):
    def __init__(self, channels, factor):
        super(UpSample, self).__init__()
        self.up_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels * factor ** 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.PixelShuffle(factor),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))

    def forward(self, in_x):
        # [b*ah*aw,c,h,w]
        out_x = self.up_conv(in_x)
        return out_x


##########################################  将特征重建为图像  ##########################################
class RecConv(nn.Module):
    def __init__(self, channels):
        super(RecConv, self).__init__()
        self.rec_conv = nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, in_x):
        # [b*ah*aw,c,h,w]
        out_x = self.rec_conv(in_x)
        return out_x
