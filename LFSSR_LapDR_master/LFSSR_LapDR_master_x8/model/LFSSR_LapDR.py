import torch
import torch.nn.functional as functional
import torch.nn as nn
from einops import rearrange
import math
from model.module import *


# ===================================== 主模型定义 =================================
class get_model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.angRes = args.angRes
        self.channels = args.channels

        # 1. HR参考浅层特征提取，三个高频+一个低频
        self.ref_sfe_low = ShallowFE(self.channels, res_num=0)
        self.ref_sfe_high1 = ShallowFE(self.channels, res_num=3)       # 8x分辨率 (H,W,U,V,1)
        self.ref_sfe_high2 = ShallowFE(self.channels, res_num=2)       # 4x分辨率 (H/2,W/2,U,V,1)
        self.ref_sfe_high3 = ShallowFE(self.channels, res_num=1)       # 2x分辨率 (H/4,W/4,U,V,1)

        # 2. LR光场浅层特征提取
        self.lf_sfe_low = ShallowFE(self.channels, res_num=0)

        # 3. 偏移量预测
        self.offset_esti = LFOffsetEsti(self.channels, self.angRes)

        # 4. 可变性卷积分支
        self.Deform_low_align = RefLowAlignLF(self.channels, self.angRes)
        self.Deform_high_align = RefHighAlignLF(self.channels, self.angRes)

        # 5. 交叉注意力计算
        self.cross_attn_esti = CrossAttnEsti(self.channels)

        # 6. 交叉注意力精细对齐
        self.Cross_attn_align = AttnFineAlign(self.channels)

        # 7. SFT层特征嵌入和融合
        self.feat_up = UpSample(self.channels, factor=2)
        self.feat_sft = SFTLayer(self.channels)
        self.feat_lfg = DRLFModule(self.channels, self.angRes, lf_num=4)

        # 8. 高频重建
        self.hr_rec = RecConv(self.channels)

        # 8. 低频光场特征增强与低频光场图像重建
        self.lr_lfg = ResSASModule(self.channels, self.angRes, sas_num=4)
        self.lr_rec = RecConv(self.channels)

    def forward(self, in_lr, in_ref):
        # in_lr: [b,c,(ah h),(aw w)] (子孔径阵列形式), in_ref: [b,c,αh,αw]

        # ========================== 步骤1：参考图像进行拉普拉斯分解与浅层特征提取 ==========================
        pyr_list = lap_pyr_decom(in_ref, max_level=4)
        ref_feat_1 = self.ref_sfe_high1(pyr_list[0])       # 对应96x96特征, 高频, [b,c,h,w]
        ref_feat_2 = self.ref_sfe_high2(pyr_list[1])       # 对应48x48特征, 高频
        ref_feat_4 = self.ref_sfe_high3(pyr_list[2])       # 对应24x24特征, 高频
        ref_feat_8 = self.ref_sfe_low(pyr_list[3])         # 对应12x12特征, 低频

        # ============================== 步骤2：光场浅层特征提取 ==============================
        in_lr = rearrange(in_lr, 'b c (ah h) (aw w) -> (b ah aw) c h w', ah=self.angRes, aw=self.angRes)  # 转换为子孔径堆栈形式
        lr_lf_feat = self.lf_sfe_low(in_lr)       # LR光场图像,低频

        # =============================== 步骤3：偏移量预测 ===============================
        central_feat = rearrange(lr_lf_feat, '(b ah aw) c h w -> b ah aw c h w', ah=self.angRes, aw=self.angRes)
        central_feat = central_feat[:, self.angRes//2, self.angRes//2, :, :, :]     # 取出central LF feature [b,c,h/8,w/8]
        offset = self.offset_esti(ref_feat_8, central_feat, lr_lf_feat)

        # ============================== 步骤4：可变形卷积对齐与交叉注意力计算 ==============================
        align_feat_8, lr_lf_feat = self.Deform_low_align(ref_feat_8, lr_lf_feat, offset)     # [h/8,w/8], Low-frequency feature
        align_attn = self.cross_attn_esti(lr_lf_feat, align_feat_8)

        # ============================= 步骤5：可变形卷积对齐 =============================
        align_feat_1 = self.Deform_high_align(ref_feat_1, offset, scale=8)      # [h,w], High-frequency feature
        align_feat_2 = self.Deform_high_align(ref_feat_2, offset, scale=4)      # [h/2,w/2], High-frequency feature
        align_feat_4 = self.Deform_high_align(ref_feat_4, offset, scale=2)      # [h/4,w/4], High-frequency feature

        # ============================= 步骤6：交叉注意力对齐 =============================
        align_feat_1 = self.Cross_attn_align(align_feat_1, align_attn, scale=8)
        align_feat_2 = self.Cross_attn_align(align_feat_2, align_attn, scale=4)
        align_feat_4 = self.Cross_attn_align(align_feat_4, align_attn, scale=2)

        # ============================ 步骤7：SFT调制和SAV融合 =========================
        lf_up_feat4 = self.feat_up(lr_lf_feat)        # 2x up-sampling
        fused_feat_4 = self.feat_sft(lf_up_feat4, align_feat_4)
        fused_feat_4 = self.feat_lfg(fused_feat_4)

        lf_up_feat2 = self.feat_up(fused_feat_4)      # 2x up-sampling
        fused_feat_2 = self.feat_sft(lf_up_feat2, align_feat_2)
        fused_feat_2 = self.feat_lfg(fused_feat_2)

        lf_up_feat1 = self.feat_up(fused_feat_2)      # 2x up-sampling
        fused_feat_1 = self.feat_sft(lf_up_feat1, align_feat_1)
        fused_feat_1 = self.feat_lfg(fused_feat_1)

        # ============================= 步骤8：高频图像重建 =============================
        rec_hf_lf1 = self.hr_rec(fused_feat_1 + ref_feat_1.repeat_interleave(self.angRes ** 2, dim=0))
        rec_hf_lf2 = self.hr_rec(fused_feat_2 + ref_feat_2.repeat_interleave(self.angRes ** 2, dim=0))
        rec_hf_lf4 = self.hr_rec(fused_feat_4 + ref_feat_4.repeat_interleave(self.angRes ** 2, dim=0))

        # ============================= 步骤8：低频光场特征增强与重建 =============================
        enh_lf_feat = self.lr_lfg(lr_lf_feat)
        rec_lr_lf = self.lr_rec(enh_lf_feat)
        rec_lr_lf = rec_lr_lf + in_lr

        # ============================= 步骤10：拉普拉斯金字塔重建 =============================
        lap_recon_list = [rec_hf_lf1, rec_hf_lf2, rec_hf_lf4, rec_lr_lf]
        sr_8x = lap_pyr_recon(lap_recon_list)
        sr_8x = to_sai_array(sr_8x, self.angRes)
        return sr_8x


# ============================= 损失函数定义 =============================
class get_loss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.angRes = args.angRes
        self.l1 = CharbonnierLoss()

    def forward(self, sr, gt):
        pixel_loss = self.l1(sr, gt)
        sr_lap = lap_pyr_decom(to_sai_stack(sr, self.angRes), max_level=4)
        gt_lap = lap_pyr_decom(to_sai_stack(gt, self.angRes), max_level=4)
        lap_loss = (self.l1(sr_lap[0], gt_lap[0]) + self.l1(sr_lap[1], gt_lap[1]) +
                    self.l1(sr_lap[2], gt_lap[2]) + self.l1(sr_lap[3], gt_lap[3]))
        total_loss = pixel_loss + lap_loss
        return total_loss


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))
        return loss


# ===================== 权重初始化 =====================
def weights_init(m):
    pass


def to_sai_array(x, ang_res):
    x = rearrange(x, '(b ah aw) c h w -> b c (ah h) (aw w)', ah=ang_res, aw=ang_res)
    return x


def to_sai_stack(x, ang_res):
    x = rearrange(x, 'b c (ah h) (aw w) -> (b ah aw) c h w', ah=ang_res, aw=ang_res)
    return x


# ===================== 测试代码 =====================
if __name__ == '__main__':
    from thop import profile
    import argparse
    parser = argparse.ArgumentParser(description="Light field image super-resolution -- train mode")
    parser.add_argument("--angRes", type=int, default=5, help="Angular resolution of light field")
    parser.add_argument("--channels", type=int, default=48, help="Number of channels")
    args = parser.parse_args()

    model = get_model(args).cuda()
    lr = torch.randn(1, 1, 5 * 12, 5 * 12).cuda()     # [b,1,u*h,v*w]
    ref = torch.randn(1, 1, 96, 96).cuda()            # [b,1,H_ref,W_ref]
    gt = torch.randn(1, 1, 5 * 96, 5 * 96).cuda()     # [b,1,u*h,v*w]

    lap_list = lap_pyr_decom(ref, max_level=4)
    fin_rec = lap_pyr_recon(lap_list)
    value = torch.mean((fin_rec - ref) ** 2)
    print('MSE:', value)

    out_x = model(lr, ref)
    print('SR resolution:', out_x.shape)

    loss = get_loss(args).cuda()
    x = loss(out_x, gt)
    print(x)

    total = sum([param.nelement() for param in model.parameters()])
    flops, params = profile(model, inputs=(lr, ref,))
    print('Number of parameters: %.2fM' % (total / 1e6))
    print('Number of FLOPs: %.2fG' % (flops / 1e9))

