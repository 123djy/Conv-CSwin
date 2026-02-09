
# import os
# import sys
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import timm
# from timm.models.layers import trunc_normal_
# from IMDLBenCo.registry import MODELS
# from extractor.high_frequency_feature_extraction import HighDctFrequencyExtractor
# from extractor.low_frequency_feature_extraction import LowDctFrequencyExtractor
#
# # ===== 导入 CSWin =====
# sys.path.append('/data/jdon492/Mesorch_with_pretrain_weight/CSWin-Transformer')
# from models.cswin import CSWinTransformer
#
#
# class DiceLoss(nn.Module):
#     def __init__(self, eps=1e-6):
#         super().__init__()
#         self.eps = eps
#
#     def forward(self, pred, target):
#         pred = torch.sigmoid(pred)
#         num = 2 * (pred * target).sum(dim=(1, 2, 3))
#         den = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + self.eps
#         dice = 1 - (num + self.eps) / (den + self.eps)
#         return dice.mean()
#
#
# class LearnableFusion(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.gate = nn.Sequential(
#             nn.Conv2d(channels * 2, channels, 1),
#             nn.GELU(),
#             nn.Conv2d(channels, 1, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, c_feat, t_feat):
#         g = self.gate(torch.cat([c_feat, t_feat], dim=1))
#         return c_feat * (1 - g) + t_feat * g
#
#
# # ---------------------------
# # ConvNeXt 分支（RGB + 高频）
# # ---------------------------
# class ConvNeXt(timm.models.convnext.ConvNeXt):
#     def __init__(self, conv_pretrain=False):
#         super().__init__(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768))
#         if conv_pretrain:
#             model = timm.create_model('convnext_tiny', pretrained=True)
#             self.load_state_dict(model.state_dict(), strict=False)
#
#         orig = self.stem[0]
#         new = nn.Conv2d(6, orig.out_channels, orig.kernel_size,
#                         stride=orig.stride, padding=orig.padding, bias=False)
#         with torch.no_grad():
#             new.weight[:, :3] = orig.weight
#             nn.init.kaiming_normal_(new.weight[:, 3:])
#         self.stem[0] = new
#
#     def forward_features(self, x):
#         x = self.stem(x)
#         outs = []
#         for stage in self.stages:
#             x = stage(x)
#             outs.append(x)
#         x = self.norm_pre(x)
#         return x, outs
#
#
# # ---------------------------
# # CSWin 分支（RGB + 低频）
# # ---------------------------
# class CSWinBackbone(nn.Module):
#     """
#     ✅ 适用于篡改检测的 CSWin-Base 主干
#     - 自动兼容 state_dict / state_dict_ema / model 三种格式
#     - 自动去掉 module. 前缀
#     - 自动适配 6 通道输入
#     - 自动插值 pos_embed (224 → 512)
#     """
#
#     def __init__(self, in_chans=6, img_size=512):
#         super().__init__()
#         print(f"✅ Using CSWin-Base (img={img_size}, in_chans={in_chans})")
#
#         # ------------------------------------------------------------
#         # 1️⃣ 初始化 CSWin 基础骨干
#         # ------------------------------------------------------------
#         self.backbone = CSWinTransformer(
#             img_size=img_size,
#             patch_size=4,
#             in_chans=in_chans,  # 支持 RGB + 高频/低频
#             embed_dim=96,
#             depth=[2, 4, 32, 2],
#             split_size=[1, 2, 8, 8],
#             num_heads=[4, 8, 16, 32],
#             mlp_ratio=4.0,
#             drop_rate=0.0,
#             attn_drop_rate=0.0,
#             drop_path_rate=0.1,
#             use_chk=False,
#         )
#
#         # ------------------------------------------------------------
#         # 2️⃣ 替换首层卷积为 6 通道 (在加载前)
#         # ------------------------------------------------------------
#         patch_conv = self.backbone.stage1_conv_embed[0]
#         old_w = patch_conv.weight.clone()
#         new_conv = nn.Conv2d(
#             in_chans,
#             patch_conv.out_channels,
#             kernel_size=patch_conv.kernel_size,
#             stride=patch_conv.stride,
#             padding=patch_conv.padding,
#             bias=False,
#         )
#         with torch.no_grad():
#             if old_w.shape[1] == 3:
#                 # 复制前 3 通道 (RGB)
#                 new_conv.weight[:, :3] = old_w
#                 # 初始化新增通道 (DCT 或频域信息)
#                 nn.init.kaiming_normal_(new_conv.weight[:, 3:])
#         self.backbone.stage1_conv_embed[0] = new_conv
#         print("✅ Modified first conv layer → 6 channels")
#
#         # ------------------------------------------------------------
#         # 3️⃣ 加载微软官方预训练权重 (支持 EMA / 去前缀 / 插值)
#         # ------------------------------------------------------------
#         weight_path = "/data/jdon492/Mesorch_with_pretrain_weight/CSWin-Transformer/cswin_base_224.pth"
#         if os.path.exists(weight_path):
#             print(f"📦 Loading pretrained weights from: {weight_path}")
#             state = torch.load(weight_path, map_location="cpu")
#
#             # ---- 兼容各种保存格式 ----
#             if "state_dict_ema" in state:
#                 state = state["state_dict_ema"]
#                 print("📘 Using EMA weights (state_dict_ema)")
#             elif "state_dict" in state:
#                 state = state["state_dict"]
#                 print("📗 Using state_dict weights")
#             elif "model" in state:
#                 state = state["model"]
#                 print("📙 Using model weights")
#
#             # ---- 去掉 module. 前缀 ----
#             state = {k.replace("module.", ""): v for k, v in state.items()}
#
#             # ---- 手动适配 6通道卷积 ----
#             conv_key = "stage1_conv_embed.0.weight"
#             if conv_key in state:
#                 old_w = state[conv_key]
#                 print(f"🔧 Adjusting first conv weight: {old_w.shape} → (96, 6, 7, 7)")
#                 new_w = torch.zeros((96, 6, 7, 7))
#                 new_w[:, :3, :, :] = old_w  # 复制前3通道
#                 nn.init.kaiming_normal_(new_w[:, 3:, :, :])  # 初始化后3通道
#                 state[conv_key] = new_w
#
#             # ---- 插值位置编码 (224 → 512) ----
#             if "pos_embed" in state:
#                 pos_embed = state["pos_embed"]
#                 old_size = int(pos_embed.shape[1] ** 0.5)
#                 new_size = img_size // 4
#                 if old_size != new_size:
#                     print(f"🔄 Interpolating pos_embed: {old_size} → {new_size}")
#                     pos_embed = pos_embed.permute(0, 2, 1).reshape(1, -1, old_size, old_size)
#                     pos_embed = F.interpolate(
#                         pos_embed,
#                         size=(new_size, new_size),
#                         mode="bicubic",
#                         align_corners=False,
#                     )
#                     state["pos_embed"] = pos_embed.flatten(2).permute(0, 2, 1)
#
#             # ---- 加载权重 ----
#             msg = self.backbone.load_state_dict(state, strict=False)
#             print(f"✅ CSWin pretrained loaded successfully! "
#                   f"(missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)})")
#
#             if len(msg.missing_keys) > 0:
#                 print("🔍 Missing keys (前30个示例):")
#                 print("\n".join(msg.missing_keys[:30]))
#             if len(msg.missing_keys) < 10:
#                 print("🎯 All major weights loaded correctly!")
#             else:
#                 print("⚠️ Many missing keys — check split_size or repo version mismatch.")
#         else:
#             print("⚠️ Warning: CSWin pretrained weight not found — training from scratch!")
#
#         # ------------------------------------------------------------
#         # 4️⃣ 提取多阶段输出模块 (for feature fusion)
#         # ------------------------------------------------------------
#         self.stage1 = self.backbone.stage1
#         self.stage2 = self.backbone.stage2
#         self.stage3 = self.backbone.stage3
#         self.stage4 = self.backbone.stage4
#         self.merge1 = self.backbone.merge1
#         self.merge2 = self.backbone.merge2
#         self.merge3 = self.backbone.merge3
#         self.norm = self.backbone.norm
#
#     # ------------------------------------------------------------
#     # 5️⃣ 前向传播：提取4阶段特征输出
#     # ------------------------------------------------------------
#     def forward(self, x):
#         B = x.shape[0]
#         x = self.backbone.stage1_conv_embed(x)
#         H = W = int(x.shape[1] ** 0.5)
#         x1 = x.transpose(-2, -1).contiguous().view(B, 96, H, W)
#
#         # Stage1
#         for blk in self.stage1:
#             x = blk(x)
#         x = self.merge1(x)
#
#         # Stage2
#         H = W = int(x.shape[1] ** 0.5)
#         x2 = x.transpose(-2, -1).contiguous().view(B, 192, H, W)
#         for blk in self.stage2:
#             x = blk(x)
#         x = self.merge2(x)
#
#         # Stage3
#         H = W = int(x.shape[1] ** 0.5)
#         x3 = x.transpose(-2, -1).contiguous().view(B, 384, H, W)
#         for blk in self.stage3:
#             x = blk(x)
#         x = self.merge3(x)
#
#         # Stage4
#         H = W = int(x.shape[1] ** 0.5)
#         x4 = x.transpose(-2, -1).contiguous().view(B, 768, H, W)
#         for blk in self.stage4:
#             x = blk(x)
#
#         # Norm + reshape
#         x = self.norm(x)
#         H = W = int(x.shape[1] ** 0.5)
#         x4 = x.transpose(-2, -1).contiguous().view(B, 768, H, W)
#
#         return x4, [x1, x2, x3, x4]
#
#
# # ---------------------------
# # 全局上下文模块（注意力引导）
# # ---------------------------
# class GlobalContext(nn.Module):
#     def __init__(self, in_ch=768):
#         super().__init__()
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(in_ch, in_ch // 4, bias=False),
#             nn.GELU(),
#             nn.Linear(in_ch // 4, in_ch, bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         B, C, _, _ = x.shape
#         w = self.fc(self.pool(x).flatten(1)).view(B, C, 1, 1)
#         return x * w + x
#
#
# # ---------------------------
# # UpsampleConcatConv（原样保留）
# # ---------------------------
# class UpsampleConcatConv(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.upsamplec2 = nn.ConvTranspose2d(192, 96, 4, 2, 1)
#         self.upsamplec3 = nn.Sequential(
#             nn.ConvTranspose2d(384, 192, 4, 2, 1),
#             nn.ConvTranspose2d(192, 96, 4, 2, 1)
#         )
#         self.upsamplec4 = nn.Sequential(
#             nn.ConvTranspose2d(768, 384, 4, 2, 1),
#             nn.ConvTranspose2d(384, 192, 4, 2, 1),
#             nn.ConvTranspose2d(192, 96, 4, 2, 1)
#         )
#         self.upsamples2 = nn.ConvTranspose2d(192, 96, 4, 2, 1)
#         self.upsamples3 = nn.Sequential(
#             nn.ConvTranspose2d(384, 192, 4, 2, 1),
#             nn.ConvTranspose2d(192, 96, 4, 2, 1)
#         )
#         self.upsamples4 = nn.Sequential(
#             nn.ConvTranspose2d(768, 384, 4, 2, 1),
#             nn.ConvTranspose2d(384, 192, 4, 2, 1),
#             nn.ConvTranspose2d(192, 96, 4, 2, 1)
#         )
#
#     def forward(self, inputs):
#         c1, c2, c3, c4, s1, s2, s3, s4 = inputs
#         c2 = self.upsamplec2(c2);
#         c3 = self.upsamplec3(c3);
#         c4 = self.upsamplec4(c4)
#         s2 = self.upsamples2(s2);
#         s3 = self.upsamples3(s3);
#         s4 = self.upsamples4(s4)
#         x = torch.cat([c1, c2, c3, c4, s1, s2, s3, s4], dim=1)
#         return x, [c1, c2, c3, c4, s1, s2, s3, s4]
#
#
# # ---------------------------
# # Score Network（原样）
# # ---------------------------
# class LayerNorm2d(nn.LayerNorm):
#     def __init__(self, num_channels, eps=1e-6, affine=True):
#         super().__init__(num_channels, eps=eps, elementwise_affine=affine)
#
#     def forward(self, x):
#         x = x.permute(0, 2, 3, 1)
#         x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
#         return x.permute(0, 3, 1, 2)
#
#
# class ScoreNetwork(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(9, 192, 7, 2, 3)
#         self.invert = nn.Sequential(LayerNorm2d(192),
#                                     nn.Conv2d(192, 192, 3, 1, 1),
#                                     nn.Conv2d(192, 768, 1),
#                                     nn.Conv2d(768, 192, 1),
#                                     nn.GELU())
#         self.conv2 = nn.Conv2d(192, 8, 7, 2, 3)
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, x):
#         x = self.conv1(x);
#         short = x
#         x = self.invert(x);
#         x = short + x
#         x = self.conv2(x)
#         return self.softmax(x.float())
#
#
# # ---------------------------
# # 主模型 FullPower
# # ---------------------------
# @MODELS.register_module(force=True)
# class Mesorch_ConvNeXt_CSWinB(nn.Module):
#     def __init__(self, image_size=512, conv_pretrain=False):
#         super().__init__()
#
#         self.conv = ConvNeXt(conv_pretrain)
#         self.cswin = CSWinBackbone(in_chans=6, img_size=image_size)
#
#         self.fusion = nn.ModuleList([
#             LearnableFusion(96),
#             LearnableFusion(192),
#             LearnableFusion(384),
#             LearnableFusion(768)
#         ])
#
#         self.decoder = UpsampleConcatConv()
#
#         self.gate = ScoreNetwork()
#         self.inverse = nn.ModuleList([nn.Conv2d(96, 1, 1) for _ in range(8)])
#
#         self.resize = nn.Upsample(size=(image_size, image_size), mode='bilinear')
#         self.loss_bce = nn.BCEWithLogitsLoss()
#         self.loss_dice = DiceLoss()
#
#         self.low_dct = LowDctFrequencyExtractor()
#         self.high_dct = HighDctFrequencyExtractor()
#
#     def forward(self, image, mask=None, *args, **kwargs):
#         high = self.high_dct(image)
#         low = self.low_dct(image)
#
#         # backbone inputs
#         x_h = torch.cat([image, high], dim=1)  # ConvNeXt
#         x_l = torch.cat([image, low], dim=1)  # CSWin
#         x_all = torch.cat([image, high, low], dim=1)  # ScoreNet: 9ch (不动)
#
#         # 1) two backbones
#         _, feats_conv = self.conv.forward_features(x_h)  # [c1,c2,c3,c4]
#         _, feats_cswin = self.cswin(x_l)  # [s1,s2,s3,s4]
#
#         # 2) stage-wise fusion (得到每个尺度的“融合特征”)
#         fused = [self.fusion[i](feats_conv[i], feats_cswin[i]) for i in range(4)]  # [f1..f4]
#
#         # 3) ✅ 让 fusion 真正生效：把 fused 注入到两个分支（保持 8 路专家结构）
#         #    解释：每个尺度的两个专家都共享一个融合后的“共识特征”，但仍保留各自差异
#         feats_conv_f = [feats_conv[i] + fused[i] for i in range(4)]
#         feats_cswin_f = [feats_cswin[i] + fused[i] for i in range(4)]
#
#         # 4) decoder 输入仍是 8 路（符合原结构）
#         decoder_inputs = feats_conv_f + feats_cswin_f  # 8 maps: c1..c4 + s1..s4
#         x, feats = self.decoder(decoder_inputs)
#
#         # 5) gate（不改你的 gate 逻辑）
#         gate_w = self.gate(x_all)  # (B, 8, H/4, W/4) 取决于ScoreNet
#
#         # 6) inverse -> reduced -> weighted sum（保持不变）
#         reduced = torch.cat([self.inverse[i](feats[i]) for i in range(8)], dim=1)
#         pred = torch.sum(gate_w * reduced, dim=1, keepdim=True)
#         pred = self.resize(pred)
#
#         loss = None
#         if mask is not None:
#             loss = self.loss_bce(pred, mask) + self.loss_dice(pred, mask)
#
#         return {
#             "backward_loss": loss,
#             "pred_mask": torch.sigmoid(pred),
#             "visual_loss": {"predict_loss": loss},
#             "visual_image": {"pred_mask": torch.sigmoid(pred)},
#         }


import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from IMDLBenCo.registry import MODELS

from extractor.high_frequency_feature_extraction import HighDctFrequencyExtractor
from extractor.low_frequency_feature_extraction import LowDctFrequencyExtractor

# ===== CSWin repo path (按你自己环境改) =====
# 你原来是：/data/jdon492/Mesorch_with_pretrain_weight/CSWin-Transformer
CSWIN_REPO = "/data/jdon492/Mesorch_with_pretrain_weight/CSWin-Transformer"
if CSWIN_REPO not in sys.path:
    sys.path.append(CSWIN_REPO)

from models.cswin import CSWinTransformer


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        num = 2 * (pred * target).sum(dim=(1, 2, 3))
        den = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + self.eps
        dice = 1 - (num + self.eps) / (den + self.eps)
        return dice.mean()


# ---------------------------
# ConvNeXt 分支（RGB + 高频）
# ---------------------------
class ConvNeXt(timm.models.convnext.ConvNeXt):
    def __init__(self, conv_pretrain: bool = False):
        super().__init__(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768))
        if conv_pretrain:
            model = timm.create_model("convnext_tiny", pretrained=True)
            self.load_state_dict(model.state_dict(), strict=False)

        # 适配 6 通道输入
        orig = self.stem[0]
        new = nn.Conv2d(
            6,
            orig.out_channels,
            kernel_size=orig.kernel_size,
            stride=orig.stride,
            padding=orig.padding,
            bias=False,
        )
        with torch.no_grad():
            new.weight[:, :3] = orig.weight
            nn.init.kaiming_normal_(new.weight[:, 3:])
        self.stem[0] = new

    def forward_features(self, x):
        x = self.stem(x)
        outs = []
        for stage in self.stages:
            x = stage(x)
            outs.append(x)
        x = self.norm_pre(x)
        return x, outs


# ---------------------------
# CSWin 分支（RGB + 低频）
# ---------------------------
class CSWinBackbone(nn.Module):
    """
    CSWin-Base backbone, 输出 4 个 stage 的特征:
    [96, 192, 384, 768]，空间尺度依次 /4 /8 /16 /32
    """

    def __init__(
        self,
        in_chans: int = 6,
        img_size: int = 512,
        weight_path: str = "/data/jdon492/Mesorch_with_pretrain_weight/CSWin-Transformer/cswin_base_224.pth",
    ):
        super().__init__()
        print(f"✅ Using CSWin-Base (img={img_size}, in_chans={in_chans})")

        self.backbone = CSWinTransformer(
            img_size=img_size,
            patch_size=4,
            in_chans=in_chans,
            embed_dim=96,
            depth=[2, 4, 32, 2],
            split_size=[1, 2, 8, 8],
            num_heads=[4, 8, 16, 32],
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            use_chk=False,
        )

        # 先把第一层 conv 改成 in_chans
        patch_conv = self.backbone.stage1_conv_embed[0]
        old_w = patch_conv.weight.clone()
        new_conv = nn.Conv2d(
            in_chans,
            patch_conv.out_channels,
            kernel_size=patch_conv.kernel_size,
            stride=patch_conv.stride,
            padding=patch_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            if old_w.shape[1] == 3:
                new_conv.weight[:, :3] = old_w
                nn.init.kaiming_normal_(new_conv.weight[:, 3:])
        self.backbone.stage1_conv_embed[0] = new_conv
        print("✅ Modified first conv layer → 6 channels")

        # 加载预训练（可选）
        if weight_path and os.path.exists(weight_path):
            print(f"📦 Loading pretrained weights from: {weight_path}")
            state = torch.load(weight_path, map_location="cpu")

            if "state_dict_ema" in state:
                state = state["state_dict_ema"]
                print("📘 Using EMA weights (state_dict_ema)")
            elif "state_dict" in state:
                state = state["state_dict"]
                print("📗 Using state_dict weights")
            elif "model" in state:
                state = state["model"]
                print("📙 Using model weights")

            state = {k.replace("module.", ""): v for k, v in state.items()}

            # 适配 6 通道 conv
            conv_key = "stage1_conv_embed.0.weight"
            if conv_key in state:
                old = state[conv_key]  # (96,3,7,7)
                new = torch.zeros((old.shape[0], 6, old.shape[2], old.shape[3]))
                new[:, :3] = old
                nn.init.kaiming_normal_(new[:, 3:])
                state[conv_key] = new

            # pos_embed 插值（224 -> img_size）
            if "pos_embed" in state:
                pos = state["pos_embed"]
                old_size = int(pos.shape[1] ** 0.5)
                new_size = img_size // 4
                if old_size != new_size:
                    print(f"🔄 Interpolating pos_embed: {old_size} → {new_size}")
                    pos = pos.permute(0, 2, 1).reshape(1, -1, old_size, old_size)
                    pos = F.interpolate(pos, size=(new_size, new_size), mode="bicubic", align_corners=False)
                    state["pos_embed"] = pos.flatten(2).permute(0, 2, 1)

            msg = self.backbone.load_state_dict(state, strict=False)
            print(f"✅ CSWin pretrained loaded! (missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)})")
        else:
            print("⚠️ CSWin pretrained weight not found — training from scratch!")

        # 取出 stage 模块
        self.stage1 = self.backbone.stage1
        self.stage2 = self.backbone.stage2
        self.stage3 = self.backbone.stage3
        self.stage4 = self.backbone.stage4
        self.merge1 = self.backbone.merge1
        self.merge2 = self.backbone.merge2
        self.merge3 = self.backbone.merge3

    def forward(self, x):
        """
        返回：
            x_final, outs=[x1(96,H/4), x2(192,H/8), x3(384,H/16), x4(768,H/32)]
        """
        B = x.shape[0]

        x = self.backbone.stage1_conv_embed(x)  # (B, N, 96)
        H = W = int(x.shape[1] ** 0.5)
        x1 = x.transpose(-2, -1).contiguous().view(B, 96, H, W)

        # stage1
        for blk in self.stage1:
            x = blk(x)
        x = self.merge1(x)

        # stage2
        H = W = int(x.shape[1] ** 0.5)
        x2 = x.transpose(-2, -1).contiguous().view(B, 192, H, W)
        for blk in self.stage2:
            x = blk(x)
        x = self.merge2(x)

        # stage3
        H = W = int(x.shape[1] ** 0.5)
        x3 = x.transpose(-2, -1).contiguous().view(B, 384, H, W)
        for blk in self.stage3:
            x = blk(x)
        x = self.merge3(x)

        # stage4
        H = W = int(x.shape[1] ** 0.5)
        x4 = x.transpose(-2, -1).contiguous().view(B, 768, H, W)
        for blk in self.stage4:
            x = blk(x)

        return x, [x1, x2, x3, x4]


# ---------------------------
# Decoder：专门给 “ConvNeXt(96/192/384/768) + CSWin(96/192/384/768)”
# 把 8 路特征都 upsample 到 H/4，并 cat 成 x
# ---------------------------
class UpsampleConcatConvCSWin(nn.Module):
    def __init__(self):
        super().__init__()
        # conv branch
        self.up_c2 = nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1)
        self.up_c3 = nn.Sequential(
            nn.ConvTranspose2d(384, 192, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1),
        )
        self.up_c4 = nn.Sequential(
            nn.ConvTranspose2d(768, 384, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(384, 192, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1),
        )

        # cswin branch（同样的通道结构）
        self.up_s2 = nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1)
        self.up_s3 = nn.Sequential(
            nn.ConvTranspose2d(384, 192, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1),
        )
        self.up_s4 = nn.Sequential(
            nn.ConvTranspose2d(768, 384, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(384, 192, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, inputs):
        c1, c2, c3, c4, s1, s2, s3, s4 = inputs

        c2 = self.up_c2(c2)
        c3 = self.up_c3(c3)
        c4 = self.up_c4(c4)

        s2 = self.up_s2(s2)
        s3 = self.up_s3(s3)
        s4 = self.up_s4(s4)

        x = torch.cat([c1, c2, c3, c4, s1, s2, s3, s4], dim=1)  # (B, 96*8, H/4, W/4)
        features = [c1, c2, c3, c4, s1, s2, s3, s4]
        return x, features


class LayerNorm2d(nn.LayerNorm):
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class ScoreNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(9, 192, kernel_size=7, stride=2, padding=3)
        self.invert = nn.Sequential(
            LayerNorm2d(192),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(192, 768, kernel_size=1),
            nn.Conv2d(768, 192, kernel_size=1),
            nn.GELU(),
        )
        self.conv2 = nn.Conv2d(192, 8, kernel_size=7, stride=2, padding=3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        short = x
        x = self.invert(x)
        x = short + x
        x = self.conv2(x)
        return self.softmax(x.float())


# ============================================================
# ✅ Ablation: +CSWin (replace SegFormer) but keep Base design
# - 8 maps -> decoder directly
# - BCE only
# - no LearnableFusion / no injection / no Dice
# ============================================================
@MODELS.register_module(force=True)
class Mesorch_ConvNeXt_CSWinB(nn.Module):
    def __init__(
        self,
        image_size: int = 512,
        conv_pretrain: bool = False,
        cswin_weight_path: str = "/data/jdon492/Mesorch_with_pretrain_weight/CSWin-Transformer/cswin_base_224.pth",
    ):
        super().__init__()

        self.conv = ConvNeXt(conv_pretrain=conv_pretrain)
        self.cswin = CSWinBackbone(in_chans=6, img_size=image_size, weight_path=cswin_weight_path)

        self.decoder = UpsampleConcatConvCSWin()

        # 8 个专家特征都被 decoder 对齐成 96 通道，所以 inverse 统一 96->1
        self.inverse = nn.ModuleList([nn.Conv2d(96, 1, 1) for _ in range(8)])

        self.gate = ScoreNetwork()
        self.resize = nn.Upsample(size=(image_size, image_size), mode="bilinear", align_corners=True)

        self.loss_bce = nn.BCEWithLogitsLoss()
        self.loss_dice = DiceLoss()

        self.low_dct = LowDctFrequencyExtractor()
        self.high_dct = HighDctFrequencyExtractor()

    def forward(self, image, mask=None, *args, **kwargs):
        high = self.high_dct(image)
        low = self.low_dct(image)

        x_h = torch.cat([image, high], dim=1)          # 6ch
        x_l = torch.cat([image, low], dim=1)           # 6ch
        x_all = torch.cat([image, high, low], dim=1)   # 9ch (给 gate)

        _, feats_conv = self.conv.forward_features(x_h)   # [c1,c2,c3,c4]
        _, feats_cswin = self.cswin(x_l)                  # [s1,s2,s3,s4]

        # ✅ Base 思路：8 路直接拼到 decoder（不做 fusion）
        decoder_inputs = feats_conv + feats_cswin
        x, feats = self.decoder(decoder_inputs)

        gate_w = self.gate(x_all)  # (B,8,H/4,W/4)

        reduced = torch.cat([self.inverse[i](feats[i]) for i in range(8)], dim=1)  # (B,8,H/4,W/4)
        pred = torch.sum(gate_w * reduced, dim=1, keepdim=True)  # (B,1,H/4,W/4)
        pred = self.resize(pred)

        loss_bce = None
        loss_dice = None
        loss = None
        if mask is not None:
            loss_bce = self.loss_bce(pred, mask)
            loss_dice = self.loss_dice(pred, mask)
            loss = loss_bce + loss_dice

        return {
            "backward_loss": loss,
            "pred_mask": torch.sigmoid(pred),
            "pred_label": None,
            "visual_loss": (
                {"loss_total": loss, "loss_bce": loss_bce, "loss_dice": loss_dice}
                if loss is not None else {}
            ),
            "visual_image": {"pred_mask": torch.sigmoid(pred)},
        }

# import os
# import sys
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import timm
# from IMDLBenCo.registry import MODELS
#
# from extractor.high_frequency_feature_extraction import HighDctFrequencyExtractor
# from extractor.low_frequency_feature_extraction import LowDctFrequencyExtractor

# # ===== CSWin repo path (按你自己环境改) =====
# # 你原来是：/data/jdon492/Mesorch_with_pretrain_weight/CSWin-Transformer
# CSWIN_REPO = "/data/jdon492/Mesorch_with_pretrain_weight/CSWin-Transformer"
# if CSWIN_REPO not in sys.path:
#     sys.path.append(CSWIN_REPO)
#
# from models.cswin import CSWinTransformer
#
#
# # ---------------------------
# # ConvNeXt 分支（RGB + 高频）
# # ---------------------------
# class ConvNeXt(timm.models.convnext.ConvNeXt):
#     def __init__(self, conv_pretrain: bool = False):
#         super().__init__(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768))
#         if conv_pretrain:
#             model = timm.create_model("convnext_tiny", pretrained=True)
#             self.load_state_dict(model.state_dict(), strict=False)
#
#         # 适配 6 通道输入
#         orig = self.stem[0]
#         new = nn.Conv2d(
#             6,
#             orig.out_channels,
#             kernel_size=orig.kernel_size,
#             stride=orig.stride,
#             padding=orig.padding,
#             bias=False,
#         )
#         with torch.no_grad():
#             new.weight[:, :3] = orig.weight
#             nn.init.kaiming_normal_(new.weight[:, 3:])
#         self.stem[0] = new
#
#     def forward_features(self, x):
#         x = self.stem(x)
#         outs = []
#         for stage in self.stages:
#             x = stage(x)
#             outs.append(x)
#         x = self.norm_pre(x)
#         return x, outs
#
#
# # ---------------------------
# # CSWin 分支（RGB + 低频）
# # ---------------------------
# class CSWinBackbone(nn.Module):
#     """
#     CSWin-Base backbone, 输出 4 个 stage 的特征:
#     [96, 192, 384, 768]，空间尺度依次 /4 /8 /16 /32
#     """
#
#     def __init__(
#         self,
#         in_chans: int = 6,
#         img_size: int = 512,
#         weight_path: str = "/data/jdon492/Mesorch_with_pretrain_weight/CSWin-Transformer/cswin_base_224.pth",
#     ):
#         super().__init__()
#         print(f"✅ Using CSWin-Base (img={img_size}, in_chans={in_chans})")
#
#         self.backbone = CSWinTransformer(
#             img_size=img_size,
#             patch_size=4,
#             in_chans=in_chans,
#             embed_dim=96,
#             depth=[2, 4, 32, 2],
#             split_size=[1, 2, 8, 8],
#             num_heads=[4, 8, 16, 32],
#             mlp_ratio=4.0,
#             drop_rate=0.0,
#             attn_drop_rate=0.0,
#             drop_path_rate=0.1,
#             use_chk=False,
#         )
#
#         # 先把第一层 conv 改成 in_chans
#         patch_conv = self.backbone.stage1_conv_embed[0]
#         old_w = patch_conv.weight.clone()
#         new_conv = nn.Conv2d(
#             in_chans,
#             patch_conv.out_channels,
#             kernel_size=patch_conv.kernel_size,
#             stride=patch_conv.stride,
#             padding=patch_conv.padding,
#             bias=False,
#         )
#         with torch.no_grad():
#             if old_w.shape[1] == 3:
#                 new_conv.weight[:, :3] = old_w
#                 nn.init.kaiming_normal_(new_conv.weight[:, 3:])
#         self.backbone.stage1_conv_embed[0] = new_conv
#         print("✅ Modified first conv layer → 6 channels")
#
#         # 加载预训练（可选）
#         if weight_path and os.path.exists(weight_path):
#             print(f"📦 Loading pretrained weights from: {weight_path}")
#             state = torch.load(weight_path, map_location="cpu")
#
#             if "state_dict_ema" in state:
#                 state = state["state_dict_ema"]
#                 print("📘 Using EMA weights (state_dict_ema)")
#             elif "state_dict" in state:
#                 state = state["state_dict"]
#                 print("📗 Using state_dict weights")
#             elif "model" in state:
#                 state = state["model"]
#                 print("📙 Using model weights")
#
#             state = {k.replace("module.", ""): v for k, v in state.items()}
#
#             # 适配 6 通道 conv
#             conv_key = "stage1_conv_embed.0.weight"
#             if conv_key in state:
#                 old = state[conv_key]  # (96,3,7,7)
#                 new = torch.zeros((old.shape[0], 6, old.shape[2], old.shape[3]))
#                 new[:, :3] = old
#                 nn.init.kaiming_normal_(new[:, 3:])
#                 state[conv_key] = new
#
#             # pos_embed 插值（224 -> img_size）
#             if "pos_embed" in state:
#                 pos = state["pos_embed"]
#                 old_size = int(pos.shape[1] ** 0.5)
#                 new_size = img_size // 4
#                 if old_size != new_size:
#                     print(f"🔄 Interpolating pos_embed: {old_size} → {new_size}")
#                     pos = pos.permute(0, 2, 1).reshape(1, -1, old_size, old_size)
#                     pos = F.interpolate(pos, size=(new_size, new_size), mode="bicubic", align_corners=False)
#                     state["pos_embed"] = pos.flatten(2).permute(0, 2, 1)
#
#             msg = self.backbone.load_state_dict(state, strict=False)
#             print(f"✅ CSWin pretrained loaded! (missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)})")
#         else:
#             print("⚠️ CSWin pretrained weight not found — training from scratch!")
#
#         # 取出 stage 模块
#         self.stage1 = self.backbone.stage1
#         self.stage2 = self.backbone.stage2
#         self.stage3 = self.backbone.stage3
#         self.stage4 = self.backbone.stage4
#         self.merge1 = self.backbone.merge1
#         self.merge2 = self.backbone.merge2
#         self.merge3 = self.backbone.merge3
#
#     def forward(self, x):
#         """
#         返回：
#             x_final, outs=[x1(96,H/4), x2(192,H/8), x3(384,H/16), x4(768,H/32)]
#         """
#         B = x.shape[0]
#
#         x = self.backbone.stage1_conv_embed(x)  # (B, N, 96)
#         H = W = int(x.shape[1] ** 0.5)
#         x1 = x.transpose(-2, -1).contiguous().view(B, 96, H, W)
#
#         # stage1
#         for blk in self.stage1:
#             x = blk(x)
#         x = self.merge1(x)
#
#         # stage2
#         H = W = int(x.shape[1] ** 0.5)
#         x2 = x.transpose(-2, -1).contiguous().view(B, 192, H, W)
#         for blk in self.stage2:
#             x = blk(x)
#         x = self.merge2(x)
#
#         # stage3
#         H = W = int(x.shape[1] ** 0.5)
#         x3 = x.transpose(-2, -1).contiguous().view(B, 384, H, W)
#         for blk in self.stage3:
#             x = blk(x)
#         x = self.merge3(x)
#
#         # stage4
#         H = W = int(x.shape[1] ** 0.5)
#         x4 = x.transpose(-2, -1).contiguous().view(B, 768, H, W)
#         for blk in self.stage4:
#             x = blk(x)
#
#         return x, [x1, x2, x3, x4]
#
#
# # ---------------------------
# # Decoder：专门给 “ConvNeXt(96/192/384/768) + CSWin(96/192/384/768)”
# # 把 8 路特征都 upsample 到 H/4，并 cat 成 x
# # ---------------------------
# class UpsampleConcatConvCSWin(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # conv branch
#         self.up_c2 = nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1)
#         self.up_c3 = nn.Sequential(
#             nn.ConvTranspose2d(384, 192, kernel_size=4, stride=2, padding=1),
#             nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1),
#         )
#         self.up_c4 = nn.Sequential(
#             nn.ConvTranspose2d(768, 384, kernel_size=4, stride=2, padding=1),
#             nn.ConvTranspose2d(384, 192, kernel_size=4, stride=2, padding=1),
#             nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1),
#         )
#
#         # cswin branch（同样的通道结构）
#         self.up_s2 = nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1)
#         self.up_s3 = nn.Sequential(
#             nn.ConvTranspose2d(384, 192, kernel_size=4, stride=2, padding=1),
#             nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1),
#         )
#         self.up_s4 = nn.Sequential(
#             nn.ConvTranspose2d(768, 384, kernel_size=4, stride=2, padding=1),
#             nn.ConvTranspose2d(384, 192, kernel_size=4, stride=2, padding=1),
#             nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1),
#         )
#
#     def forward(self, inputs):
#         c1, c2, c3, c4, s1, s2, s3, s4 = inputs
#
#         c2 = self.up_c2(c2)
#         c3 = self.up_c3(c3)
#         c4 = self.up_c4(c4)
#
#         s2 = self.up_s2(s2)
#         s3 = self.up_s3(s3)
#         s4 = self.up_s4(s4)
#
#         x = torch.cat([c1, c2, c3, c4, s1, s2, s3, s4], dim=1)  # (B, 96*8, H/4, W/4)
#         features = [c1, c2, c3, c4, s1, s2, s3, s4]
#         return x, features
#
#
# class LayerNorm2d(nn.LayerNorm):
#     def __init__(self, num_channels, eps=1e-6, affine=True):
#         super().__init__(num_channels, eps=eps, elementwise_affine=affine)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = x.permute(0, 2, 3, 1)
#         x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
#         x = x.permute(0, 3, 1, 2)
#         return x
#
#
# class ScoreNetwork(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(9, 192, kernel_size=7, stride=2, padding=3)
#         self.invert = nn.Sequential(
#             LayerNorm2d(192),
#             nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
#             nn.Conv2d(192, 768, kernel_size=1),
#             nn.Conv2d(768, 192, kernel_size=1),
#             nn.GELU(),
#         )
#         self.conv2 = nn.Conv2d(192, 8, kernel_size=7, stride=2, padding=3)
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         short = x
#         x = self.invert(x)
#         x = short + x
#         x = self.conv2(x)
#         return self.softmax(x.float())
#
#
# # ============================================================
# # ✅ Ablation: +CSWin (replace SegFormer) but keep Base design
# # - 8 maps -> decoder directly
# # - BCE only
# # - no LearnableFusion / no injection / no Dice
# # ============================================================
# @MODELS.register_module(force=True)
# class Mesorch_ConvNeXt_CSWinB(nn.Module):
#     def __init__(
#         self,
#         image_size: int = 512,
#         conv_pretrain: bool = False,
#         cswin_weight_path: str = "/data/jdon492/Mesorch_with_pretrain_weight/CSWin-Transformer/cswin_base_224.pth",
#     ):
#         super().__init__()
#
#         self.conv = ConvNeXt(conv_pretrain=conv_pretrain)
#         self.cswin = CSWinBackbone(in_chans=6, img_size=image_size, weight_path=cswin_weight_path)
#
#         self.decoder = UpsampleConcatConvCSWin()
#
#         # 8 个专家特征都被 decoder 对齐成 96 通道，所以 inverse 统一 96->1
#         self.inverse = nn.ModuleList([nn.Conv2d(96, 1, 1) for _ in range(8)])
#
#         self.gate = ScoreNetwork()
#         self.resize = nn.Upsample(size=(image_size, image_size), mode="bilinear", align_corners=True)
#
#         self.loss_bce = nn.BCEWithLogitsLoss()
#
#         self.low_dct = LowDctFrequencyExtractor()
#         self.high_dct = HighDctFrequencyExtractor()
#
#     def forward(self, image, mask=None, *args, **kwargs):
#         high = self.high_dct(image)
#         low = self.low_dct(image)
#
#         x_h = torch.cat([image, high], dim=1)          # 6ch
#         x_l = torch.cat([image, low], dim=1)           # 6ch
#         x_all = torch.cat([image, high, low], dim=1)   # 9ch (给 gate)
#
#         _, feats_conv = self.conv.forward_features(x_h)   # [c1,c2,c3,c4]
#         _, feats_cswin = self.cswin(x_l)                  # [s1,s2,s3,s4]
#
#         # ✅ Base 思路：8 路直接拼到 decoder（不做 fusion）
#         decoder_inputs = feats_conv + feats_cswin
#         x, feats = self.decoder(decoder_inputs)
#
#         gate_w = self.gate(x_all)  # (B,8,H/4,W/4)
#
#         reduced = torch.cat([self.inverse[i](feats[i]) for i in range(8)], dim=1)  # (B,8,H/4,W/4)
#         pred = torch.sum(gate_w * reduced, dim=1, keepdim=True)  # (B,1,H/4,W/4)
#         pred = self.resize(pred)
#
#         loss = None
#         if mask is not None:
#             loss = self.loss_bce(pred, mask)
#
#         return {
#             "backward_loss": loss,
#             "pred_mask": torch.sigmoid(pred),
#             "pred_label": None,
#             "visual_loss": {"predict_loss": loss} if loss is not None else {},
#             "visual_image": {"pred_mask": torch.sigmoid(pred)},
#         }