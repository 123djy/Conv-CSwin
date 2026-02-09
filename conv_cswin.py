import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from IMDLBenCo.registry import MODELS

from extractor.high_frequency_feature_extraction import HighDctFrequencyExtractor
from extractor.low_frequency_feature_extraction import LowDctFrequencyExtractor

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


class ConvNeXt(timm.models.convnext.ConvNeXt):
    def __init__(self, conv_pretrain: bool = False):
        super().__init__(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768))
        if conv_pretrain:
            model = timm.create_model("convnext_tiny", pretrained=True)
            self.load_state_dict(model.state_dict(), strict=False)

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


class CSWinBackbone(nn.Module):
    def __init__(
        self,
        in_chans: int = 6,
        img_size: int = 512,
        weight_path: str = "/data/jdon492/Mesorch_with_pretrain_weight/CSWin-Transformer/cswin_base_224.pth",
    ):
        super().__init__()

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

        if weight_path and os.path.exists(weight_path):
            state = torch.load(weight_path, map_location="cpu")

            if "state_dict_ema" in state:
                state = state["state_dict_ema"]
            elif "state_dict" in state:
                state = state["state_dict"]
            elif "model" in state:
                state = state["model"]

            state = {k.replace("module.", ""): v for k, v in state.items()}

            conv_key = "stage1_conv_embed.0.weight"
            if conv_key in state:
                old = state[conv_key]
                new = torch.zeros((old.shape[0], 6, old.shape[2], old.shape[3]))
                new[:, :3] = old
                nn.init.kaiming_normal_(new[:, 3:])
                state[conv_key] = new

            if "pos_embed" in state:
                pos = state["pos_embed"]
                old_size = int(pos.shape[1] ** 0.5)
                new_size = img_size // 4
                if old_size != new_size:
                    pos = pos.permute(0, 2, 1).reshape(1, -1, old_size, old_size)
                    pos = F.interpolate(
                        pos,
                        size=(new_size, new_size),
                        mode="bicubic",
                        align_corners=False,
                    )
                    state["pos_embed"] = pos.flatten(2).permute(0, 2, 1)

            self.backbone.load_state_dict(state, strict=False)

        self.stage1 = self.backbone.stage1
        self.stage2 = self.backbone.stage2
        self.stage3 = self.backbone.stage3
        self.stage4 = self.backbone.stage4
        self.merge1 = self.backbone.merge1
        self.merge2 = self.backbone.merge2
        self.merge3 = self.backbone.merge3

    def forward(self, x):
        B = x.shape[0]

        x = self.backbone.stage1_conv_embed(x)
        H = W = int(x.shape[1] ** 0.5)
        x1 = x.transpose(-2, -1).contiguous().view(B, 96, H, W)

        for blk in self.stage1:
            x = blk(x)
        x = self.merge1(x)

        H = W = int(x.shape[1] ** 0.5)
        x2 = x.transpose(-2, -1).contiguous().view(B, 192, H, W)
        for blk in self.stage2:
            x = blk(x)
        x = self.merge2(x)

        H = W = int(x.shape[1] ** 0.5)
        x3 = x.transpose(-2, -1).contiguous().view(B, 384, H, W)
        for blk in self.stage3:
            x = blk(x)
        x = self.merge3(x)

        H = W = int(x.shape[1] ** 0.5)
        x4 = x.transpose(-2, -1).contiguous().view(B, 768, H, W)
        for blk in self.stage4:
            x = blk(x)

        return x, [x1, x2, x3, x4]


class UpsampleConcatConvCSWin(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_c2 = nn.ConvTranspose2d(192, 96, 4, 2, 1)
        self.up_c3 = nn.Sequential(
            nn.ConvTranspose2d(384, 192, 4, 2, 1),
            nn.ConvTranspose2d(192, 96, 4, 2, 1),
        )
        self.up_c4 = nn.Sequential(
            nn.ConvTranspose2d(768, 384, 4, 2, 1),
            nn.ConvTranspose2d(384, 192, 4, 2, 1),
            nn.ConvTranspose2d(192, 96, 4, 2, 1),
        )

        self.up_s2 = nn.ConvTranspose2d(192, 96, 4, 2, 1)
        self.up_s3 = nn.Sequential(
            nn.ConvTranspose2d(384, 192, 4, 2, 1),
            nn.ConvTranspose2d(192, 96, 4, 2, 1),
        )
        self.up_s4 = nn.Sequential(
            nn.ConvTranspose2d(768, 384, 4, 2, 1),
            nn.ConvTranspose2d(384, 192, 4, 2, 1),
            nn.ConvTranspose2d(192, 96, 4, 2, 1),
        )

    def forward(self, inputs):
        c1, c2, c3, c4, s1, s2, s3, s4 = inputs

        c2 = self.up_c2(c2)
        c3 = self.up_c3(c3)
        c4 = self.up_c4(c4)

        s2 = self.up_s2(s2)
        s3 = self.up_s3(s3)
        s4 = self.up_s4(s4)

        x = torch.cat([c1, c2, c3, c4, s1, s2, s3, s4], dim=1)
        features = [c1, c2, c3, c4, s1, s2, s3, s4]
        return x, features


class LayerNorm2d(nn.LayerNorm):
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class ScoreNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(9, 192, 7, 2, 3)
        self.invert = nn.Sequential(
            LayerNorm2d(192),
            nn.Conv2d(192, 192, 3, 1, 1),
            nn.Conv2d(192, 768, 1),
            nn.Conv2d(768, 192, 1),
            nn.GELU(),
        )
        self.conv2 = nn.Conv2d(192, 8, 7, 2, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        short = x
        x = self.invert(x)
        x = short + x
        x = self.conv2(x)
        return self.softmax(x.float())


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
        self.cswin = CSWinBackbone(6, image_size, cswin_weight_path)

        self.decoder = UpsampleConcatConvCSWin()
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

        x_h = torch.cat([image, high], dim=1)
        x_l = torch.cat([image, low], dim=1)
        x_all = torch.cat([image, high, low], dim=1)

        _, feats_conv = self.conv.forward_features(x_h)
        _, feats_cswin = self.cswin(x_l)

        x, feats = self.decoder(feats_conv + feats_cswin)
        gate_w = self.gate(x_all)

        reduced = torch.cat([self.inverse[i](feats[i]) for i in range(8)], dim=1)
        pred = torch.sum(gate_w * reduced, dim=1, keepdim=True)
        pred = self.resize(pred)

        loss = None
        if mask is not None:
            loss = self.loss_bce(pred, mask) + self.loss_dice(pred, mask)

        return {
            "backward_loss": loss,
            "pred_mask": torch.sigmoid(pred),
            "pred_label": None,
            "visual_loss": {"loss_total": loss} if loss is not None else {},
            "visual_image": {"pred_mask": torch.sigmoid(pred)},
        }
