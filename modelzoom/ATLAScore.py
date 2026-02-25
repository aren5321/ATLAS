"""

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 1. base block
# ============================================================
class ChannelMixer(nn.Module):
    def __init__(self, ch, expand=2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(ch, ch * expand, 1, bias=False),
            nn.GroupNorm(1, ch * expand),
            nn.GELU(),
            nn.Conv2d(ch * expand, ch, 1, bias=False),
            nn.GroupNorm(1, ch),
            nn.GELU(),
        )

    def forward(self, x):
        return x + self.mlp(x)


class LiteGlobalContextBlock(nn.Module):
    """
    CNN-only global context block
    - large kernel DWConv
    - axial DWConv
    - gated fusion
    """
    def __init__(self, ch, k=21):
        super().__init__()

        self.local = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False),
            nn.GroupNorm(1, ch),
            nn.GELU()
        )

        self.large = nn.Sequential(
            nn.Conv2d(ch, ch, k, padding=k // 2, groups=ch, bias=False),
            nn.GroupNorm(1, ch),
            nn.GELU()
        )

        self.axial_h = nn.Conv2d(
            ch, ch, kernel_size=(1, k),
            padding=(0, k // 2), groups=ch, bias=False
        )
        self.axial_w = nn.Conv2d(
            ch, ch, kernel_size=(k, 1),
            padding=(k // 2, 0), groups=ch, bias=False
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(ch * 3, ch, 1, bias=False),
            nn.GroupNorm(1, ch),
            nn.GELU()
        )

        self.gate = nn.Sequential(
            nn.Conv2d(ch * 3, ch, 1, bias=False),
            nn.GroupNorm(1, ch),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_l = self.local(x)
        x_g = self.large(x)
        x_a = self.axial_h(x) + self.axial_w(x)

        x_cat = torch.cat([x_l, x_g, x_a], dim=1)
        return x + self.gate(x_cat) * self.fuse(x_cat)


class PartialMLPBlock(nn.Module):
    """
    FasterNet-style block:
    - Partial spatial mixing (only 1/n channels do 3x3 conv)
    - Conv-MLP channel mixing
    """
    def __init__(self, ch, n_div=4, mlp_ratio=2):
        super().__init__()
        self.ch = ch
        self.dim_conv = ch // n_div
        self.dim_pass = ch - self.dim_conv

        # Partial spatial mixing
        self.partial_conv = nn.Conv2d(
            self.dim_conv, self.dim_conv,
            kernel_size=3, padding=1,
            groups=self.dim_conv, bias=False
        )

        # Conv-MLP
        hidden = int(ch * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(ch, hidden, 1, bias=False),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, ch, 1, bias=False),
            nn.GroupNorm(1, ch),
        )

    def forward(self, x):
        # partial spatial conv
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_pass], dim=1)
        x1 = self.partial_conv(x1)
        x = torch.cat([x1, x2], dim=1)

        # channel mixing
        return x + self.mlp(x)


class DWConvBlock(nn.Module):
    """depth wise convolution block"""

    def __init__(self, ch, kernel):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, kernel, padding=kernel // 2, groups=ch, bias=False),
            nn.GroupNorm(1, ch),
            nn.GELU(),
            nn.Conv2d(ch, ch, 1, bias=False),
            nn.GroupNorm(1, ch),
            nn.GELU(),
        )

    def forward(self, x):
        return x + self.block(x)


class LiteAxialDW(nn.Module):
    """Axial wise convolution block"""

    def __init__(self, ch,k=7):
        super().__init__()
        assert k % 2 == 1, "Axial kernel size must be odd"

        pad=k//2
        self.norm = nn.GroupNorm(1, ch)

        self.dw_h = nn.Conv2d(ch, ch, (1, k), padding=(0, pad), groups=ch, bias=False)
        self.dw_w = nn.Conv2d(ch, ch, (k, 1), padding=(pad, 0), groups=ch, bias=False)
        self.pw = nn.Conv2d(ch, ch, 1, bias=False)

    def forward(self, x):
        y = self.norm(x)
        y = self.dw_h(y) + self.dw_w(y)
        y = self.pw(y)
        return x + y


class AvgDown(nn.Module):
    """Anti-aliasing downsampling"""

    def __init__(self, inp, oup):
        super().__init__()
        self.pool = nn.AvgPool2d(2)
        self.conv = nn.Sequential(
            nn.Conv2d(inp, oup, 1, bias=False),
            nn.GroupNorm(1, oup),
            nn.GELU()
        )

    def forward(self, x):
        return self.conv(self.pool(x))


# ============================================================
# 2.  Backbone
# ============================================================

class Backbone(nn.Module):
    """
    Backbone
    """
    def __init__(self, in_ch=3, dims=[8, 16, 24, 32, 48]):
        super().__init__()

        # =========================
        # Stem (C1, stride=2)
        # =========================
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, dims[0], 3, padding=1, bias=False),
            nn.GroupNorm(1, dims[0]),
            nn.GELU(),

            nn.Conv2d(dims[0], dims[0], 3, padding=1, groups=dims[0], bias=False),
            nn.GroupNorm(1, dims[0]),
            nn.GELU(),

            ChannelMixer(dims[0], expand=2),

            nn.Conv2d(dims[0], dims[0], 3, stride=2, padding=1, groups=dims[0], bias=False),
            nn.GroupNorm(1, dims[0]),
            nn.GELU(),
        )

        # =========================
        # Downsample blocks
        # =========================
        self.down1 = AvgDown(dims[0], dims[1])  # C1 -> C2
        self.down2 = AvgDown(dims[1], dims[2])  # C2 -> C3
        self.down3 = AvgDown(dims[2], dims[3])  # C3 -> C4
        self.down4 = AvgDown(dims[3], dims[4])  # C4 -> C5

        # =========================
        # Stage 1 (C2)
        # depth = 2
        # =========================
        self.stage1 = nn.Sequential(
            DWConvBlock(dims[1],kernel=3),
            DWConvBlock(dims[1], kernel=3)
        )

        # =========================
        # Stage 2 (C3)
        # depth = 3
        # =========================
        self.stage2 = nn.Sequential(
            DWConvBlock(dims[2], kernel=3),
            DWConvBlock(dims[2], kernel=3),
            LiteGlobalContextBlock(dims[2], k=13),
        )

        # =========================
        # Stage 3 (C4)
        # depth = 4 + Axial
        # =========================
        self.stage3 = nn.Sequential(
            PartialMLPBlock(dims[3]),
            PartialMLPBlock(dims[3]),
            LiteAxialDW(dims[3]),

            PartialMLPBlock(dims[3]),
            PartialMLPBlock(dims[3]),
            LiteAxialDW(dims[3]),

            LiteGlobalContextBlock(dims[3], k=21),
        )

        # =========================
        # Stage 4 (C5)
        # depth = 3 + Axial
        # =========================
        self.stage4 = nn.Sequential(
            PartialMLPBlock(dims[4]),
            LiteAxialDW(dims[4]),

            PartialMLPBlock(dims[4]),
            LiteAxialDW(dims[4])
        )

        self.out_channels = {
            "C1": dims[0],
            "C2": dims[1],
            "C3": dims[2],
            "C4": dims[3],
            "C5": dims[4],
        }

    def forward(self, x):
        feats = {}

        # C1
        x = self.stem(x)
        feats["C1"] = x

        # C2
        x = self.stage1(self.down1(x))
        feats["C2"] = x

        # C3
        x = self.stage2(self.down2(x))
        feats["C3"] = x

        # C4
        x = self.stage3(self.down3(x))
        feats["C4"] = x

        # C5
        x = self.stage4(self.down4(x))
        feats["C5"] = x

        return feats

# ============================================================
# 3. Neck
# ============================================================
class Neck(nn.Module):
    """
    Dual fused Neck: Top-down + Bottom-up
    - alpha control intensity of fusion
    - full scales outputs
    """
    def __init__(self, feat_channels, alpha=0.2):
        super().__init__()
        self.alpha = alpha
        c1, c2, c3, c4, c5 = feat_channels["C1"], feat_channels["C2"], feat_channels["C3"], feat_channels["C4"], feat_channels["C5"]

        # --------------------
        # projection layers
        # --------------------
        self.p5 = nn.Conv2d(c5, c4, 1, bias=False)
        self.p4 = nn.Conv2d(c4, c3, 1, bias=False)
        self.p3 = nn.Conv2d(c3, c2, 1, bias=False)
        self.p2 = nn.Conv2d(c2, c1, 1, bias=False)

        self.down1 = nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False)
        self.down2 = nn.Conv2d(c2, c3, 3, stride=2, padding=1, bias=False)
        self.down3 = nn.Conv2d(c3, c4, 3, stride=2, padding=1, bias=False)
        self.down4 = nn.Conv2d(c4, c5, 3, stride=2, padding=1, bias=False)

        # --------------------
        # gate layers
        # --------------------
        self.gate4 = nn.Sequential(nn.GroupNorm(1, c4*2), nn.Conv2d(c4*2, c4, 1), nn.Sigmoid())
        self.gate3 = nn.Sequential(nn.GroupNorm(1, c3*2), nn.Conv2d(c3*2, c3, 1), nn.Sigmoid())
        self.gate2 = nn.Sequential(nn.GroupNorm(1, c2*2), nn.Conv2d(c2*2, c2, 1), nn.Sigmoid())
        self.gate1 = nn.Sequential(nn.GroupNorm(1, c1*2), nn.Conv2d(c1*2, c1, 1), nn.Sigmoid())

        self.gate2b = nn.Sequential(nn.GroupNorm(1, c2*2), nn.Conv2d(c2*2, c2, 1), nn.Sigmoid())
        self.gate3b = nn.Sequential(nn.GroupNorm(1, c3*2), nn.Conv2d(c3*2, c3, 1), nn.Sigmoid())
        self.gate4b = nn.Sequential(nn.GroupNorm(1, c4*2), nn.Conv2d(c4*2, c4, 1), nn.Sigmoid())
        self.gate5b = nn.Sequential(nn.GroupNorm(1, c5*2), nn.Conv2d(c5*2, c5, 1), nn.Sigmoid())

        # --------------------
        # refine
        # --------------------
        self.refine1 = DWConvBlock(c1, 3)
        self.refine2 = DWConvBlock(c2, 3)
        self.refine3 = DWConvBlock(c3, 3)
        self.refine4 = DWConvBlock(c4, 3)
        self.refine5 = DWConvBlock(c5, 3)

    def forward(self, feats):
        C1, C2, C3, C4, C5 = feats["C1"], feats["C2"], feats["C3"], feats["C4"], feats["C5"]

        # -----------------------------
        # Top-down
        # -----------------------------
        U5 = F.interpolate(self.p5(C5), size=C4.shape[2:], mode='bilinear', align_corners=False)
        C4 = C4 + self.alpha * self.gate4(torch.cat([C4, U5], dim=1)) * U5

        U4 = F.interpolate(self.p4(C4), size=C3.shape[2:], mode='bilinear', align_corners=False)
        C3 = C3 + self.alpha * self.gate3(torch.cat([C3, U4], dim=1)) * U4

        U3 = F.interpolate(self.p3(C3), size=C2.shape[2:], mode='bilinear', align_corners=False)
        C2 = C2 + self.alpha * self.gate2(torch.cat([C2, U3], dim=1)) * U3

        U2 = F.interpolate(self.p2(C2), size=C1.shape[2:], mode='bilinear', align_corners=False)
        C1 = C1 + self.alpha * self.gate1(torch.cat([C1, U2], dim=1)) * U2

        # -----------------------------
        # Bottom-up
        # -----------------------------
        D1 = self.down1(C1)
        # Prevent pixel loss caused by odd spatial sizes
        if D1.shape[2:] != C2.shape[2:]:
            D1 = F.interpolate(D1, size=C2.shape[2:], mode='bilinear', align_corners=False)
        C2 = C2 + self.alpha * self.gate2b(torch.cat([C2, D1], dim=1)) * D1

        D2 = self.down2(C2)
        if D2.shape[2:] != C3.shape[2:]:
            D2 = F.interpolate(D2, size=C3.shape[2:], mode='bilinear', align_corners=False)
        C3 = C3 + self.alpha * self.gate3b(torch.cat([C3, D2], dim=1)) * D2

        D3 = self.down3(C3)
        if D3.shape[2:] != C4.shape[2:]:
            D3 = F.interpolate(D3, size=C4.shape[2:], mode='bilinear', align_corners=False)
        C4 = C4 + self.alpha * self.gate4b(torch.cat([C4, D3], dim=1)) * D3

        D4 = self.down4(C4)
        if D4.shape[2:] != C5.shape[2:]:
            D4 = F.interpolate(D4, size=C5.shape[2:], mode='bilinear', align_corners=False)
        C5 = C5 + self.alpha * self.gate5b(torch.cat([C5, D4], dim=1)) * D4

        # -----------------------------
        # refine
        # -----------------------------
        C1 = self.refine1(C1)
        C2 = self.refine2(C2)
        C3 = self.refine3(C3)
        C4 = self.refine4(C4)
        C5 = self.refine5(C5)

        return {"C1": C1, "C2": C2, "C3": C3, "C4": C4, "C5": C5}
# ============================================================
# 4. Heads
# ============================================================

class ClsHead(nn.Module):
    """ global enhanced cls head"""
    def __init__(self, feat_channels, out_ch):
        super().__init__()
        last_key = list(feat_channels.keys())[-1]
        ch = feat_channels[last_key]
        self.mix = ChannelMixer(ch, expand=1)
        self.pool_avg = nn.AdaptiveAvgPool2d(1)
        self.pool_max = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(ch*2, out_ch)

    def forward(self, feats):
        x = feats[list(feats.keys())[-1]]
        x = self.mix(x)
        avg = self.pool_avg(x).flatten(1)
        maxp = self.pool_max(x).flatten(1)
        x = torch.cat([avg, maxp], dim=1)
        return self.fc(x)


class SegHead(nn.Module):
    def __init__(self, feat_channels, out_ch):
        super().__init__()
        c1 = feat_channels["C1"]
        self.refine = nn.Sequential(
            DWConvBlock(c1, kernel=3),
            LiteAxialDW(c1)
        )
        self.out = nn.Conv2d(c1, out_ch, 1)

    def forward(self, feats):
        x = feats["C1"]
        x = self.refine(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.out(x)

# =========================================================
# RegHead
# =========================================================

class RegHead(nn.Module):
    def __init__(self, feat_channels, out_ch):
        super().__init__()
        c5_ch = feat_channels["C5"]

        self.sap = nn.Sequential(
            nn.Conv2d(c5_ch , 1, 1),
            nn.Sigmoid()
        )
        self.fc = nn.Linear(c5_ch , out_ch)

    def forward(self, feats):
        x = feats["C5"]
        mask = self.sap(x)
        # average pooling
        feat = (x * mask).sum(dim=(2, 3)) / (mask.sum(dim=(2, 3)) + 1e-6)
        return self.fc(feat)


# =========================================================
# Det Heads
# =========================================================

class DetHeadBlock(nn.Module):
    """
    Single-scale detection head (FCOS-lite style)
    Separate heads for cls, ctr, reg
    """
    def __init__(self, in_ch, num_cls=1):
        super().__init__()
        self.num_cls = num_cls

        # Shared conv block before splitting
        self.shared = nn.Sequential(
            DWConvBlock(in_ch, 3),
            DWConvBlock(in_ch, 3),
            ChannelMixer(in_ch, expand=2)
        )

        # Separate heads
        self.cls_head = nn.Conv2d(in_ch, num_cls + 1, 1)  # 1 bg + num_cls
        self.ctr_head = nn.Conv2d(in_ch, 1, 1)
        self.reg_head = nn.Conv2d(in_ch, 4, 1)

    def forward(self, x):
        x = self.shared(x)
        cls_p = self.cls_head(x)
        ctr_p = self.ctr_head(x)
        reg_p = self.reg_head(x)
        return cls_p, ctr_p, reg_p


class DetHead(nn.Module):
    """
    Multi-scale Detection Head (P3 / P4 / P5)
    FCOS-lite style
    """
    def __init__(self, feat_channels, num_cls=1):
        super().__init__()
        c3_ch = feat_channels["C3"]
        c4_ch = feat_channels["C4"]
        c5_ch = feat_channels["C5"]

        # separate heads per scale
        self.head_p3 = DetHeadBlock(c3_ch, num_cls)
        self.head_p4 = DetHeadBlock(c4_ch, num_cls)
        self.head_p5 = DetHeadBlock(c5_ch, num_cls)

    def forward(self, feats):
        C3, C4, C5 = feats["C3"], feats["C4"], feats["C5"]
        B = C3.shape[0]

        # forward per scale
        cls_p3, ctr_p3, reg_p3 = self.head_p3(C3)
        cls_p4, ctr_p4, reg_p4 = self.head_p4(C4)
        cls_p5, ctr_p5, reg_p5 = self.head_p5(C5)

        # flatten spatial dims
        def flatten_head(t):
            return t.view(B, t.shape[1], -1).permute(0, 2, 1)  # [B, H*W, C]

        cls_p = torch.cat([flatten_head(cls_p3),
                           flatten_head(cls_p4),
                           flatten_head(cls_p5)], dim=1)

        ctr_p = torch.cat([flatten_head(ctr_p3),
                           flatten_head(ctr_p4),
                           flatten_head(ctr_p5)], dim=1)

        reg_p = torch.cat([flatten_head(reg_p3),
                           flatten_head(reg_p4),
                           flatten_head(reg_p5)], dim=1)

        # concat along channel dim to match previous output format
        total_prediction = torch.cat([cls_p, ctr_p, reg_p], dim=-1)  # [B, total_pixels, total_channels]

        return total_prediction



class ATLAScore(nn.Module):
    def __init__(self, in_ch=3, num_classes=1, dims=[8, 16, 24, 32, 48],task_id=1):
        """
        Args:
        in_ch (int): Number of input channels, default: 3
        num_classes (int): Number of classes for classification, default: 1
        dims (list): List of feature dimensions, default: [8, 16, 24, 32, 48]
        task_id (int): Task ID for selecting different task heads, default: 1
        0: Classification task
        1: Segmentation task
        2: Regression task
        3: Detection task
        """
        super().__init__()
        self.task_id = task_id

        self.encoder = Backbone(in_ch=in_ch, dims=dims)

        self.neck = Neck(self.encoder.out_channels)

        if task_id==0:
            self.head = ClsHead(self.encoder.out_channels, num_classes)
        elif task_id==1:
            self.head = SegHead(self.encoder.out_channels, num_classes)
        elif task_id==2:
            self.head = RegHead(self.encoder.out_channels, num_classes)
        elif task_id==3:
            self.head = DetHead(self.encoder.out_channels, num_classes)
        else:
            raise ValueError("task_id must be in [0,1,2,3]")

    def forward(self,x):

        feats = self.encoder(x)

        feats = self.neck(feats)

        return self.head(feats)

# -------------------------------------------------------------------------
# Parameter count check
# -------------------------------------------------------------------------
if __name__ == "__main__":
    task_names = {0: "cls", 1: "seg", 2: "reg", 3: "det"}
    x = torch.randn(1, 3, 256, 256)
    for tid in range(4):
        model = ATLAScore(in_ch=3, num_classes=8, task_id=tid)
        # print(model)
        params = sum(p.numel() for p in model.parameters())
        print(f"ATLAS-core [{task_names[tid]}] parameters: {params / 1000:.2f} K")