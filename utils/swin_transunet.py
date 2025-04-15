import torch
import torch.nn as nn
from einops import rearrange

from utils.swin_transformer import SwinTransformer


class EncoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, base_width=64):
        super().__init__()

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        width = int(out_channels * (base_width / 64))

        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, stride=1, bias=False)
        self.norm1 = nn.BatchNorm2d(width)

        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=2, groups=1, padding=1, dilation=1, bias=False)
        self.norm2 = nn.BatchNorm2d(width)

        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1, stride=1, bias=False)
        self.norm3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_down = self.downsample(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = x + x_down
        x = self.relu(x)

        return x


class DecoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, skip_channels=0):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        
        # 如果有skip connection，则计算拼接后的通道数
        total_channels = in_channels + skip_channels if skip_channels > 0 else in_channels
        
        self.layer = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, x_concat=None):
        x = self.upsample(x)

        if x_concat is not None:
            # 检查空间尺寸是否匹配
            if x.shape[2:] != x_concat.shape[2:]:
                # 如果空间尺寸不匹配，将x调整为x_concat的尺寸
                x = nn.functional.interpolate(
                    x, 
                    size=x_concat.shape[2:], 
                    mode='bilinear', 
                    align_corners=True
                )
                print(f'调整特征图尺寸：从{x.shape}到{x_concat.shape}')
            
            x = torch.cat([x_concat, x], dim=1)

        x = self.layer(x)
        return x


class Encoder(nn.Module):
    def __init__(self, img_dim, in_channels, out_channels, depths, num_heads, window_size):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.encoder1 = EncoderBottleneck(out_channels, out_channels * 2, stride=2)
        self.encoder2 = EncoderBottleneck(out_channels * 2, out_channels * 4, stride=2)
        self.encoder3 = EncoderBottleneck(out_channels * 4, out_channels * 8, stride=2)

        # 使用Swin Transformer替代Vision Transformer
        self.swin = SwinTransformer(
            img_size=img_dim // 16,  # 因为经过了4次下采样 (2^4 = 16)
            patch_size=1,           # 已经通过CNN进行了patch embedding
            in_chans=out_channels * 8,
            embed_dim=out_channels * 8,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.1,
            attn_drop_rate=0.1,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
            use_checkpoint=False
        )

        # 通道适配器，用于处理Swin Transformer输出的特征图与后续卷积层的通道数不匹配问题
        # 根据错误信息，现在Swin Transformer输出的是4096个通道（由于我们减小了模型复杂度）
        self.input_channels = 4096  # 更新后的Swin Transformer输出通道数
        self.num_features = out_channels * 8  # 期望的通道数 (512，因为out_channels现在是64)
        self.channel_adapter = nn.Conv2d(self.input_channels, self.num_features, kernel_size=1, bias=False)
        
        self.conv2 = nn.Conv2d(out_channels * 8, 512, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(512)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x1 = self.relu(x)

        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x = self.encoder3(x3)

        # Swin Transformer处理
        features = self.swin(x)
        x = features[-1]  # 获取最后一层特征
        
        # 使用事先定义的通道适配器调整特征通道数
        # 注意：如果这是第一次运行，特征图尺寸与通道数可能不符合预期
        x = self.channel_adapter(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)

        return x, x1, x2, x3


class Decoder(nn.Module):
    def __init__(self, out_channels, class_num):
        super().__init__()

        # 根据错误信息和通道数分析，指定正确的skip connection通道数
        # encoder3输出通道数为out_channels * 4 = 256
        self.decoder1 = DecoderBottleneck(out_channels * 8, out_channels * 2, skip_channels=out_channels * 4)
        # encoder2输出通道数为out_channels * 2 = 128
        self.decoder2 = DecoderBottleneck(out_channels * 2, out_channels, skip_channels=out_channels * 2)
        # encoder1输出通道数为out_channels = 64
        self.decoder3 = DecoderBottleneck(out_channels, int(out_channels * 1 / 2), skip_channels=out_channels)
        # 最后一层没有skip connection
        self.decoder4 = DecoderBottleneck(int(out_channels * 1 / 2), int(out_channels * 1 / 8))

        self.conv1 = nn.Conv2d(int(out_channels * 1 / 8), class_num, kernel_size=1)

    def forward(self, x, x1, x2, x3):
        x = self.decoder1(x, x3)
        x = self.decoder2(x, x2)
        x = self.decoder3(x, x1)
        x = self.decoder4(x)
        x = self.conv1(x)

        return x


class SwinTransUNet(nn.Module):
    def __init__(self, img_dim, in_channels, out_channels, depths, num_heads, window_size, class_num):
        super().__init__()

        self.encoder = Encoder(img_dim, in_channels, out_channels, depths, num_heads, window_size)
        self.decoder = Decoder(out_channels, class_num)

    def forward(self, x):
        x, x1, x2, x3 = self.encoder(x)
        x = self.decoder(x, x1, x2, x3)

        return x


if __name__ == '__main__':
    import torch

    swin_transunet = SwinTransUNet(
        img_dim=128,
        in_channels=3,
        out_channels=128,
        depths=[2, 2, 2, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        class_num=1
    )

    print(sum(p.numel() for p in swin_transunet.parameters()))
    print(swin_transunet(torch.randn(1, 3, 128, 128)).shape)
