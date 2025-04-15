import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """将特征图分割成不重叠的窗口，支持特征图尺寸不被窗口大小整除的情况
    Args:
        x: (B, H, W, C)
        window_size (int): 窗口大小
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    
    # 计算填充值，确保高度和宽度能被窗口大小整除
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    
    # 在高度和宽度维度上进行填充
    if pad_h > 0 or pad_w > 0:
        x = nn.functional.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        # 更新高度和宽度
        _, H, W, _ = x.shape
    
    # 现在H和W能被window_size整除，可以安全地进行视图操作
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W, orig_H=None, orig_W=None):
    """将窗口重新组合成特征图，支持去除padding
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): 窗口大小
        H (int): 填充后的特征图高度
        W (int): 填充后的特征图宽度
        orig_H (int, optional): 原始特征图高度
        orig_W (int, optional): 原始特征图宽度
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    
    # 如果提供了原始尺寸信息，去除填充部分
    if orig_H is not None and orig_W is not None and (orig_H < H or orig_W < W):
        x = x[:, :orig_H, :orig_W, :]
    return x


class WindowAttention(nn.Module):
    """窗口内的多头自注意力机制，支持移位窗口
    Args:
        dim (int): 输入特征维度
        window_size (tuple[int]): 窗口大小
        num_heads (int): 注意力头数
        qkv_bias (bool, optional): 是否在QKV投影中使用偏置. Default: True
        qk_scale (float | None, optional): 缩放因子. Default: None
        attn_drop (float, optional): 注意力权重的dropout率. Default: 0.0
        proj_drop (float, optional): 输出的dropout率. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # 相对位置偏置参数表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # 获取窗口内每对位置的相对位置索引
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # 从0开始
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """前向传播函数
        Args:
            x: 输入特征 shape为 (B*nW, N, C)
            mask: (0/-inf) mask, shape为 (nW, N, N) 或 None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C/nH

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block.
    Args:
        dim (int): 输入特征维度
        num_heads (int): 注意力头数
        window_size (int): 窗口大小
        shift_size (int): 移位窗口大小
        mlp_ratio (float): MLP隐藏层维度与输入维度的比率
        qkv_bias (bool, optional): 是否在QKV投影中使用偏置. Default: True
        qk_scale (float | None, optional): 缩放因子. Default: None
        drop (float, optional): Dropout率. Default: 0.0
        attn_drop (float, optional): 注意力权重的Dropout率. Default: 0.0
        drop_path (float, optional): DropPath率. Default: 0.0
        act_layer (nn.Module, optional): 激活层. Default: nn.GELU
        norm_layer (nn.Module, optional): 归一化层. Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask):
        H, W = self.H, self.W
        B, L, C = x.shape
        
        # 如果输入特征形状与预期不符，将其调整为匹配的尺寸
        if L != H * W:
            print(f"调整Swin Transformer块的输入形状: {x.shape} -> ({B}, {H*W}, {C})")
            # 将输入强制调整为正确的尺寸
            x = x.view(B, -1, C)  # 将形状展平
            # 确保长度匹配
            if x.shape[1] > H * W:
                x = x[:, :H*W, :]  # 截取
            elif x.shape[1] < H * W:
                padding = torch.zeros(B, H*W - x.shape[1], C, device=x.device)
                x = torch.cat([x, padding], dim=1)  # 填充
            L = H * W  # 更新L值

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # 移位窗口
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # 保存原始尺寸
        orig_H, orig_W = H, W
        
        # 分割窗口
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # 合并窗口
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        # 获取当前的填充后尺寸
        curr_H = (H + self.window_size - 1) // self.window_size * self.window_size
        curr_W = (W + self.window_size - 1) // self.window_size * self.window_size
        shifted_x = window_reverse(attn_windows, self.window_size, curr_H, curr_W, orig_H, orig_W)  # B H' W' C

        # 反向移位
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.contiguous().view(B, H * W, C)  # 确保张量内存连续

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    """Patch Merging Layer
    Args:
        dim (int): 输入特征维度
        norm_layer (nn.Module, optional): 归一化层. Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """前向传播函数
        Args:
            x: 输入特征, shape为 (B, H*W, C)
            H, W: 空间分辨率
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # 填充
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Module):
    """Swin Transformer的基本层
    Args:
        dim (int): 输入特征维度
        depth (int): 块的数量
        num_heads (int): 注意力头数
        window_size (int): 窗口大小
        mlp_ratio (float): MLP隐藏层维度与输入维度的比率
        qkv_bias (bool, optional): 是否在QKV投影中使用偏置. Default: True
        qk_scale (float | None, optional): 缩放因子. Default: None
        drop (float, optional): Dropout率. Default: 0.0
        attn_drop (float, optional): 注意力权重的Dropout率. Default: 0.0
        drop_path (float | tuple[float], optional): DropPath率. Default: 0.0
        norm_layer (nn.Module, optional): 归一化层. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): 下采样层. Default: None
        use_checkpoint (bool): 是否使用checkpointing来节省显存. Default: False
    """

    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.shift_size = window_size // 2

        # 构建块
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # 下采样层
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def create_mask(self, x, H, W):
        # 计算注意力掩码
        # 确保Hp和Wp是window_size的倍数
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        # 图像特征的填充
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # nW, window_size*window_size
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # nW, window_size*window_size, window_size*window_size
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, H, W):
        """前向传播函数
        Args:
            x: 输入特征, shape为 (B, H*W, C)
            H, W: 空间分辨率
        """
        # 计算注意力掩码
        attn_mask = self.create_mask(x, H, W)  # nW, window_size*window_size, window_size*window_size

        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)

        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class SwinTransformer(nn.Module):
    """Swin Transformer
    Args:
        img_size (int | tuple(int)): 输入图像大小
        patch_size (int | tuple(int)): Patch大小
        in_chans (int): 输入通道数
        embed_dim (int): 嵌入维度
        depths (tuple(int)): 每个阶段的块数量
        num_heads (tuple(int)): 每个阶段的注意力头数
        window_size (int): 窗口大小
        mlp_ratio (float): MLP隐藏层维度与输入维度的比率
        qkv_bias (bool): 是否在QKV投影中使用偏置
        qk_scale (float): 缩放因子
        drop_rate (float): Dropout率
        attn_drop_rate (float): 注意力权重的Dropout率
        drop_path_rate (float): DropPath率
        norm_layer (nn.Module): 归一化层
        ape (bool): 是否使用绝对位置嵌入
        patch_norm (bool): 是否在嵌入层后使用归一化
        use_checkpoint (bool): 是否使用checkpointing来节省显存
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96,
                 depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # 分割图像成非重叠的patch
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # 绝对位置嵌入
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # 构建层
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        x_out = []
        for layer in self.layers:
            x, H, W, x_down, Wh, Ww = layer(x, *self.patches_resolution)
            x_out.append(x.view(-1, H, W, x.shape[-1]).permute(0, 3, 1, 2).contiguous())
            self.patches_resolution = (Wh, Ww)
            x = x_down

        x = self.norm(x)  # B L C
        x = x.view(-1, *self.patches_resolution, x.shape[-1]).permute(0, 3, 1, 2).contiguous()
        x_out.append(x)

        return x_out

    def forward(self, x):
        x = self.forward_features(x)
        return x


class PatchEmbed(nn.Module):
    """将图像分割成patch并进行线性嵌入
    Args:
        img_size (int): 输入图像大小
        patch_size (int): Patch大小
        in_chans (int): 输入通道数
        embed_dim (int): 嵌入维度
        norm_layer (nn.Module, optional): 归一化层
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # 确保输入图像大小是正确的
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x
