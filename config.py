from easydict import EasyDict
import os

# 通用配置参数
cfg = EasyDict()
cfg.batch_size = 4  # 减小批次大小以减少显存使用
cfg.epoch = 200
cfg.learning_rate = 1e-2
cfg.momentum = 0.9
cfg.weight_decay = 1e-4
cfg.patience = 25
cfg.inference_threshold = 0.75

# 模型保存路径
cfg.model_dir = './saved_models'  # 模型保存的根目录
# 创建模型保存目录
if not os.path.exists(cfg.model_dir):
    os.makedirs(cfg.model_dir)

# TransUNet模型配置
cfg.transunet = EasyDict()
cfg.transunet.img_dim = 512
cfg.transunet.in_channels = 3
cfg.transunet.out_channels = 128
cfg.transunet.head_num = 4
cfg.transunet.mlp_dim = 512
cfg.transunet.block_num = 8
cfg.transunet.patch_dim = 16
cfg.transunet.class_num = 1
cfg.transunet.model_path = os.path.join(cfg.model_dir, 'transunet_model.pth')  # 模型保存路径

# Swin TransUNet模型配置
cfg.swin_transunet = EasyDict()
cfg.swin_transunet.img_dim = 512
cfg.swin_transunet.in_channels = 3
cfg.swin_transunet.out_channels = 64  # 减小输出通道数以减少显存消耗
cfg.swin_transunet.depths = [2, 2, 4, 2]  # Swin Transformer的深度
cfg.swin_transunet.num_heads = [2, 4, 8, 16]  # 注意力头数
cfg.swin_transunet.window_size = 7  # 窗口大小
cfg.swin_transunet.class_num = 1
cfg.swin_transunet.model_path = os.path.join(cfg.model_dir, 'swin_transunet_model.pth')  # 模型保存路径
