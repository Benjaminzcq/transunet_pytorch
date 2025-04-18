#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
将DRIVE数据集中的TIF和GIF图像转换为PNG格式
"""

import os
import glob
import shutil
from PIL import Image
import numpy as np
from tqdm import tqdm

def ensure_dir(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def convert_images(src_dir, dst_dir, pattern, is_mask=False):
    """
    将指定目录下的图像转换为PNG格式
    
    参数:
        src_dir: 源目录
        dst_dir: 目标目录
        pattern: 文件匹配模式，如 '*.tif'
        is_mask: 是否为掩码图像
    """
    # 确保目标目录存在
    ensure_dir(dst_dir)
    
    # 获取所有匹配的文件
    files = glob.glob(os.path.join(src_dir, pattern))
    
    if not files:
        print(f"在 {src_dir} 中未找到匹配 {pattern} 的文件")
        return
    
    print(f"找到 {len(files)} 个文件，开始转换...")
    
    # 转换每个文件
    for file_path in tqdm(files):
        try:
            # 获取文件名（不含扩展名）
            filename = os.path.basename(file_path)
            name_without_ext = os.path.splitext(filename)[0]
            
            # 新文件路径
            new_file_path = os.path.join(dst_dir, f"{name_without_ext}.png")
            
            # 打开图像
            img = Image.open(file_path)
            
            # 如果是掩码，确保是二值图像
            if is_mask:
                # 转换为灰度图
                img = img.convert('L')
                # 二值化处理
                img_array = np.array(img)
                img_array = (img_array > 0).astype(np.uint8) * 255
                img = Image.fromarray(img_array)
            else:
                # 确保RGB模式
                img = img.convert('RGB')
            
            # 保存为PNG
            img.save(new_file_path, 'PNG')
            
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")

def copy_other_files(src_dir, dst_dir, exclude_patterns):
    """
    复制其他格式的文件（不在排除列表中的文件）
    
    参数:
        src_dir: 源目录
        dst_dir: 目标目录
        exclude_patterns: 要排除的文件模式列表
    """
    # 确保目标目录存在
    ensure_dir(dst_dir)
    
    # 获取所有文件
    all_files = glob.glob(os.path.join(src_dir, '*'))
    
    # 获取要排除的文件
    exclude_files = []
    for pattern in exclude_patterns:
        exclude_files.extend(glob.glob(os.path.join(src_dir, pattern)))
    
    # 过滤出要复制的文件
    files_to_copy = [f for f in all_files if f not in exclude_files and os.path.isfile(f)]
    
    if not files_to_copy:
        print(f"在 {src_dir} 中没有需要复制的文件")
        return
    
    print(f"找到 {len(files_to_copy)} 个其他文件，开始复制...")
    
    # 复制每个文件
    for file_path in tqdm(files_to_copy):
        try:
            # 获取文件名
            filename = os.path.basename(file_path)
            
            # 新文件路径
            new_file_path = os.path.join(dst_dir, filename)
            
            # 复制文件
            shutil.copy2(file_path, new_file_path)
            
        except Exception as e:
            print(f"复制文件 {file_path} 时出错: {str(e)}")

def main():
    # 源数据集路径
    src_root = './DRIVE'
    
    # 目标数据集路径
    dst_root = './DRIVE_PNG'
    
    # 确保目标根目录存在
    ensure_dir(dst_root)
    
    #################################################################
    # 转换训练集图像
    print("处理训练集图像...")
    src_train_images = os.path.join(src_root, 'training', 'images')
    dst_train_images = os.path.join(dst_root, 'training', 'images')
    convert_images(src_train_images, dst_train_images, '*.tif')
    copy_other_files(src_train_images, dst_train_images, ['*.tif'])
    
    # 转换训练集掩码
    print("处理训练集掩码...")
    src_train_masks = os.path.join(src_root, 'training', 'mask')
    dst_train_masks = os.path.join(dst_root, 'training', 'mask')
    convert_images(src_train_masks, dst_train_masks, '*.gif', is_mask=True)
    copy_other_files(src_train_masks, dst_train_masks, ['*.gif'])
        
    # 转换训练集手动掩码
    print("处理训练集手动掩码...")
    src_train_masks = os.path.join(src_root, 'training', '1st_manual')
    dst_train_masks = os.path.join(dst_root, 'training', '1st_manual')
    convert_images(src_train_masks, dst_train_masks, '*.gif', is_mask=True)
    copy_other_files(src_train_masks, dst_train_masks, ['*.gif'])

    #################################################################
    # 转换测试集图像
    print("处理测试集图像...")
    src_test_images = os.path.join(src_root, 'test', 'images')
    dst_test_images = os.path.join(dst_root, 'test', 'images')
    convert_images(src_test_images, dst_test_images, '*.tif')
    copy_other_files(src_test_images, dst_test_images, ['*.tif'])
    
    # 转换测试集掩码
    print("处理测试集掩码...")
    src_test_masks = os.path.join(src_root, 'test', 'mask')
    dst_test_masks = os.path.join(dst_root, 'test', 'mask')
    convert_images(src_test_masks, dst_test_masks, '*.gif', is_mask=True)
    copy_other_files(src_test_masks, dst_test_masks, ['*.gif'])
    
    # 转换测试集手动掩码1
    print("处理测试集手动掩码...")
    src_test_masks = os.path.join(src_root, 'test', '1st_manual')
    dst_test_masks = os.path.join(dst_root, 'test', '1st_manual')
    convert_images(src_test_masks, dst_test_masks, '*.gif', is_mask=True)
    copy_other_files(src_test_masks, dst_test_masks, ['*.gif'])
    
    # 转换测试集手动掩码2
    print("处理测试集手动掩码...")
    src_test_masks = os.path.join(src_root, 'test', '2nd_manual')
    dst_test_masks = os.path.join(dst_root, 'test', '2nd_manual')
    convert_images(src_test_masks, dst_test_masks, '*.gif', is_mask=True)
    copy_other_files(src_test_masks, dst_test_masks, ['*.gif'])

    print("转换完成！所有图像已保存到 DRIVE_PNG 目录")

if __name__ == '__main__':
    main()
