# coding:utf-8
# filename:data.py
# function:鲸鱼识别程序,数据预处理


from __future__ import absolute_import
# Read the dataset description
import gzip, tkinter
# Read or generate p2h, a dictionary of image name to image id (picture to hash)
import pickle
import platform
import random
# Suppress annoying stderr output when importing keras.
import sys
import os

from math import sqrt
# Determine the size of each image
from os.path import isfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image as pil_image
from imagehash import phash
from pandas import read_csv
from scipy.ndimage import affine_transform
from tqdm import tqdm


# print (os.path.abspath('.'))#获得当前工作目录
# 源路径获取, 改成自己的工程文件所在根目录
src_dir = "/home/harley/Program/Kaggle_Competiton/Humpback-Whale-Identification/"
# src_dir = os.getcwd()
# src_dir = src_dir.replace('\\', '/');
print(src_dir)

TRAIN_DF = os.path.join(src_dir, "data/train.csv")
SUB_DF = os.path.join(src_dir, "data/sample_submission.csv")
TRAIN = os.path.join(src_dir, "data/train")
TEST = os.path.join(src_dir, "data/test")
P2H = os.path.join(src_dir, "data/p2h.pickle")
P2SIZE = os.path.join(src_dir, "data/p2size.pickle")
BB_DF = os.path.join(src_dir, "data/bounding_boxes.csv")

# 把Windows下os.path.join()生成的反斜杠（\）全部替换为斜杠（/）
TRAIN_DF = TRAIN_DF.replace('\\', '/')
SUB_DF = SUB_DF.replace('\\', '/')
TRAIN = TRAIN.replace('\\', '/')
TEST = TEST.replace('\\', '/')
P2H = P2H.replace('\\', '/')
P2SIZE = P2SIZE.replace('\\', '/')
BB_DF = BB_DF.replace('\\', '/')

print(TRAIN, '\n', TEST, '\n', TRAIN_DF, '\n', SUB_DF, '\n', P2H, '\n', BB_DF, '\n', type(P2SIZE))  # 路径打印测试

# 将训练集标签文件DataFrame结构转换为字典结构方便操作
tagged = dict([(p, w) for _, p, w in read_csv(TRAIN_DF).to_records()])

print('The number of train data image files is', len(tagged.keys()))
# 将测试集标签文件DataFrame的'Image'列转换为列表方便后续处理
submit = [p for _, p, _ in read_csv(SUB_DF).to_records()]
print('The number of test data image files is', len(submit))
# 训练集和测试集文件名'Image'相加,生成一个新列表
join = list(tagged.keys()) + submit
print('The number of train and test data image files is', len(join))

# Read the bounding box data from the bounding box kernel (see reference above)
p2bb = pd.read_csv(BB_DF).set_index("Image")

old_stderr = sys.stderr
sys.stderr = open('/dev/null' if platform.system() != 'Windows' else 'nul', 'w')
sys.stderr = old_stderr


# ---------------------------------------重复的图像识别------------------------------------------
# 根据图像Id,获取文件完整路径
def expand_path(p):
    if isfile(os.path.join(TRAIN, p)):
        return os.path.join(TRAIN, p)
    if isfile(os.path.join(TEST, p)):
        return os.path.join(TEST, p)
    return p


# file_path = expand_path('0000e88ab.jpg');print(file_path)

# 图像哈希值匹配函数
# 对所有图像对，如果满足下列条件，则认为是重复的:
# 1) 它们具有相同的模式和大小;
# 2) 在将像素归一化为零均值和一方差之后，均方误差不超过0.1
def match(h1, h2):
    for p1 in h2ps[h1]:
        for p2 in h2ps[h2]:
            i1 = pil_image.open(expand_path(p1))
            i2 = pil_image.open(expand_path(p2))
            # 图像像素模式和尺寸相等则进行下一步
            if i1.mode != i2.mode or i1.size != i2.size: return False
            a1 = np.array(i1)  # 转化为numpy数组
            a1 = a1 - a1.mean()
            a1 = a1 / sqrt((a1 ** 2).mean())
            a2 = np.array(i2)
            a2 = a2 - a2.mean()
            a2 = a2 / sqrt((a2 ** 2).mean())
            a = ((a1 - a2) ** 2).mean()
            if a > 0.1: return False
    return True


# 判断p2size.pickle文件是否存在,否,则本地生成
if isfile(P2SIZE):
    print("P2SIZE exists.")
    with open(P2SIZE, 'rb') as f:
        p2size = pickle.load(f)
    # print(type(p2size));print(p2size)
else:
    # 图像id-图像尺寸,字典
    p2size = {}
    for p in tqdm(join):
        size = pil_image.open(expand_path(p)).size
        p2size[p] = size
    with open('P2SIZE', 'wb') as f:
        pickle.dump(p2size, f)

# 判断p2h.pickle文件是否存在,否,则本地生成
if isfile(P2H):
    print("P2H exists.")
    with open(P2H, 'rb') as f:
        p2h = pickle.load(f)  # 从指定文件中读出序列化前的obj对象
    # print(type(p2h));print(p2h)
else:
    # 计算训练和测试集中每个图像的哈希值。
    # 图像'image'-哈希值,字典。字典 p2h 为每张图片关联唯一图像ID（phash）,即pic2hash。
    p2h = {}
    for p in tqdm(join):
        img = pil_image.open(expand_path(p))
        h = phash(img)  # 计算图像的感知哈希值
        p2h[p] = h

    # 查找与给定hash值关联的所有图像
    # 哈希值-图像'image'列表,字典
    h2ps = {}
    # 同时迭代p2h字典的键和键值,即图像'image'和图像哈希值
    for p, h in p2h.items():
        if h not in h2ps: h2ps[h] = []
        if p not in h2ps[h]: h2ps[h].append(p)  # 一个哈希值可能有多个图像id

    # 找到所有不同的hash值
    hs = list(h2ps.keys())
    print('The length of hash-values-list is', length(hs))  # 哈希值列表

    # 如果图像足够接近，则关联两个hash值 (这部分非常慢: 算法复杂度 n^2 )
    # 哈希值-哈希值,字典
    h2h = {}
    for i, h1 in enumerate(tqdm(hs)):
        for h2 in hs[:i]:
            if h1 - h2 <= 6 and match(h1, h2):
                s1 = str(h1)
                s2 = str(h2)
                if s1 < s2: s1, s2 = s2, s1
                h2h[s1] = s2

    # 将相同hash的图像组合在一起，并用字符串格式的hash替换（更快，更可读）
    for p, h in p2h.items():
        h = str(h)
        if h in h2h: h = h2h[h]
        p2h[p] = h
# with open(P2H, 'wb') as f:
# 	pickle.dump(p2h, f)
print('The length of p2h is', len(p2h))
print('The first five elements of p2h-dict', list(p2h.items())[:5])


# 对于每个图像ID，选择首选的图像
def prefer(ps):
    if len(ps) == 1:
        return ps[0]
    best_p = ps[0]
    best_s = p2size[best_p]
    for i in range(1, len(ps)):
        p = ps[i]  # 图像id
        s = p2size[p]  # 根据图像id的得到图像尺寸
        if s[0] * s[1] > best_s[0] * best_s[1]:  # Select the image with highest resolution
            best_p = p
            best_s = s
    return best_p


# ---------------------------------------------------------------------------------------------
# 注意到33321张图像是如何仅使用33317个不同的图像ID。
h2ps = {}
for p, h in p2h.items():
    if h not in h2ps: h2ps[h] = []
    if p not in h2ps[h]: h2ps[h].append(p)
print('The length of h2ps is %d' % len(h2ps))
# For each hash value,select the prefered image
h2p = {}
for h, ps in h2ps.items():
    h2p[h] = prefer(ps)

print('The length of h2p is', len(h2p))
print('The first five elements of h2p-list', list(h2p.items())[:5])


# -----------------------------------------------------------------------------------------------
# 鲸鱼图像显示函数,每行显示2个图像
def show_whale(imgs, per_row=2):
    """
    imgs:The list of images path.
    """
    n = len(imgs)
    rows = (n + per_row - 1) // per_row  # 计算共有多少行
    cols = min(per_row, n)
    # 返回Figure object and array of Axes object
    fig, axes = plt.subplots(rows, cols, figsize=(24 // per_row * cols, 24 // per_row * rows))
    for ax in axes.flatten():
        ax.axis('off')
    # 同时迭代多个序列,结合enumerate,迭代显示图像
    for i, (img, ax) in enumerate(zip(imgs, axes.flatten())):
        ax.imshow(img.convert('RGB'))


# # 展示一些重复图像
# for h, ps in h2ps.items():
#     if len(ps) > 2:
#         print('Images:', ps)
#         imgs = [pil_image.open(expand_path(p)) for p in ps]
#         show_whale(imgs, per_row=len(ps))
#         break

p = list(tagged.keys())[312]  # list data
print('The name of the 312 image is', p)
# imgs = [
#     read_raw_image(p),
#     array_to_img(read_for_validation(p)),
#     array_to_img(read_for_training(p))
# ]
# show_whale(imgs, per_row=3)
# ------------------------------------------训练集数据架构-----------------------------------------------
# 图像哈希值-多个图像标签,字典（p-图像名,h-图像哈希值,w-图像鲸鱼种类,）
h2ws = {}
new_whale = 'new_whale'
# 找到与图像ID关联的所有鲸鱼. 它可能不明确, 因为重复的图像可能有不同的鲸鱼ID
for p, w in tagged.items():
    if w != new_whale:  # Use only identified whales
        h = p2h[p]  # 获取对应图像id的图像哈希值
        if h not in h2ws: h2ws[h] = []
        if w not in h2ws[h]: h2ws[h].append(w)
for h, ws in h2ws.items():
    if len(ws) > 1:
        h2ws[h] = sorted(ws)  # 对ws列表进行排序
    # print(ws)
    # print('There are len(ws) >1 ')
len(h2ws)  # 15696
# print(h2ws)
# 图像标签-多个图像哈希值,字典
w2hs = {}
# 对于每条鲸鱼, 找到明确的图像ID, 如果一个hash对应两个whale,就不将其加入到w2hs中
for h, ws in h2ws.items():
    if len(ws) == 1:  # Use only unambiguous pictures
        w = ws[0]
        if w not in w2hs: w2hs[w] = []
        if h not in w2hs[w]: w2hs[w].append(h)
num_hs = 0
for w, hs in w2hs.items():
    if len(hs) > 1:
        w2hs[w] = sorted(hs)  # 对hs列表进行排序
        num_hs += 1
    # print(hs)
print('There are %d whale that have images more than 2' % num_hs)
print('The length of w2hs-dict is', len(w2hs.values()))
# 训练集图像哈希值列表
train = []
# 获取训练图像哈希值列表, 里面只保留至少有两个图像的鲸鱼
for hs in w2hs.values():
    if len(hs) > 1:
        train += hs
random.shuffle(train)
train_set = set(train)  # 创建无序不重复元素集
print('The number of  total train phash vlues list is', len(train))
# print(train)
# 图像标签-图像哈希值, 字典(一个鲸鱼标签,图像哈希值至少有两个)
w2ts = {}
# 将鲸鱼id映射到相关的训练图片的hash表上去
for w, hs in w2hs.items():
    for h in hs:
        if h in train_set:
            if w not in w2ts: w2ts[w] = []
            if h not in w2ts[w]: w2ts[w].append(h)
for w, ts in w2ts.items():
    w2ts[w] = np.array(ts)

t2i = {}
# 将训练图片hash值映射到'train'数组中的索引
for i, t in enumerate(train):
    t2i[t] = i
# print('The number of whale id is',len(w2ts.keys()))

print('The number of items in p2size,h2ps,h2ws,w2hs,w2ts:', len(p2size), len(h2ps), len(h2ws), len(w2hs), len(w2ts))

