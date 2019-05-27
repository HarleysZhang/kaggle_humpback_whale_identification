# Read the dataset description
import gzip
# Read or generate p2h, a dictionary of image name to image id (picture to hash)
import pickle
import platform
import random
# Suppress annoying stderr output when importing keras.
import sys
from lapjv import lapjv
from math import sqrt
# Determine the size of each image
from os.path import isfile

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image as pil_image
from imagehash import phash
from keras import backend as K
from keras import regularizers
from keras.engine.topology import Input
from keras.layers import Activation, Add, BatchNormalization, Concatenate, Conv2D, Dense, Flatten, GlobalMaxPooling2D, \
	Lambda, MaxPooling2D, Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.utils import Sequence
from pandas import read_csv
from scipy.ndimage import affine_transform
from tqdm import tqdm
import time
import os
import threading

src_dir = os.getcwd()
src_dir = src_dir.replace('\\', '/')
print(src_dir)

TRAIN_DF = os.path.join(src_dir, "dataset/train.csv")
SUB_DF = os.path.join(src_dir, "dataset/sample_submission.csv")
TRAIN = os.path.join(src_dir, "dataset/train")
TEST = os.path.join(src_dir, "dataset/test")
P2H = os.path.join(src_dir, "dataset/p2h.pickle")
P2SIZE = os.path.join(src_dir, "dataset/p2size.pickle")
BB_DF = os.path.join(src_dir, "dataset/bounding_boxes.csv")

# 把Windows下os.path.join()生成的反斜杠（\）全部替换为斜杠（/）
TRAIN_DF = TRAIN_DF.replace('\\', '/')
SUB_DF = SUB_DF.replace('\\', '/')
TRAIN = TRAIN.replace('\\', '/')
TEST = TEST.replace('\\', '/')
P2H = P2H.replace('\\', '/')
P2SIZE = P2SIZE.replace('\\', '/')
BB_DF = BB_DF.replace('\\', '/')

tagged = dict([(p, w) for _, p, w in read_csv(TRAIN_DF).to_records()])
submit = [p for _, p, _ in read_csv(SUB_DF).to_records()]
join = list(tagged.keys()) + submit

if not os.path.exists("./logs"):
	os.mkdir("./logs")

# 根据图像Id,获取文件完整路径
def expand_path(p):
	if isfile(os.path.join(TRAIN, p)):
		return os.path.join(TRAIN, p)
	if isfile(os.path.join(TEST, p)):
		return os.path.join(TEST, p)
	return p


if isfile(P2SIZE):
	print("P2SIZE exists.")
	with open(P2SIZE, 'rb') as f:
		p2size = pickle.load(f)
else:
	p2size = {}
	for p in tqdm(join):
		size = pil_image.open(expand_path(p)).size
		p2size[p] = size


def match(h1, h2):
	for p1 in h2ps[h1]:
		for p2 in h2ps[h2]:
			i1 = pil_image.open(expand_path(p1))
			i2 = pil_image.open(expand_path(p2))
			if i1.mode != i2.mode or i1.size != i2.size: return False
			a1 = np.array(i1)
			a1 = a1 - a1.mean()
			a1 = a1 / sqrt((a1 ** 2).mean())
			a2 = np.array(i2)
			a2 = a2 - a2.mean()
			a2 = a2 / sqrt((a2 ** 2).mean())
			a = ((a1 - a2) ** 2).mean()
			if a > 0.1: return False
	return True


if isfile(P2H):
	print("P2H exists.")
	with open(P2H, 'rb') as f:
		p2h = pickle.load(f)
else:
	# Compute phash for each image in the training and test set.
	p2h = {}
	for p in tqdm(join):
		img = pil_image.open(expand_path(p))
		h = phash(img)
		p2h[p] = h

	# Find all images associated with a given phash value.
	h2ps = {}
	for p, h in p2h.items():
		if h not in h2ps: h2ps[h] = []
		if p not in h2ps[h]: h2ps[h].append(p)

	# Find all distinct phash values
	hs = list(h2ps.keys())

	# If the images are close enough, associate the two phash values (this is the slow part: n^2 algorithm)
	h2h = {}
	for i, h1 in enumerate(tqdm(hs)):
		for h2 in hs[:i]:
			if h1 - h2 <= 6 and match(h1, h2):
				s1 = str(h1)
				s2 = str(h2)
				if s1 < s2: s1, s2 = s2, s1
				h2h[s1] = s2

	# Group together images with equivalent phash, and replace by string format of phash (faster and more readable)
	for p, h in p2h.items():
		h = str(h)
		if h in h2h: h = h2h[h]
		p2h[p] = h
#     with open(P2H, 'wb') as f:
#         pickle.dump(p2h, f)
# For each image id, determine the list of pictures
h2ps = {}
for p, h in p2h.items():
	if h not in h2ps: h2ps[h] = []
	if p not in h2ps[h]: h2ps[h].append(p)


def show_whale(imgs, per_row=2):
	n = len(imgs)
	rows = (n + per_row - 1) // per_row
	cols = min(per_row, n)
	fig, axes = plt.subplots(rows, cols, figsize=(24 // per_row * cols, 24 // per_row * rows))
	for ax in axes.flatten(): ax.axis('off')
	for i, (img, ax) in enumerate(zip(imgs, axes.flatten())): ax.imshow(img.convert('RGB'))


def read_raw_image(p):
	img = pil_image.open(expand_path(p))
	return img


# For each images id, select the prefered image
def prefer(ps):
	if len(ps) == 1: return ps[0]
	best_p = ps[0]
	best_s = p2size[best_p]
	for i in range(1, len(ps)):
		p = ps[i]
		s = p2size[p]
		if s[0] * s[1] > best_s[0] * best_s[1]:  # Select the image with highest resolution
			best_p = p
			best_s = s
	return best_p


h2p = {}
for h, ps in h2ps.items():
	h2p[h] = prefer(ps)
len(h2p), list(h2p.items())[:5]

# Read the bounding box data from the bounding box kernel (see reference above)
p2bb = pd.read_csv(BB_DF).set_index("Image")

old_stderr = sys.stderr
sys.stderr = open('/dev/null' if platform.system() != 'Windows' else 'nul', 'w')

sys.stderr = old_stderr

img_shape = (512, 512, 1)  # The image shape used by the model
anisotropy = 2.15  # The horizontal compression ratio
crop_margin = 0.05  # The margin added around the bounding box to compensate for bounding box inaccuracy


def build_transform(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
	"""
	Build a transformation matrix with the specified characteristics.
	"""
	rotation = np.deg2rad(rotation)
	shear = np.deg2rad(shear)
	rotation_matrix = np.array(
		[[np.cos(rotation), np.sin(rotation), 0], [-np.sin(rotation), np.cos(rotation), 0], [0, 0, 1]])
	shift_matrix = np.array([[1, 0, height_shift], [0, 1, width_shift], [0, 0, 1]])
	shear_matrix = np.array([[1, np.sin(shear), 0], [0, np.cos(shear), 0], [0, 0, 1]])
	zoom_matrix = np.array([[1.0 / height_zoom, 0, 0], [0, 1.0 / width_zoom, 0], [0, 0, 1]])
	shift_matrix = np.array([[1, 0, -height_shift], [0, 1, -width_shift], [0, 0, 1]])
	return np.dot(np.dot(rotation_matrix, shear_matrix), np.dot(zoom_matrix, shift_matrix))


def read_cropped_image(p, augment):
	"""
	@param p : the name of the picture to read
	@param augment: True/False if data augmentation should be performed
	@return a numpy array with the transformed image
	"""
	# If an image id was given, convert to filename
	if p in h2p:
		p = h2p[p]
	size_x, size_y = p2size[p]

	# Determine the region of the original image we want to capture based on the bounding box.
	row = p2bb.loc[p]
	x0, y0, x1, y1 = row['x0'], row['y0'], row['x1'], row['y1']
	dx = x1 - x0
	dy = y1 - y0
	x0 -= dx * crop_margin
	x1 += dx * crop_margin + 1
	y0 -= dy * crop_margin
	y1 += dy * crop_margin + 1
	if x0 < 0:
		x0 = 0
	if x1 > size_x:
		x1 = size_x
	if y0 < 0:
		y0 = 0
	if y1 > size_y:
		y1 = size_y
	dx = x1 - x0
	dy = y1 - y0
	if dx > dy * anisotropy:
		dy = 0.5 * (dx / anisotropy - dy)
		y0 -= dy
		y1 += dy
	else:
		dx = 0.5 * (dy * anisotropy - dx)
		x0 -= dx
		x1 += dx

	# Generate the transformation matrix
	trans = np.array([[1, 0, -0.5 * img_shape[0]], [0, 1, -0.5 * img_shape[1]], [0, 0, 1]])
	trans = np.dot(np.array([[(y1 - y0) / img_shape[0], 0, 0], [0, (x1 - x0) / img_shape[1], 0], [0, 0, 1]]), trans)
	if augment:
		trans = np.dot(build_transform(
			random.uniform(-5, 5),
			random.uniform(-5, 5),
			random.uniform(0.8, 1.0),
			random.uniform(0.8, 1.0),
			random.uniform(-0.05 * (y1 - y0), 0.05 * (y1 - y0)),
			random.uniform(-0.05 * (x1 - x0), 0.05 * (x1 - x0))
		), trans)
	trans = np.dot(np.array([[1, 0, 0.5 * (y1 + y0)], [0, 1, 0.5 * (x1 + x0)], [0, 0, 1]]), trans)

	# Read the image, transform to black and white and comvert to numpy array
	img = read_raw_image(p).convert('L')
	img = img_to_array(img)

	# Apply affine transformation
	matrix = trans[:2, :2]
	offset = trans[:2, 2]
	img = img.reshape(img.shape[:-1])
	img = affine_transform(img, matrix, offset, output_shape=img_shape[:-1],
						   order=1, mode='constant', cval=np.average(img))
	img = img.reshape(img_shape)

	# Normalize to zero mean and unit variance
	img -= np.mean(img, keepdims=True)
	img /= np.std(img, keepdims=True) + K.epsilon()
	return img


def read_for_training(p):
	"""
	Read and preprocess an image with data augmentation (random transform).
	"""
	return read_cropped_image(p, True)


def read_for_validation(p):
	"""
	Read and preprocess an image without data augmentation (use for testing).
	"""
	return read_cropped_image(p, False)


p = list(tagged.keys())[312]


def subblock(x, filter, **kwargs):
	x = BatchNormalization()(x)
	y = x
	y = Conv2D(filter, (1, 1), activation='relu', **kwargs)(y)  # Reduce the number of features to 'filter'
	y = BatchNormalization()(y)
	y = Conv2D(filter, (3, 3), activation='relu', **kwargs)(y)  # Extend the feature field
	y = BatchNormalization()(y)
	y = Conv2D(K.int_shape(x)[-1], (1, 1), **kwargs)(y)  # no activation # Restore the number of original features
	y = Add()([x, y])  # Add the bypass connection
	y = Activation('relu')(y)
	return y


def build_model(lr, l2, activation='sigmoid'):
	##############
	# BRANCH MODEL
	##############
	regul = regularizers.l2(l2)
	optim = Adam(lr=lr)
	kwargs = {'padding': 'same', 'kernel_regularizer': regul}

	inp = Input(shape=img_shape)  # 384x384x1
	x = Conv2D(64, (9, 9), strides=2, activation='relu', **kwargs)(inp)

	x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 96x96x64
	for _ in range(2):
		x = BatchNormalization()(x)
		x = Conv2D(64, (3, 3), activation='relu', **kwargs)(x)

	x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 48x48x64
	x = BatchNormalization()(x)
	x = Conv2D(128, (1, 1), activation='relu', **kwargs)(x)  # 48x48x128
	for _ in range(4):
		x = subblock(x, 64, **kwargs)

	x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 24x24x128
	x = BatchNormalization()(x)
	x = Conv2D(256, (1, 1), activation='relu', **kwargs)(x)  # 24x24x256
	for _ in range(4):
		x = subblock(x, 64, **kwargs)

	x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 12x12x256
	x = BatchNormalization()(x)
	x = Conv2D(384, (1, 1), activation='relu', **kwargs)(x)  # 12x12x384
	for _ in range(4):
		x = subblock(x, 96, **kwargs)

	x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 6x6x384
	x = BatchNormalization()(x)
	x = Conv2D(512, (1, 1), activation='relu', **kwargs)(x)  # 6x6x512
	for _ in range(4):
		x = subblock(x, 128, **kwargs)

	x = GlobalMaxPooling2D()(x)  # 512, 输出shape: 形如(nb_samples, channels)的2D张量
	branch_model = Model(inp, x)

	############
	# HEAD MODEL
	############
	mid = 32
	xa_inp = Input(shape=branch_model.output_shape[1:])
	xb_inp = Input(shape=branch_model.output_shape[1:])
	x1 = Lambda(lambda x: x[0] * x[1])([xa_inp, xb_inp])
	x2 = Lambda(lambda x: x[0] + x[1])([xa_inp, xb_inp])
	x3 = Lambda(lambda x: K.abs(x[0] - x[1]))([xa_inp, xb_inp])
	x4 = Lambda(lambda x: K.square(x))(x3)
	x = Concatenate()([x1, x2, x3, x4])
	x = Reshape((4, branch_model.output_shape[1], 1), name='reshape1')(x)

	# Per feature NN with shared weight is implemented using CONV2D with appropriate stride.
	x = Conv2D(mid, (4, 1), activation='relu', padding='valid')(x)
	x = Reshape((branch_model.output_shape[1], mid, 1))(x)
	x = Conv2D(1, (1, mid), activation='linear', padding='valid')(x)
	x = Flatten(name='flatten')(x)

	# Dense layer的实现为加权和.
	x = Dense(1, use_bias=True, activation=activation, name='weighted-average')(x)
	head_model = Model([xa_inp, xb_inp], x, name='head')

	########################
	# SIAMESE NEURAL NETWORK
	########################
	# Complete model is constructed by calling the branch model on each input image,
	# and then the head model on the resulting 512-vectors.
	img_a = Input(shape=img_shape)
	img_b = Input(shape=img_shape)
	xa = branch_model(img_a)
	xb = branch_model(img_b)
	x = head_model([xa, xb])
	model = Model([img_a, img_b], x)
	model.compile(optim, loss='binary_crossentropy', metrics=['binary_crossentropy', 'acc'])
	return model, branch_model, head_model


model, branch_model, head_model = build_model(64e-5, 0)

# Find all the whales associated with an image id.
# It can be ambiguous as duplicate images may have different whale ids.
h2ws = {}
new_whale = 'new_whale'
for p, w in tagged.items():
	if w != new_whale:  # Use only identified whales
		h = p2h[p]
		if h not in h2ws: h2ws[h] = []
		if w not in h2ws[h]: h2ws[h].append(w)
for h, ws in h2ws.items(): # print(len(h2ws))    # 15696
	if len(ws) > 1:
		h2ws[h] = sorted(ws)

# For each whale, find the unambiguous images ids.
w2hs = {}
for h, ws in h2ws.items():
	if len(ws) == 1:  # Use only unambiguous pictures
		w = ws[0]
		if w not in w2hs: w2hs[w] = []
		if h not in w2hs[w]: w2hs[w].append(h)
for w, hs in w2hs.items():
	if len(hs) > 1:
		w2hs[w] = sorted(hs)
# print(len(w2hs))    # 5004

# Find the list of training images, keep only whales with at least two images.
# 可用于训练的图像总共13263张
train = []  # A list of training image ids
for hs in w2hs.values():
	if len(hs) > 1:
		train += hs
random.shuffle(train)
train_set = set(train)
# print(len(train))    # 13263

w2ts = {}  # Associate the image ids from train to each whale id.
for w, hs in w2hs.items():
	for h in hs:
		if h in train_set:
			if w not in w2ts:
				w2ts[w] = []
			if h not in w2ts[w]:
				w2ts[w].append(h)
for w, ts in w2ts.items():
	w2ts[w] = np.array(ts)
# print(len(w2ts))    # 2931

t2i = {}  # The position in train of each training image id
for i, t in enumerate(train):
	t2i[t] = i


def my_lapjv(score):
	num_threads = 6
	batch = score.shape[0] // (num_threads - 1)
	if score.shape[0] % batch <= 3:
		num_threads = 5
		if score.shape[0] % batch is not 0:
			batch += 1
	# print(batch)
	tmp = num_threads * [None]
	threads = []    # 创建线程数组
	thread_input = num_threads * [None]
	thread_idx = 0
	for start in range(0, score.shape[0], batch):
		end = min(score.shape[0], start + batch)
		# print('%d %d' % (start, end))
		thread_input[thread_idx] = score[start:end, start:end]
		thread_idx += 1

	def worker(data_idx):
		x, _, _ = lapjv(thread_input[data_idx]) # 根据n*n的一个分数方阵来计算，以总体最小代价实现任务分配，每一个数值不会重复分配
		tmp[data_idx] = x + data_idx * batch    # 索引更新

	# print("Start worker threads")
	for i in range(num_threads):
		t = threading.Thread(target=worker, args=(i,), daemon=True)    # 创建子线程
		t.start()            # 开始执行线程
		threads.append(t)    # 把创建好的t线程装进threads线程数组
	for t in threads:
		if t is not None:
			t.join()
	x = np.concatenate(tmp)    # Join a sequence of arrays along an existing axis.
	# print("LAP completed")
	return x


class TrainingData(Sequence):
	def __init__(self, score, steps=1000, batch_size=32):
		"""
		@param score the cost matrix for the picture matching
		@param steps the number of epoch we are planning with this score matrix
		"""
		super(TrainingData, self).__init__()
		self.score = -score  # 为使用lapjv, 使得lapjv找出来的索引对应分数最大
		self.steps = steps
		self.batch_size = batch_size
		for ts in w2ts.values():
			idxs = [t2i[t] for t in ts]
			for i in idxs:
				for j in idxs:
					self.score[i, j] = 10000.0  # 为匹配鲸鱼设置一个很大的值 - 消除了这种潜在的配对
		self.on_epoch_end()

	def __getitem__(self, index):
		start = self.batch_size * index
		end = min(start + self.batch_size, len(self.match) + len(self.unmatch))
		size = end - start
		assert size > 0
		a = np.zeros((size,) + img_shape, dtype=K.floatx())
		b = np.zeros((size,) + img_shape, dtype=K.floatx())
		c = np.zeros((size, 1), dtype=K.floatx())
		j = start // 2
		for i in range(0, size, 2):
			a[i, :, :, :] = read_for_training(self.match[j][0])
			b[i, :, :, :] = read_for_training(self.match[j][1])
			c[i, 0] = 1  # This is a match
			a[i + 1, :, :, :] = read_for_training(self.unmatch[j][0])
			b[i + 1, :, :, :] = read_for_training(self.unmatch[j][1])
			c[i + 1, 0] = 0  # Different whales
			j += 1
		return [a, b], c

	def on_epoch_end(self):
		if self.steps <= 0: return  # Skip this on the last epoch.
		self.steps -= 1
		self.match = []
		self.unmatch = []
		x = my_lapjv(self.score)  # Solve the linear assignment problem
		print(x.shape)    # (13263,)
		y = np.arange(len(x), dtype=np.int32)

		# Compute a derangement for matching whales
		# match, 能匹配的图像对列表, 元祖里面的图像是一种图像
		for ts in w2ts.values():
			d = ts.copy()
			while True:
				random.shuffle(d)
				if not np.any(ts == d): break    # 直到每个索引上的图像id都不相等ｇ
			for ab in zip(ts, d): self.match.append(ab)    # (13263), the list of tuples,

		# Construct unmatched whale pairs from the LAP solution.
		for i, j in zip(x, y):
			if i == j:
				print(self.score)
				print(x)
				print(y)
				print(i, j)
			assert i != j
			self.unmatch.append((train[i], train[j]))    # 不能匹配的图像对列表, 元祖里面的图像不是一种图像

		# Force a different choice for an eventual next epoch.
		self.score[x, y] = 10000.0
		self.score[y, x] = 10000.0
		random.shuffle(self.match)
		random.shuffle(self.unmatch)
		# print(len(self.match), len(train), len(self.unmatch), len(train))
		assert len(self.match) == len(train) and len(self.unmatch) == len(train)

	def __len__(self):
		return (len(self.match) + len(self.unmatch) + self.batch_size - 1) // self.batch_size


# Test on a batch of 32 with random costs.
score = np.random.random_sample(size=(len(train), len(train)))
data = TrainingData(score)
x = len(data)
print(x)
(a, b), c = data[1]


# A Keras generator to evaluate only the BRANCH MODEL
class FeatureGen(Sequence):
	def __init__(self, data, batch_size=64, verbose=0):
		super(FeatureGen, self).__init__()
		self.data = data
		self.batch_size = batch_size
		self.verbose = verbose
		if self.verbose > 0: self.progress = tqdm(total=len(self), desc='Features')

	def __getitem__(self, index):
		start = self.batch_size * index
		size = min(len(self.data) - start, self.batch_size)
		a = np.zeros((size,) + img_shape, dtype=K.floatx())
		for i in range(size): a[i, :, :, :] = read_for_validation(self.data[start+i])    # 图像数组数据,
		# start = self.batch_size * index
		# size = min(len(self.data) - start, self.batch_size)
		# a = np.zeros((size,) + img_shape, dtype=K.floatx())
		# for i in range(size): a[i, :, :, :] = read_for_validation(self.data[start + i])
		if self.verbose > 0:
			self.progress.update()
			if self.progress.n >= len(self): self.progress.close()
		return a    # 返回图像数组数据, (batch_size?, img_size, img_size, 1)

	def __len__(self):
		return (len(self.data) + self.batch_size - 1) // self.batch_size


# A Keras generator to evaluate on the HEAD MODEL on features already pre-computed.
# It computes only the upper triangular matrix of the cost matrix if y is None.
class ScoreGen(Sequence):
	def __init__(self, x, y=None, batch_size=2048, verbose=0):
		super(ScoreGen, self).__init__()
		self.x = x    # x = features, shape (2725/13263, 512)
		self.y = y
		self.batch_size = batch_size
		self.verbose = verbose
		if y is None:
			self.y = self.x
			self.ix, self.iy = np.triu_indices(x.shape[0], 1)    # 上三角矩阵对角线偏移１位置后的索引
			print(self.ix.shape, self.iy.shape)
		else:
			self.iy, self.ix = np.indices((y.shape[0], x.shape[0]))
			self.ix = self.ix.reshape((self.ix.size,))
			self.iy = self.iy.reshape((self.iy.size,))
		self.subbatch = (len(self.x) + self.batch_size - 1) // self.batch_size
		if self.verbose > 0:
			self.progress = tqdm(total=len(self), desc='Scores')

	def __getitem__(self, index):
		start = index * self.batch_size
		end = min(start + self.batch_size, len(self.ix))
		a = self.y[self.iy[start:end], :]    # 图像特征向量获取, shape (2048, 512)
		b = self.x[self.ix[start:end], :]    # 图像特征向量获取
		if self.verbose > 0:
			self.progress.update()
			if self.progress.n >= len(self): self.progress.close()
		return [a, b]

	def __len__(self):
		return (len(self.ix) + self.batch_size - 1) // self.batch_size


def set_lr(model, lr):
	K.set_value(model.optimizer.lr, float(lr))


def get_lr(model):
	return K.get_value(model.optimizer.lr)


def score_reshape(score, x, y=None):
	"""
    将packed matrix的'得分'转换为方阵
    @param score the packed matrix
    @param x 第一张图像的特征张量
    @param y 第二张图像的张量，如果与x不同
    @结果为方阵
    """
	if y is None:
		# When y is None, score is a packed upper triangular matrix.
		# Unpack, and transpose to form the symmetrical lower triangular matrix.
		# 当y为None时, 得分是打包的上三角矩阵
		# 解包, 并转置以形成对称的下三角矩阵
		m = np.zeros((x.shape[0], x.shape[0]), dtype=K.floatx())  # 利用features.shape[0]创建２维度的零数组
		# numpy.squeeze() 从数组的形状中删除单维度条目，即把shape中为1的维度去掉
		m[np.triu_indices(x.shape[0], 1)] = score.squeeze()       # 和ScoreGen产生的输入给head_model的a, b特征向量索引对应
		# 加上m的转置矩阵(下三角矩阵)
		m += m.transpose()
	else:
		m = np.zeros((y.shape[0], x.shape[0]), dtype=K.floatx())
		iy, ix = np.indices((y.shape[0], x.shape[0]))
		ix = ix.reshape((ix.size,))
		iy = iy.reshape((iy.size,))
		m[iy, ix] = score.squeeze()
	return m


# def compute_score(verbose=0):
# 	"""
# 	Compute the score matrix by scoring every pictures from the training set against every other picture O(n^2).
# 	"""
# 	features = branch_model.predict_generator(FeatureGen(train, verbose=verbose), max_queue_size=12, workers=6,
# 											  verbose=1)
# 	print(features.shape) # (92786253, 1)
# 	score = head_model.predict_generator(ScoreGen(features, verbose=verbose), max_queue_size=12, workers=6, verbose=1)
# 	print(score.shape)    # (13263,)
# 	score = score_reshape(score, features)
# 	print(score.shape)    # (13263, 13263)
# 	return features, score


def compute_score(verbose=1):
	"""
	Compute the score matrix by scoring every pictures from the training set against every other picture O(n^2).
	"""
	features = branch_model.predict_generator(
		FeatureGen(train, batch_size=64, verbose=verbose),
		max_queue_size=12, workers=6, verbose=0)
	num_threads = 6
	batch = features.shape[0] // (num_threads - 1)
	if features.shape[0] % batch <= 3:
		num_threads = 5
		if features.shape[0] % batch is not 0:
			batch += 1
	all_score = []
	for start in range(0, features.shape[0], batch):
		end = min(features.shape[0], start + batch)
		temp_features = features[start:end, :]
		temp_score = head_model.predict_generator(
			ScoreGen(temp_features, batch_size=2048, verbose=verbose),
			max_queue_size=12, workers=6, verbose=0)
		temp_score = score_reshape(temp_score, temp_features)
		all_score.append(temp_score)
	score = np.zeros((features.shape[0], features.shape[0]), dtype=K.floatx())
	for i, start in enumerate(range(0, features.shape[0], batch)):
		end = min(features.shape[0], start + batch)
		score[start:end, start:end] = all_score[i]
	return features, score


def make_steps(step, ampl):
	"""
	Perform training epochs
	@param step Number of epochs to perform
	@param ampl the K, the randomized component of the score matrix.
	"""
	global w2ts, t2i, steps, features, score, histories

	# shuffle the training pictures
	random.shuffle(train)    # list, length is 13263

	# 将鲸鱼id映射到相关的训练图片的hash表上去
	w2ts = {}
	for w, hs in w2hs.items():
		for h in hs:
			if h in train_set:
				if w not in w2ts: w2ts[w] = []
				if h not in w2ts[w]: w2ts[w].append(h)
	for w, ts in w2ts.items(): w2ts[w] = np.array(ts)

	# Map training picture hash value to index in 'train' array
	t2i = {}
	for i, t in enumerate(train): t2i[t] = i

	# Compute the match score for each picture pair
	features, score = compute_score()    # features shape (13263, 512), score shape (13263, 13263)

	# Train the model for 'step' epochs
	history = model.fit_generator(
		TrainingData(score + ampl * np.random.random_sample(size=score.shape), steps=step, batch_size=32),
		initial_epoch=steps,
		epochs=steps + step,
		max_queue_size=12,
		workers=6,
		verbose=2).history
	steps += step

	# Collect history data
	history['epochs'] = steps
	history['ms'] = np.mean(score)
	history['lr'] = get_lr(model)
	print(history['epochs'], history['lr'], history['ms'])
	histories.append(history)

	# Collect history data
	history['epochs'] = steps
	history['ms'] = np.mean(score)
	history['lr'] = get_lr(model)
	print(history['epochs'], history['lr'], history['ms'])
	histories.append(history)

	# 将历史数据写入文本保存
	log = open('model_log.txt', "a")
	log_str = '.'
	his_loss, his_bc, his_acc = history['loss'], history['binary_crossentropy'], history['acc']
	for ep in range(len(his_loss)):
		log_str += 'Epoch %d/%d - ' % (steps - len(his_loss) + 1 + ep, steps)
		log_str += 'loss: %.4f - binary_crossentropy: %.4f - acc: %.4f\n' % (his_loss[ep], his_bc[ep], his_acc[ep])
	log_str += 'Epoch %d, lr: %.8f\n' % (history['epochs'], history['lr'])
	log_str += '\n'
	log.write(log_str)  # 将字符串以'a'模式写入文本文件


histories = []
steps = 0

if isfile('./models/standard_epoch450.model'):
	tmp = keras.models.load_model('./models/standard_epoch450.model')
	model.set_weights(tmp.get_weights())
	print('Load model success!')
else:
	log = open('model_log.txt', "a")
	start_str = '\n[Siamese========lapjv========Size512*3========Epoch450========MultiThreads]\n'
	log.write(start_str)
	print('Start information write success!')

	# epoch -> 10
	make_steps(10, 1000)

	ampl = 100.0
	for _ in range(2):
		print('noise ampl.  = ', ampl)
		make_steps(5, ampl)
		ampl = max(1.0, 100 ** -0.1 * ampl)

	# epoch -> 110
	for _ in range(18): make_steps(5, 1.0)
	model.save('./models/standard_epoch110.model', overwrite=True)

	# epoch -> 160
	set_lr(model, 16e-5)
	for _ in range(10): make_steps(5, 0.5)
	model.save('./models/standard_epoch160.model', overwrite=True)

	# epoch -> 200
	set_lr(model, 4e-5)
	for _ in range(8): make_steps(5, 0.25)
	model.save('./models/standard_epoch200.model', overwrite=True)

	# epoch -> 210
	set_lr(model, 1e-5)
	for _ in range(2): make_steps(5, 0.25)
	model.save('./models/standard_epoch210.model', overwrite=True)

	# epoch -> 220
	for _ in range(2): make_steps(5, 0.25)
	model.save('./models/standard_epoch220.model', overwrite=True)

	# epoch -> 230
	for _ in range(2): make_steps(5, 0.25)
	model.save('./models/standard_epoch230.model', overwrite=True)

	# epoch -> 240
	for _ in range(2): make_steps(5, 0.25)
	model.save('./models/standard_epoch240.model', overwrite=True)

	# epoch -> 250
	for _ in range(2): make_steps(5, 0.25)
	model.save('./models/standard_epoch250.model', overwrite=True)

	# epoch -> 300
	weights = model.get_weights()

	model, branch_model, head_model = build_model(64e-5, 0.0002)
	model.set_weights(weights)

	for _ in range(10): make_steps(5, 1.0)
	model.save('./models/standard_epoch300.model', overwrite=True)

	tmp = keras.models.load_model('./models/standard_epoch300.model')
	model.set_weights(tmp.get_weights())
	print('Load standard_epoch300 model success!')

	# epoch -> 350
	set_lr(model, 16e-5)
	for _ in range(10): make_steps(5, 0.5)
	model.save('./models/standard_epoch350.model', overwrite=True)

	# epoch -> 390
	set_lr(model, 4e-5)
	for _ in range(8): make_steps(5, 0.25)
	model.save('./models/standard_epoch390.model', overwrite=True)

	# epoch -> 400
	set_lr(model, 1e-5)
	for _ in range(2): make_steps(5, 0.25)
	model.save('./models/standard_epoch400.model', overwrite=True)

	# epoch -> 410
	set_lr(model, 1e-5)
	for _ in range(2): make_steps(5, 0.25)
	model.save('./models/standard_epoch410.model', overwrite=True)

	# epoch -> 420
	for _ in range(2): make_steps(5, 0.25)
	model.save('./models/standard_epoch420.model', overwrite=True)

	# epoch -> 430
	for _ in range(2): make_steps(5, 0.25)
	model.save('./models/standard_epoch430.model', overwrite=True)

	# epoch -> 440
	for _ in range(2): make_steps(5, 0.25)
	model.save('./models/standard_epoch440.model', overwrite=True)

	# epoch -> 450
	for _ in range(2): make_steps(5, 0.25)
	model.save('./models/standard_epoch450.model', overwrite=True)

model.summary()


def prepare_submission(threshold, filename):
	"""
	Generate a Kaggle submission file.
	@param threshold the score given to 'new_whale'
	@param filename the submission file name
	"""
	vtop = 0
	vhigh = 0
	pos = [0, 0, 0, 0, 0, 0]
	with open(filename, 'wt', newline='\n') as f:
		f.write('Image,Id\n')
		for i, p in enumerate(tqdm(submit)):
			t = []
			s = set()
			a = score[i, :]
			# 对a数组进行排序, 并返回反向的索引迭代器
			for j in list(reversed(np.argsort(a))):
				h = known[j]
				if a[j] < threshold and new_whale not in s:
					pos[len(t)] += 1
					s.add(new_whale)
					t.append(new_whale)
					if len(t) == 5: break;
				for w in h2ws[h]:
					assert w != new_whale
					if w not in s:
						if a[j] > 1.0:
							vtop += 1
						elif a[j] >= threshold:
							vhigh += 1
						s.add(w)
						t.append(w)
						if len(t) == 5: break;
				if len(t) == 5: break;
			if new_whale not in s: pos[5] += 1
			assert len(t) == 5 and len(s) == 5
			f.write(p + ',' + ' '.join(t[:5]) + '\n')
	return vtop, vhigh, pos


def prepare_sub_and_bootstrap_pickle(threshold, file_path):
	"""
	Generate a Kaggle submission file.
	@param threshold the score given to 'new_whale'
	@param filename the submission file name
	"""
	vtop = 0
	bootstrap_pickle = {}
	file_name = file_path + 'sub_999999.csv'
	pickle_name = file_path + 'bootstrap.pickle'
	with open(file_name, 'wt', newline='\n') as f:
		f.write('Image,Id\n')
		for i, p in enumerate(tqdm(submit)):
			t = []
			a = score[i, :]
			for j in list(reversed(np.argsort(a))):
				if a[j] > threshold:
					vtop += 1
					h = known[j]
					w = h2ws[h][0]
					bootstrap_pickle[p] = w
					for k in range(5):
						t.append(w)
				else:
					w = 'none'
					for k in range(5):
						t.append(w)
				if len(t) >= 5: break;
			f.write(p + ',' + ' '.join(t[:5]) + '\n')
		output_bootstrap = open(pickle_name, 'wb')
		pickle.dump(bootstrap_pickle, output_bootstrap)
	print("vtop = %d" % vtop)
	return vtop


# Find elements from training sets not 'new_whale'
tic = time.time()
h2ws = {}
for p, w in tagged.items():
	if w != new_whale:  # Use only identified whales
		h = p2h[p]
		if h not in h2ws: h2ws[h] = []
		if w not in h2ws[h]: h2ws[h].append(w)
known = sorted(list(h2ws.keys()))

# Dictionary of picture indices
h2i = {}
for i, h in enumerate(known): h2i[h] = i

# Evaluate the model.
fknown = branch_model.predict_generator(FeatureGen(known), max_queue_size=20, workers=10, verbose=0)
fsubmit = branch_model.predict_generator(FeatureGen(submit), max_queue_size=20, workers=10, verbose=0)
score = head_model.predict_generator(ScoreGen(fknown, fsubmit), max_queue_size=20, workers=10, verbose=0)
score = score_reshape(score, fknown, fsubmit)

# 判断submits目录是否存在
if os.path.isdir('./submits'):
	print('The directory of outputs has been created')
else:
	os.mkdir('./submits')

# Generate the subsmission file.
prepare_submission(0.99, './submits/submission.csv')
prepare_sub_and_bootstrap_pickle(0.999999, './')
toc = time.time()
print("Submission time: ", (toc - tic) / 60.)
