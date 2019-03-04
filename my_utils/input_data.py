# coding:utf-8
# filename:input_data.py
# function:鲸鱼识别程序,数据输入模块

from __future__ import absolute_import
# from . import data_preprocess    # 导入数据预处理模块
import sys
import platform
import threading
import numpy as np
import random
from keras import backend as K
from scipy.ndimage import affine_transform
from keras.utils import Sequence
from lapjv import lapjv
from keras.preprocessing.image import img_to_array, array_to_img

# 抑制导入keras时烦人的stderr输出
old_stderr = sys.stderr
sys.stderr = open('/dev/null' if platform.system() != 'Windows' else 'nul', 'w')

sys.stderr = old_stderr
img_shape = (512, 512, 3)  # 模型使用的图像形状
anisotropy = 2.15  # 水平压缩比
crop_margin = 0.05  # 在边界框周围添加余量以补偿边界框的不精确性


# -----------------------------------------------图像预处理------------------------------------------------
# 读取指定路径图像
def read_raw_image(p):
	img = pil_image.open(expand_path(p))
	return img


# 图像矩阵变换函数, 返回numpy array
def build_transform(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
	"""
	Build a transformation matrix with the specified characteristics.
	"""
	rotation = np.deg2rad(rotation)
	shear = np.deg2rad(shear)
	rotation_matrix = np.array(
		[[np.cos(rotation), np.sin(rotation), 0], [-np.sin(rotation), np.cos(rotation), 0], [0, 0, 1]])
	# shift_matrix = np.array([[1, 0, height_shift], [0, 1, width_shift], [0, 0, 1]])
	shear_matrix = np.array([[1, np.sin(shear), 0], [0, np.cos(shear), 0], [0, 0, 1]])
	zoom_matrix = np.array([[1.0 / height_zoom, 0, 0], [0, 1.0 / width_zoom, 0], [0, 0, 1]])
	shift_matrix = np.array([[1, 0, -height_shift], [0, 1, -width_shift], [0, 0, 1]])
	return np.dot(np.dot(rotation_matrix, shear_matrix), np.dot(zoom_matrix, shift_matrix))


# 返回经过预处理图像的矩阵值
def read_cropped_image(p, augment):
	"""
	@param p : 要读取的图片的名称
	@param augment: 是否需要做图像增强
	@返回变换后的图像
	"""
	# 如果给出了图像ID，则转换为文件名
	if p in h2p:
		p = h2p[p]
	size_x, size_y = p2size[p]

	# 根据边界框确定要捕获的原始图像的区域
	row = p2bb.loc[p]  # 返回image的标定框值--DataFrame
	x0, y0, x1, y1 = row['x0'], row['y0'], row['x1'], row['y1']
	dx = x1 - x0
	dy = y1 - y0
	# 为边框添加空白值,重新设置(x0,x1,y0,y1)
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
	# 计算鲸鱼尾巴ROI的长宽值
	dx = x1 - x0
	dy = y1 - y0
	# 重新计算(x0,x1,y0,y1)，使得水平压缩比anisotropy等于初始化值2.15
	# dx过大,则增加dy
	if dx > dy * anisotropy:
		dy = 0.5 * (dx / anisotropy - dy)
		y0 -= dy
		y1 += dy
	# dy过大,则增加dx
	else:
		dx = 0.5 * (dy * anisotropy - dx)
		x0 -= dx
		x1 += dx

	# 生成随机变换矩阵(增强图像矩阵)
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

	# 读取图像,转化为numpy数组
	img = read_raw_image(p).convert('L')
	img = np.asarray(img)
	# print('original shape:',img.shape)
	# 应用放射变换, 矩阵计算实际在这里
	matrix = trans[:2, :2]
	offset = trans[:2, 2]
	img_affine = affine_transform(img, matrix, offset, order=1, output_shape=img_shape[:-1], mode='constant',
								cval=np.average(img))
	# print('after affine_transform shape:',img_affine.shape)
	# 将3个二维数组重叠为一个三维数组
	rgb_array = np.zeros((img_affine.shape[0], img_affine.shape[1], 3), "float64")
	rgb_array[:, :, 0], rgb_array[:, :, 1], rgb_array[:, :, 2] = img_affine, img_affine, img_affine
	img = np.asarray(rgb_array, 'f')

	# 归一化为零均值和方差
	img -= np.mean(img, keepdims=True)
	img /= np.std(img, keepdims=True) + K.epsilon()

	return img


def read_for_training(p):
	"""
	使用数据增强（随机变换）读取和预处理图像。
	"""
	return read_cropped_image(p, True)


def read_for_validation(p):
	"""
	在没有数据增强的情况下读取和预处理图像（用于测试）
	"""
	return read_cropped_image(p, False)


# ----------------------------------------------多线程lapjv------------------------------------------------
def my_lapjv(score):
	num_threads = 6
	batch = score.shape[0] // (num_threads - 1)
	if score.shape[0] % batch <= 3:
		num_threads = 5
		if score.shape[0] % batch is not 0:
			batch += 1
	# print(batch)
	tmp = num_threads * [None]
	threads = []
	thread_input = num_threads * [None]
	thread_idx = 0
	for start in range(0, score.shape[0], batch):
		end = min(score.shape[0], start + batch)
		# print('%d %d' % (start, end))
		thread_input[thread_idx] = score[start:end, start:end]
		thread_idx += 1

	def worker(data_idx):
		x, _, _ = lapjv(thread_input[data_idx])
		tmp[data_idx] = x + data_idx * batch

	# print("Start worker threads")
	for i in range(num_threads):
		t = threading.Thread(target=worker, args=(i,), daemon=True)
		t.start()
		threads.append(t)
	for t in threads:
		if t is not None:
			t.join()
	x = np.concatenate(tmp)
	# print("LAP completed")
	return x

# ------------------------------------------------数据生成器类定义-----------------------------------------------
class TrainingData(Sequence):
	# 类实例属性初始化方法
	def __init__(self, score, steps=1000, batch_size=32):
		"""
		@param score: 图片匹配的cost matrix
		@param steps: epoch数，用来设计score matrix
		"""
		super(TrainingData, self).__init__()
		self.score = -score  # 最大化分数与最小化负分数相同
		self.steps = steps
		self.batch_size = batch_size
		# 将鲸鱼id映射到相关的训练图片的hash表上去
		for ts in w2ts.values():
			idxs = [t2i[t] for t in ts]
			for i in idxs:
				for j in idxs:
					self.score[i, j] = 10000.0  # 为匹配鲸鱼设置一个很大的值 - 消除了这种潜在的配对
		self.on_epoch_end()

	# 定义获取一个batch数据的方法
	def __getitem__(self, index):
		"""Gets batch at position `index`.

		# Arguments
			index: position of the batch in the Sequence.

		# Returns
			A batch
		"""
		start = self.batch_size * index
		end = min(start + self.batch_size, len(self.match) + len(self.unmatch))
		size = end - start
		assert size > 0
		a = np.zeros((size,) + img_shape, dtype=K.floatx())
		b = np.zeros((size,) + img_shape, dtype=K.floatx())
		c = np.zeros((size, 1), dtype=K.floatx())
		j = start // 2
		for i in range(0, size, 2):
			a[i, :, :, :] = read_for_training(self.match[j][0])  # 数据增强
			b[i, :, :, :] = read_for_training(self.match[j][1])
			c[i, 0] = 1  # This is a match
			a[i + 1, :, :, :] = read_for_training(self.unmatch[j][0])
			b[i + 1, :, :, :] = read_for_training(self.unmatch[j][1])
			c[i + 1, 0] = 0  # Different whales
			j += 1
		return [a, b], c

	def on_epoch_end(self):
		"""Method called at the end of every epoch.
		"""
		if self.steps <= 0:
			return  # 跳过最后一个batch
		self.steps -= 1
		self.match = []
		self.unmatch = []
		x = my_lapjv(self.score)  # Solve the linear assignment problem
		y = np.arange(len(x), dtype=np.int32)

		# 计算匹配鲸鱼的derangement
		for ts in w2ts.values():
			d = ts.copy()
			while True:
				random.shuffle(d)
				if not np.any(ts == d): break
			for ab in zip(ts, d):
				self.match.append(ab)

		# Construct unmatched whale pairs from the LAP solution.
		for i, j in zip(x, y):
			if i == j:
				print(self.score)
				print(x)
				print(y)
				print(i, j)
			assert i != j
			self.unmatch.append((train[i], train[j]))

		# Force a different choice for an eventual next epoch.
		self.score[x, y] = 10000.0
		self.score[y, x] = 10000.0
		random.shuffle(self.match)
		random.shuffle(self.unmatch)
		# print(len(self.match), len(train), len(self.unmatch), len(train))
		assert len(self.match) == len(train) and len(self.unmatch) == len(train)

	# 定义获取长度的方法
	def __len__(self):
		"""Number of batch in the Sequence.

		# Returns
			The number of batches in the Sequence.
		"""
		return (len(self.match) + len(self.unmatch) + self.batch_size - 1) // self.batch_size


# # Test on a batch of 32 with random costs.
# score = np.random.random_sample(size=(len(train), len(train)))
# data = TrainingData(score)
# print(data)
# (a, b), c = data[0]
# print((a.shape, b.shape), c.shape)


class FeatureGen(Sequence):
	"""
	# Keras生成器，仅评估branch model
	"""

	def __init__(self, data, batch_size=64, verbose=1):
		super(FeatureGen, self).__init__()
		self.data = data
		self.batch_size = batch_size
		self.verbose = verbose
		if self.verbose > 0: self.progress = tqdm(total=len(self), desc='Features')

	def __getitem__(self, index):
		start = self.batch_size * index
		size = min(len(self.data) - start, self.batch_size)
		a = np.zeros((size,) + img_shape, dtype=K.floatx())
		for i in range(size): a[i, :, :, :] = read_for_validation(self.data[start + i])
		if self.verbose > 0:
			self.progress.update()
			if self.progress.n >= len(self): self.progress.close()
		return a

	def __len__(self):
		return (len(self.data) + self.batch_size - 1) // self.batch_size


class ScoreGen(Sequence):
	"""
	# Keras生成器，用于评估head model上已预先计算的特征。
	# 如果y为None，则仅计算cost matrix的上三角矩阵。
	"""

	def __init__(self, x, y=None, batch_size=2048, verbose=1):
		super(ScoreGen, self).__init__()
		self.x = x
		self.y = y
		self.batch_size = batch_size
		self.verbose = verbose
		if y is None:
			self.y = self.x
			self.ix, self.iy = np.triu_indices(x.shape[0], 1)
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
		a = self.y[self.iy[start:end], :]
		b = self.x[self.ix[start:end], :]
		if self.verbose > 0:
			self.progress.update()
			if self.progress.n >= len(self): self.progress.close()
		return [a, b]

	def __len__(self):
		return (len(self.ix) + self.batch_size - 1) // self.batch_size
